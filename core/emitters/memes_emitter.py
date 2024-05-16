from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from chex import ArrayTree
from qdax.core.containers.repertoire import Repertoire
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from core.emitters.es_novelty_archives import NoveltyArchive
from core.emitters.memes_fix_reset_emitter import (
    MEMESFixResetEmitter,
    MEMESFixResetEmitterState,
    added_repertoire,
)


@dataclass
class MEMESConfig:
    """Configuration for the MEMES emitter.

    Args:
        sample_number: num of samples for gradient estimate
        sample_sigma: std to sample the samples for gradient estimate
        sample_mirror: if True, use mirroring sampling
        sample_rank_norm: if True, use normalisation

        num_in_optimizer_steps: num gradient step per generation

        adam_optimizer: if True, use ADAM, if False, use SGD
        learning_rate
        l2_coefficient: coefficient for regularisation

        novelty_nearest_neighbors: number of nearest neighbors for
            novelty computation
        use_novelty_archive: if True use novelty archive for novelty
            (default is to use the content of the repertoire)
        use_novelty_fifo: if True use fifo archive for novelty
            (default is to use the content of the repertoire)
        fifo_size: size of the novelty fifo buffer if used

        proportion_explore: proportion of explore

        num_generations_stagnate: number of generations wihtout addition before reset
    """

    sample_number: int = 512
    sample_sigma: float = 0.02
    sample_mirror: bool = True
    sample_rank_norm: bool = True

    num_in_optimizer_steps: int = 1

    adam_optimizer: bool = True
    learning_rate: float = 0.01
    l2_coefficient: float = 0.0

    novelty_nearest_neighbors: int = 10
    use_novelty_archive: bool = False
    use_novelty_fifo: bool = False
    fifo_size: int = 100000

    proportion_explore: float = 0.5

    num_generations_stagnate: int = 2


class MEMESEmitterState(MEMESFixResetEmitterState):
    """Emitter State for the MEMES emitter.

    Args:
        initial_optimizer_state: stored to re-initialise when sampling new parent
        optimizer_state: current optimizer state
        offspring: offspring generated through gradient estimate
        generation_count: generation counter used to update the novelty archive
        novelty_archive: used to compute novelty for explore
        proportion_explore: proportion of explore and exploit
        emitter_adaptive_reset: number of generations since last addition
        random_key: key to handle stochastic operations
    """

    initial_optimizer_state: optax.OptState
    optimizer_states: ArrayTree
    offspring: Genotype
    generation_count: int
    novelty_archive: NoveltyArchive
    emitter_adaptive_reset: jnp.ndarray
    random_key: RNGKey


class MEMESEmitter(MEMESFixResetEmitter):
    """
    An emitter that uses gradients approximated through sampling.
    It dedicates part of the es process to fitness gradients and part to the
    novelty gradients and reset all emitters independently based on
    stagnate criteria.

    This scan version scans through parents instead of performing all es
    operations in parallell, to avoid memory overload issue.
    """

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def init(
        self,
        init_genotypes: Genotype,
        random_key: RNGKey,
    ) -> Tuple[MEMESEmitterState, RNGKey]:
        """Initializes the emitter state.

        Args:
            init_genotypes: The initial population.
            random_key: A random key.

        Returns:
            The initial state of the emitter, a new random key.
        """

        # Initialise optimizer
        params = jax.tree_util.tree_map(lambda x: x[0], init_genotypes)
        initial_optimizer_state = self._optimizer.init(params)

        # One optimizer_state per lineage
        optimizer_states = jax.tree_util.tree_map(
            lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), self._batch_size, axis=0),
            initial_optimizer_state,
        )

        # Empty Novelty archive
        novelty_archive = self._init_novelty_archive(
            self._batch_size, self._batch_size * self._config.sample_number
        )

        return (
            MEMESEmitterState(
                initial_optimizer_state=initial_optimizer_state,
                optimizer_states=optimizer_states,
                offspring=init_genotypes,
                generation_count=0,
                novelty_archive=novelty_archive,
                emitter_adaptive_reset=jnp.zeros(self._batch_size),
                random_key=random_key,
            ),
            random_key,
        )

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "batch_size",
        ),
    )
    def _sample_parents(
        self,
        emitter_state: MEMESEmitterState,
        repertoire: Repertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
        added: jnp.ndarray,
        batch_size: int,
    ) -> Tuple[MEMESEmitterState, Genotype, optax.OptState, RNGKey]:
        """Sample new parents."""

        random_key = emitter_state.random_key

        # Select between new sampled parents and previous parents
        emitter_adaptive_reset = jnp.where(
            added, 0, emitter_state.emitter_adaptive_reset + 1
        )
        reset_emitter = emitter_adaptive_reset > self._config.num_generations_stagnate
        (
            parents,
            random_key,
        ) = repertoire.sample(random_key, batch_size)
        parents = jax.tree_util.tree_map(
            lambda parent, genotype: jnp.where(
                jnp.reshape(
                    jnp.repeat(
                        reset_emitter,
                        parent[0].size,
                        total_repeat_length=parent[0].size * batch_size,
                    ),
                    parent.shape,
                ),
                parent,
                genotype,
            ),
            parents,
            genotypes,
        )
        optimizer_states = jax.tree_util.tree_map(
            lambda initial_opt_state, opt_state: jnp.where(
                jnp.reshape(
                    jnp.repeat(
                        reset_emitter,
                        initial_opt_state.size,
                        total_repeat_length=initial_opt_state.size * batch_size,
                    ),
                    opt_state.shape,
                ),
                initial_opt_state,
                opt_state,
            ),
            emitter_state.initial_optimizer_state,
            emitter_state.optimizer_states,
        )
        emitter_adaptive_reset = jnp.where(reset_emitter, 0, emitter_adaptive_reset)

        return emitter_state, parents, optimizer_states, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: MEMESEmitterState,
        repertoire: Repertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> MEMESEmitterState:
        """
        Update the novelty archive and generation count from current call.
        Generate the gradient offsprings for the next emitter call.
        """
        assert emitter_state is not None, "\n!!!ERROR!!! No emitter state."

        # Check if offspring have been added to repertoire
        added = added_repertoire(genotypes, descriptors, repertoire)

        # Update novelty archive
        generation_count = emitter_state.generation_count
        novelty_archive = emitter_state.novelty_archive.update(
            descriptors, repertoire.descriptors, repertoire.fitnesses
        )
        emitter_state = emitter_state.replace(novelty_archive=novelty_archive)

        # Get parents
        (emitter_state, parents, optimizer_states, random_key,) = self._sample_parents(
            emitter_state,
            repertoire,
            genotypes,
            fitnesses,
            descriptors,
            {},
            added,
            self._batch_size,
        )

        # Apply num_in_optimizer_steps steps of gradients to compute the offspring
        @jax.jit
        def _apply_step(
            carry: Tuple[Genotype, optax.OptState, RNGKey],
            unused: Tuple[()],
            repertoire: Repertoire,
            emitter_state: MEMESEmitterState,
        ) -> Tuple[Tuple[Genotype, optax.OptState, RNGKey], Tuple[()]]:
            parents, optimizer_states, random_key = carry
            parents, optimizer_states, random_key = self._optimizer_step(
                parents=parents,
                optimizer_states=optimizer_states,
                random_key=random_key,
                repertoire=repertoire,
                emitter_state=emitter_state,
            )
            return (parents, optimizer_states, random_key), ()

        apply_step_fn = partial(
            _apply_step,
            repertoire=repertoire,
            emitter_state=emitter_state,
        )
        (offspring, optimizer_states, random_key), () = jax.lax.scan(
            apply_step_fn,
            (parents, optimizer_states, random_key),
            (),
            length=self._config.num_in_optimizer_steps,
        )

        # Increase generation counter
        generation_count += 1

        return emitter_state.replace(  # type: ignore
            optimizer_states=optimizer_states,
            offspring=offspring,
            generation_count=generation_count,
            novelty_archive=novelty_archive,
            random_key=random_key,
        )

from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from chex import ArrayTree
from qdax.core.containers.repertoire import Repertoire
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from core.emitters.es_novelty_archives import (
    EmptyNoveltyArchive,
    NoveltyArchive,
    ParallelNoveltyArchive,
    RepertoireNoveltyArchive,
    SequentialNoveltyArchive,
    SequentialScanNoveltyArchive,
)
from core.emitters.memes_emitter import added_repertoire
from core.emitters_metrics.metrics_es_base_emitter import ESEmitter, ESEmitterState


@dataclass
class MEMESSequentialConfig:
    """Configuration for the MEMESSequential emitter.

    Args:
        sample_number: num of samples for gradient estimate
        sample_sigma: std to sample the samples for gradient estimate
        sample_mirror: if True, use mirroring sampling
        sample_rank_norm: if True, use normalisation

        num_generations_sample: frequency of archive-sampling
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

        use_explore: if False, use only fitness gradient
        use_exploit: if False, use only novelty gradient
    """

    sample_number: int = 512
    sample_sigma: float = 0.02
    sample_mirror: bool = True
    sample_rank_norm: bool = True

    num_generations_sample: int = 10
    num_in_optimizer_steps: int = 1

    adam_optimizer: bool = True
    learning_rate: float = 0.01
    l2_coefficient: float = 0.0

    novelty_nearest_neighbors: int = 10
    use_novelty_archive: bool = False
    use_novelty_fifo: bool = False
    fifo_size: int = 100000

    use_explore: bool = True
    use_exploit: bool = True


class MEMESSequentialEmitterState(ESEmitterState):
    """Emitter State for the MEMESSequential emitter.

    Args:
        initial_optimizer_state: stored to re-initialise when sampling new parent
        optimizer_state: current optimizer state
        offspring: offspring generated through gradient estimate
        generation_count: generation counter used to update the novelty archive
        novelty_archive: used to compute novelty for explore
        explore_usage: proportion of explore offspring added to repertoire
        exploit_usage: proportion of exploit offspring added to repertoire
        parents_descriptors: parents descriptors store to compute parents_distance
        parents_distance: distance to parents in BD space
        random_key: key to handle stochastic operations
    """

    initial_optimizer_state: optax.OptState
    optimizer_states: ArrayTree
    offspring: Genotype
    generation_count: int
    novelty_archive: NoveltyArchive
    explore_usage: int
    exploit_usage: int
    parents_descriptors: Descriptor
    parents_distance: float
    random_key: RNGKey


class MEMESSequentialEmitter(ESEmitter):
    """
    An emitter that uses gradients approximated through sampling.
    It either alternates between num_generations_sample of fitness
    gradients and num_generations_sample of novelty gradients.

    This scan version scans through parents instead of performing all es
    operations in parallell, to avoid memory overload issue.
    """

    def __init__(
        self,
        config: MEMESSequentialConfig,
        batch_size: int,
        scoring_fn: Callable[
            [Genotype, RNGKey],
            Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
        ],
        num_descriptors: int,
        scan_batch_size: int = 1,
        scan_novelty: int = 1,
        total_generations: int = 1,
        num_centroids: int = 1,
    ) -> None:
        """Initialise the emitter.

        Args:
            config
            batch_size: number of individuals generated per generation.
            scoring_fn: used to evaluate the samples for the gradient estimate.
            num_descriptors: dimension of the descriptors, used to initialise
                metrics and the empty novelty archive.
            total_generations: total number of generations for which the
                emitter will run, allow to initialise the novelty archive.
        """
        assert (
            batch_size % scan_batch_size == 0 or scan_batch_size > batch_size
        ), "!!!ERROR!!! Batch-size should be dividible by scan-batch-size."
        total_samples = batch_size * config.sample_number
        assert (
            total_samples % scan_novelty == 0 or scan_novelty > total_samples,
        ), "!!!ERROR!!! Total number of samples should be dividible by scan-novelty."

        # Set up config
        self._config = config

        # Uniformise use_explore and use_exploit
        assert (
            self._config.use_explore or self._config.use_exploit
        ), "!!!ERROR!!! Cannot use neither explore nor exloit."

        # Set up other parameters
        self._batch_size = batch_size
        self._scoring_fn = scoring_fn
        self._num_descriptors = num_descriptors
        self._scan_batch_size = (
            scan_batch_size if batch_size > scan_batch_size else batch_size
        )
        self._scan_novelty = (
            scan_novelty if total_samples > scan_novelty else total_samples
        )
        self._num_scan = self._batch_size // self._scan_batch_size
        self._total_generations = total_generations
        self._num_centroids = num_centroids
        assert not (
            self._config.use_novelty_archive and self._config.use_novelty_fifo
        ), "!!!ERROR!!! Use both novelty archive and novelty fifo."

        self._non_scan_explore = jnp.zeros(self._batch_size)
        self._explore = jnp.zeros((self._num_scan))

        # Initialise optimizer
        if self._config.adam_optimizer:
            self._optimizer = optax.adam(learning_rate=config.learning_rate)
        else:
            self._optimizer = optax.sgd(learning_rate=config.learning_rate)

    @partial(
        jax.jit,
        static_argnames=("self", "batch_size", "novelty_batch_size"),
    )
    def _init_novelty_archive(
        self, batch_size: int, novelty_batch_size: int
    ) -> NoveltyArchive:
        """Init the novelty archive for the emitter."""

        if self._config.use_explore and self._config.use_novelty_archive:
            novelty_archive = SequentialNoveltyArchive.init(
                self._total_generations * batch_size, self._num_descriptors
            )
        elif (
            self._config.use_explore
            and self._config.use_novelty_fifo
            and self._scan_novelty < novelty_batch_size
        ):
            novelty_archive = SequentialScanNoveltyArchive.init(
                self._config.fifo_size,
                self._num_descriptors,
                scan_size=self._scan_novelty,
            )
        elif (
            self._config.use_explore
            and self._config.use_novelty_fifo
            and self._scan_novelty == 1
        ):
            novelty_archive = SequentialNoveltyArchive.init(
                self._config.fifo_size, self._num_descriptors
            )
        elif self._config.use_explore and self._config.use_novelty_fifo:
            novelty_archive = ParallelNoveltyArchive.init(
                self._config.fifo_size, self._num_descriptors
            )
        elif self._config.use_explore:
            novelty_archive = RepertoireNoveltyArchive.init(
                self._num_centroids, self._num_descriptors
            )
        else:
            novelty_archive = EmptyNoveltyArchive.init(1, 1)

        return novelty_archive

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def init(
        self,
        init_genotypes: Genotype,
        random_key: RNGKey,
    ) -> Tuple[MEMESSequentialEmitterState, RNGKey]:
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
            MEMESSequentialEmitterState(
                initial_optimizer_state=initial_optimizer_state,
                optimizer_states=optimizer_states,
                offspring=init_genotypes,
                generation_count=0,
                explore_usage=0,
                exploit_usage=0,
                parents_descriptors=jnp.zeros(
                    (self._batch_size, self._num_descriptors)
                ),
                parents_distance=0,
                novelty_archive=novelty_archive,
                random_key=random_key,
            ),
            random_key,
        )

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def _sub_scores(
        self,
        explore: jnp.ndarray,
        emitter_state: MEMESSequentialEmitterState,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> jnp.ndarray:
        """Compute the score from evaluation."""

        # If sequential explore and exploit emitters
        if self._config.use_explore:
            return jax.lax.cond(
                (emitter_state.generation_count // self._config.num_generations_sample)
                % 2
                == 0,
                lambda scoring: emitter_state.novelty_archive.novelty(
                    scoring[1],
                    self._config.novelty_nearest_neighbors,
                ),
                lambda scoring: scoring[0],
                (fitnesses, descriptors),
            )

        # If no explore return fitness
        return fitnesses

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "batch_size",
        ),
    )
    def _sample_parents(
        self,
        emitter_state: MEMESSequentialEmitterState,
        repertoire: Repertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
        added: jnp.ndarray,
        batch_size: int,
    ) -> Tuple[
        MEMESSequentialEmitterState, Genotype, Descriptor, optax.OptState, RNGKey
    ]:
        """Sample new parents."""

        random_key = emitter_state.random_key

        # Sample new parents every num_generations_sample generations
        sample_parents = (
            emitter_state.generation_count % self._config.num_generations_sample == 0
        )
        parents, parents_descriptors, parents_fitnesses, random_key = jax.lax.cond(
            sample_parents,
            lambda random_key: repertoire.sample_full(random_key, batch_size),
            lambda random_key: (genotypes, descriptors, fitnesses, random_key),
            (random_key),
        )

        # Same for optimizer_states
        optimizer_states = jax.lax.cond(
            sample_parents,
            lambda unused: jax.tree_util.tree_map(
                lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), batch_size, axis=0),
                emitter_state.initial_optimizer_state,
            ),
            lambda unused: emitter_state.optimizer_states,
            (),
        )

        return emitter_state, parents, optimizer_states, parents_descriptors, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: MEMESSequentialEmitterState,
        repertoire: Repertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> MEMESSequentialEmitterState:
        """
        Update the novelty archive and generation count from current call.
        Generate the gradient offsprings for the next emitter call.
        """
        assert emitter_state is not None, "\n!!!ERROR!!! No emitter state."

        # Check if offspring have been added to repertoire
        added = added_repertoire(genotypes, descriptors, repertoire)

        # Update usage metrics
        proportion_explore = (
            emitter_state.generation_count // self._config.num_generations_sample
        ) % 2 == 0
        explore_usage, exploit_usage = jax.lax.cond(
            proportion_explore,
            lambda added: (jnp.sum(added[0]), 0),
            lambda added: (0, jnp.sum(added[0])),
            (added),
        )

        # Update parents_distance metrics
        parents_distance = jnp.average(
            jnp.sqrt(
                jnp.sum(
                    jnp.square(emitter_state.parents_descriptors - descriptors), axis=1
                )
            ),
            axis=0,
        )

        # Update novelty archive
        generation_count = emitter_state.generation_count
        novelty_archive = emitter_state.novelty_archive.update(
            descriptors, repertoire.descriptors, repertoire.fitnesses
        )
        emitter_state = emitter_state.replace(novelty_archive=novelty_archive)

        # Get parents
        (
            emitter_state,
            parents,
            optimizer_states,
            parents_descriptors,
            random_key,
        ) = self._sample_parents(
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
            emitter_state: MEMESSequentialEmitterState,
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
            explore_usage=explore_usage,
            exploit_usage=exploit_usage,
            parents_descriptors=parents_descriptors,
            parents_distance=parents_distance,
            random_key=random_key,
        )

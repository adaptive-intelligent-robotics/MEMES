from functools import partial
from math import floor
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from chex import ArrayTree
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from core.emitters.es_novelty_archives import NoveltyArchive
from core.emitters.memes_emitter import MEMESConfig, MEMESEmitter, added_repertoire


class MEMESAllEmitterState(EmitterState):
    """Emitter State for the ES emitter.

    Args:
        initial_optimizer_state: stored to re-initialise when sampling new parent
        optimizer_state: current optimizer state
        generation_count: generation counter used to update the novelty archive
        novelty_archive: used to compute novelty for explore
        emitter_adaptive_reset: number of generations since last addition
        random_key: key to handle stochastic operations
    """

    initial_optimizer_state: optax.OptState
    optimizer_states: ArrayTree

    parents: Genotype
    gradient_noises: Genotype

    generation_count: int
    novelty_archive: NoveltyArchive

    emitter_adaptive_reset: jnp.ndarray
    random_key: RNGKey


class MEMESAllEmitter(MEMESEmitter):
    """
    An emitter that uses gradients approximated through sampling.
    Consider all generated individuals for addition to the archive.

    !!!WARNING!!! This emitter requires a specific type of container
    as it relies on 2 disinct calls to the emitter to improve the
    vanilla all-es emitter and return synchronised gradient offspring.
    """

    def __init__(
        self,
        config: MEMESConfig,
        batch_size: int,
        num_descriptors: int,
        scan_novelty: int = 1,
        total_generations: int = 1,
        num_centroids: int = 1,
    ) -> None:
        """Initializes the emitter.

        Args:
            config
            total_generations: total number of generations for which the
                emitter will run, necessary to set up the novelty archive.
            batch_size: number of individuals generated per generation.
            num_descriptors: dimension of the descriptors, used to initialise
                the empty novelty archive.

        Returns: /
        """

        # Divide the batch between gradient-generated offsprings and es samples
        assert (1 + config.sample_number) < batch_size, (
            "!!!ERROR!!! Num samples for the es: "
            + str(config.sample_number)
            + " too high for batch-size: "
            + str(batch_size)
        )
        self._gradient_batch_size = batch_size // (1 + config.sample_number)
        print("self._gradient_batch_size", self._gradient_batch_size)
        self._samples_batch_size = self._gradient_batch_size * config.sample_number
        print("self._samples_batch_size", self._samples_batch_size)
        self._additional_batch_size = batch_size % (1 + config.sample_number)
        print("self._additional_batch_size", self._additional_batch_size)

        total_samples = batch_size * config.sample_number
        assert (
            total_samples % scan_novelty == 0 or scan_novelty > total_samples,
        ), "!!!ERROR!!! Total number of samples should be dividible by scan-novelty."

        # Set up config
        self._config = config
        self._config.use_explore = self._config.proportion_explore > 0

        # Uniformise use_explore use_exploit and proportion_explore
        assert (
            self._config.use_explore or self._config.use_exploit
        ), "!!!ERROR!!! Cannot use neither explore nor exloit."

        # IMPORTANT allow to use ES gradient functions
        self._scan_batch_size = self._gradient_batch_size

        # Set up other parameters
        self._batch_size = batch_size
        self._num_descriptors = num_descriptors
        self._scan_novelty = (
            scan_novelty if total_samples > scan_novelty else total_samples
        )
        self._total_generations = total_generations
        self._num_centroids = num_centroids
        assert not (
            self._config.use_novelty_archive and self._config.use_novelty_fifo
        ), "!!!ERROR!!! Use both novelty archive and novelty fifo."

        number_explore = floor(
            self._gradient_batch_size * self._config.proportion_explore
        )
        self._non_scan_explore = jnp.concatenate(
            [
                jnp.ones(number_explore),
                jnp.zeros(self._gradient_batch_size - number_explore),
            ],
            axis=0,
        )
        self._explore = jnp.repeat(
            self._non_scan_explore, self._config.sample_number, axis=0
        )
        self._explore = jnp.expand_dims(self._explore, axis=0)

        # Initialise optimizer
        if self._config.adam_optimizer:
            self._optimizer = optax.adam(learning_rate=config.learning_rate)
        else:
            self._optimizer = optax.sgd(learning_rate=config.learning_rate)

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[MEMESAllEmitterState, RNGKey]:
        """Initializes the emitter state.

        Args:
            init_genotypes: The initial population.
            random_key: A random key.

        Returns:
            The initial state of the emitter, a new random key.
        """

        # Initialise parents and gradient offsprings
        genotypes = jax.tree_map(
            lambda x: x[: self._gradient_batch_size], init_genotypes
        )

        # Initialise samples
        if self._config.sample_mirror:
            sample_number = self._config.sample_number // 2
        else:
            sample_number = self._config.sample_number
        gradient_noises = jax.tree_map(
            lambda x: jnp.repeat(jnp.zeros_like(x), sample_number, axis=0), genotypes
        )

        # Initialise optimizer
        params = jax.tree_util.tree_map(lambda x: x[0], init_genotypes)
        initial_optimizer_state = self._optimizer.init(params)

        # One optimizer_state per lineage
        optimizer_states = jax.tree_util.tree_map(
            lambda x: jnp.repeat(
                jnp.expand_dims(x, axis=0), self._gradient_batch_size, axis=0
            ),
            initial_optimizer_state,
        )

        # Empty Novelty archive
        novelty_archive = self._init_novelty_archive(
            self._gradient_batch_size, self._samples_batch_size
        )

        return (
            MEMESAllEmitterState(
                initial_optimizer_state=initial_optimizer_state,
                optimizer_states=optimizer_states,
                generation_count=0,
                novelty_archive=novelty_archive,
                emitter_adaptive_reset=jnp.zeros(self._gradient_batch_size),
                random_key=random_key,
                parents=genotypes,
                gradient_noises=gradient_noises,
            ),
            random_key,
        )

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        unused_emitter_state: MEMESAllEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        Empty function to ensure everything works as expected.
        """
        random_key, subkey = jax.random.split(random_key)
        assert 0, "!!!ERRROR!!! SHOULD NOT BE HERE"

        return jnp.zeros(1), random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: MEMESAllEmitterState,
        repertoire: Repertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> MEMESAllEmitterState:
        """
        First call to state_update, with full batch-size of random indiv.
        """

        assert emitter_state is not None, "\n!!!ERROR!!! No emitter state."
        return emitter_state

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit_first(
        self,
        repertoire: Repertoire,
        emitter_state: MEMESAllEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, MEMESAllEmitterState, RNGKey]:
        """
        Return the samples needed to estimate the gradient.

        Params:
            repertoire: the MAP-Elites repertoire to sample from
            emitter_state
            random_key: a jax PRNG random key

        Returns:
            a batch of offsprings
            a new jax PRNG key
        """

        assert emitter_state is not None, "\n!!!ERROR!!! No emitter state."

        # Creating samples for gradient estimate
        gradient_noises, random_key = self._pre_es_noise(
            emitter_state.parents, emitter_state, random_key
        )
        emitter_state = emitter_state.replace(gradient_noises=gradient_noises)

        # Get samples
        samples, random_key = self._pre_es_apply(
            emitter_state.parents, gradient_noises, emitter_state, random_key
        )

        # Sampling additional noise offsprings to complete batch-size
        if self._additional_batch_size > 0:
            parents, random_key = repertoire.sample(
                random_key, self._additional_batch_size
            )
            random_key, subkey = jax.random.split(random_key)
            samples_additional = jax.tree_map(
                lambda x: x
                + self._config.sample_sigma
                * jax.random.normal(key=subkey, shape=x.shape),
                parents,
            )

            # Concatenating all noise offsprings
            offspring = jax.tree_map(
                lambda sample, sample_add: jnp.concatenate(
                    [sample, sample_add], axis=0
                ),
                samples,
                samples_additional,
            )
        else:
            offspring = samples

        return offspring, emitter_state, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit_second(
        self,
        repertoire: Repertoire,
        emitter_state: MEMESAllEmitterState,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
        random_key: RNGKey,
    ) -> Tuple[Genotype, MEMESAllEmitterState, RNGKey]:
        """
        Return the offspring generated through gradient update.

        Params:
            repertoire: the MAP-Elites repertoire to sample from
            emitter_state: void
            random_key: a jax PRNG random key

        Returns:
            a batch of offsprings
            a new jax PRNG key
        """

        assert emitter_state is not None, "\n!!!ERROR!!! No emitter state."

        # Remove additional samples to compute gradient
        samples_fitnesses = fitnesses[: self._samples_batch_size]
        samples_descriptors = descriptors[: self._samples_batch_size]

        # Get scores
        scores = self._sub_scores(
            self._explore,
            emitter_state,
            samples_fitnesses,
            samples_descriptors,
            {},
        )

        # Compute gradient offsprings
        gradients, random_key = self._post_es_emitter(
            emitter_state.parents,
            scores,
            emitter_state.gradient_noises,
            emitter_state,
            random_key,
        )
        gradient_offspring, optimizer_states = jax.vmap(self._apply_optimizer)(
            emitter_state.parents, gradients, emitter_state.optimizer_states
        )
        emitter_state = emitter_state.replace(optimizer_states=optimizer_states)

        return gradient_offspring, emitter_state, random_key

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "batch_size",
        ),
    )
    def _sample_parents(
        self,
        emitter_state: MEMESAllEmitterState,
        repertoire: Repertoire,
        genotypes: Genotype,
        added: jnp.ndarray,
        batch_size: int,
    ) -> Tuple[MEMESAllEmitterState, Genotype, optax.OptState, RNGKey]:
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

        emitter_state = emitter_state.replace(
            emitter_adaptive_reset=emitter_adaptive_reset,
        )

        return emitter_state, parents, optimizer_states, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update_second(
        self,
        emitter_state: MEMESAllEmitterState,
        repertoire: Repertoire,
        gradient_repertoire: Repertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> MEMESAllEmitterState:
        """
        Update the novelty archive and generation count.
        """
        assert emitter_state is not None, "\n!!!ERROR!!! No emitter state."

        # Check if offspring have been added to repertoire
        added = added_repertoire(genotypes, descriptors, repertoire)

        # Update novelty archive
        generation_count = emitter_state.generation_count
        novelty_archive = emitter_state.novelty_archive.update(
            descriptors, repertoire.descriptors, repertoire.fitnesses
        )

        # Get parents
        gradient_added = added_repertoire(genotypes, descriptors, gradient_repertoire)
        (emitter_state, parents, optimizer_states, random_key,) = self._sample_parents(
            emitter_state,
            repertoire,
            genotypes,
            gradient_added,
            self._gradient_batch_size,
        )

        # Increase generation counter
        generation_count += 1

        return emitter_state.replace(  # type: ignore
            generation_count=generation_count,
            novelty_archive=novelty_archive,
            random_key=random_key,
            parents=parents,
            optimizer_states=optimizer_states,
        )

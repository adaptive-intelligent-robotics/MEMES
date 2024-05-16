from functools import partial
from math import floor
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from qdax.core.containers.repertoire import Repertoire
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from core.emitters.memes_emitter import added_repertoire
from core.emitters_metrics.metrics_memes_emitter import (
    MEMESConfig,
    MEMESEmitter,
    MEMESEmitterState,
)


class MEMESGAEmitter(MEMESEmitter):
    """
    An emitter that uses gradients approximated through sampling and GA for exploration.
    """

    def __init__(
        self,
        config: MEMESConfig,
        mutation_fn: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_fn: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_percentage: float,
        batch_size: int,
        scoring_fn: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        num_descriptors: int,
        scan_batch_size: int = 1,
    ) -> None:
        """Initializes the emitter.

        Args:
            config
            mutation_fn: mutation operator for the GA.
            variation_fn: variation operator for the GA.
            variation_percentage: pourcentage of variation for the GA.
            batch_size: number of individuals generated per generation.
            scoring_fn: used to evaluate the samples for the gradient estimate.
            num_descriptors: dimension of the descriptors, used to initialise
                metrics of BD distance.

        Returns: /
        """

        # Set up config
        self._config = config

        # Set up other parameters
        self._mutation_fn = mutation_fn
        self._variation_fn = variation_fn
        self._variation_percentage = variation_percentage
        self._batch_size = batch_size
        self._scoring_fn = scoring_fn
        self._num_descriptors = num_descriptors

        # Set up scan batch size
        self._number_explore = floor(self._batch_size * self._config.proportion_explore)
        self._number_exploit = self._batch_size - self._number_explore
        self._scan_batch_size = (
            scan_batch_size
            if self._number_exploit > scan_batch_size
            else self._number_exploit
        )
        assert self._number_exploit % self._scan_batch_size == 0, (
            "!!!ERROR!!! Gradient-batch-size: "
            + str(self._number_exploit)
            + " should be dividible by scan-batch-size."
        )
        self._num_scan = self._number_exploit // self._scan_batch_size

        # Create the score repartition based on proportion_explore
        self._non_scan_explore = jnp.concatenate(
            [
                jnp.ones(self._number_explore),
                jnp.zeros(self._number_exploit),
            ],
            axis=0,
        )
        self._explore = jnp.repeat(
            jnp.zeros(self._number_exploit), self._config.sample_number, axis=0
        )
        self._explore = jnp.reshape(self._explore, (self._num_scan, -1))

        # Optimizer
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
            lambda x: jnp.repeat(
                jnp.expand_dims(x, axis=0), self._number_exploit, axis=0
            ),
            initial_optimizer_state,
        )

        # Initial offspring
        offspring = jax.tree_util.tree_map(
            lambda x: x[: self._number_exploit], init_genotypes
        )

        return (
            MEMESEmitterState(
                initial_optimizer_state=initial_optimizer_state,
                optimizer_states=optimizer_states,
                offspring=offspring,
                generation_count=0,
                novelty_archive=None,
                explore_usage=0,
                exploit_usage=0,
                parents_descriptors=jnp.zeros(
                    (self._number_exploit, self._num_descriptors)
                ),
                parents_distance=0,
                emitter_adaptive_reset=jnp.zeros(self._number_exploit),
                emitter_gen_reset_count=jnp.zeros(self._number_exploit),
                emitter_max_gen_reset_count=jnp.zeros(self._number_exploit),
                emitter_mean_gen_reset_count=jnp.zeros(self._number_exploit),
                emitter_n_reset_count=jnp.zeros(self._number_exploit),
                random_key=random_key,
            ),
            random_key,
        )

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def _ga_step(
        self,
        repertoire: Repertoire,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Apply one step of ga to the parents.

        Args:
            repertoire: current repertoire, used to compute scores
            random_key

        Returns:
            the gradients to apply and a new random key
        """

        n_variation = int(self._number_explore * self._variation_percentage)
        n_mutation = self._number_explore - n_variation

        if n_variation > 0:
            x1, random_key = repertoire.sample(random_key, n_variation)
            x2, random_key = repertoire.sample(random_key, n_variation)

            x_variation, random_key = self._variation_fn(x1, x2, random_key)

        if n_mutation > 0:
            x1, random_key = repertoire.sample(random_key, n_mutation)
            x_mutation, random_key = self._mutation_fn(x1, random_key)

        if n_variation == 0:
            offspring = x_mutation
        elif n_mutation == 0:
            offspring = x_variation
        else:
            offspring = jax.tree_map(
                lambda x_1, x_2: jnp.concatenate([x_1, x_2], axis=0),
                x_variation,
                x_mutation,
            )

        return offspring, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def emit(
        self,
        repertoire: Repertoire,
        emitter_state: MEMESEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        Return the offspring generated through gradient update, and
        add the offspring generated through GA.

        Params:
            repertoire: the MAP-Elites repertoire to sample from
            emitter_state
            random_key: a jax PRNG random key

        Returns:
            a batch of offsprings
            a new jax PRNG key
        """

        assert emitter_state is not None, "\n!!!ERROR!!! No emitter state."

        explore_offspring, random_key = self._ga_step(repertoire, random_key)

        # Concatenating gradient-generated offspring
        offspring = jax.tree_map(
            lambda explore_off, exploit_off: jnp.concatenate(
                [explore_off, exploit_off], axis=0
            ),
            explore_offspring,
            emitter_state.offspring,
        )

        return offspring, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def _scores(
        self,
        samples: Genotype,
        explore: jnp.ndarray,
        repertoire: Repertoire,
        emitter_state: MEMESEmitterState,
        random_key: RNGKey,
    ) -> Tuple[jnp.ndarray, RNGKey]:
        """Compute the scores associated with each sample.

        Args:
            samples: generated samples
            explore: repartition of explore and exploit emitters
            repertoire: current repertoire
            emitter_state
            random_key

        Returns:
            the gradients to apply and a new random key
        """

        # Evaluate samples
        fitnesses, descriptors, _, random_key = self._scoring_fn(
            samples,
            random_key,
        )

        # Return the fitnesses as scores
        return fitnesses, random_key

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
        Update the generation count.
        Generate the gradient offsprings for the next emitter call.
        """
        assert emitter_state is not None, "\n!!!ERROR!!! No emitter state."

        # Update usage metrics
        added = added_repertoire(genotypes, descriptors, repertoire)
        explore_usage = jnp.sum(jnp.where(self._non_scan_explore, added, 0))
        exploit_usage = jnp.sum(jnp.where(self._non_scan_explore, 0, added))

        # Update parents_distance metrics
        parents_distance = jnp.average(
            jnp.sqrt(
                jnp.sum(
                    jnp.square(
                        emitter_state.parents_descriptors
                        - descriptors[self._number_explore :]
                    ),
                    axis=1,
                )
            ),
            axis=0,
        )

        # Sample new parents
        random_key = emitter_state.random_key

        # Select between new sampled parents and previous parents
        generation_count = emitter_state.generation_count
        (
            emitter_state,
            parents,
            optimizer_states,
            parents_descriptors,
            random_key,
        ) = self._sample_parents(
            emitter_state,
            repertoire,
            emitter_state.offspring,
            fitnesses[self._number_explore :],
            descriptors[self._number_explore :],
            {},
            added[self._number_explore :],
            self._number_exploit,
        )

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
            explore_usage=explore_usage,
            exploit_usage=exploit_usage,
            parents_descriptors=parents_descriptors,
            parents_distance=parents_distance,
            random_key=random_key,
        )

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from core.emitters.memes_emitter import added_repertoire


class UsageEmitterState(EmitterState):
    usage: float

    parents: Genotype
    parents_descriptors: Descriptor
    parents_distance: float

    random_key: RNGKey


class MixingEmitter(Emitter):
    def __init__(
        self,
        mutation_fn: Callable[[Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_fn: Callable[[Genotype, Genotype, RNGKey], Tuple[Genotype, RNGKey]],
        variation_percentage: float,
        batch_size: int,
        behavior_descriptor_length: float,
    ) -> None:
        self._mutation_fn = mutation_fn
        self._variation_fn = variation_fn
        self._variation_percentage = variation_percentage
        self._batch_size = batch_size
        self._behavior_descriptor_length = behavior_descriptor_length

    def init(
        self, init_genotypes: Optional[Genotype], random_key: RNGKey
    ) -> Tuple[Optional[EmitterState], RNGKey]:
        return (
            UsageEmitterState(
                usage=0,
                parents=init_genotypes,
                parents_descriptors=jnp.zeros(
                    (self._batch_size, self._behavior_descriptor_length)
                ),
                parents_distance=0,
                random_key=random_key,
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
        emitter_state: Optional[UsageEmitterState],
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        Emitter that performs both mutation and variation. Two batches of
        variation_percentage * batch_size genotypes are sampled in the repertoire,
        copied and cross-over to obtain new offsprings. One batch of
        (1.0 - variation_percentage) * batch_size genotypes are sampled in the
        repertoire, copied and mutated.

        Note: this emitter has no state. A fake none state must be added
        through a function redefinition to make this emitter usable with MAP-Elites.

        Params:
            repertoire: the MAP-Elites repertoire to sample from
            emitter_state: void
            random_key: a jax PRNG random key

        Returns:
            a batch of offsprings
            a new jax PRNG key
        """
        n_variation = int(self._batch_size * self._variation_percentage)
        n_mutation = self._batch_size - n_variation

        if n_variation > 0:
            x1 = jax.tree_util.tree_map(
                lambda x: x[:n_variation], emitter_state.parents
            )
            x2, random_key = repertoire.sample(random_key, n_variation)

            x_variation, random_key = self._variation_fn(x1, x2, random_key)

        if n_mutation > 0:
            x1 = jax.tree_util.tree_map(
                lambda x: x[n_variation:], emitter_state.parents
            )
            x_mutation, random_key = self._mutation_fn(x1, random_key)

        if n_variation == 0:
            genotypes = x_mutation
        elif n_mutation == 0:
            genotypes = x_variation
        else:
            genotypes = jax.tree_util.tree_map(
                lambda x_1, x_2: jnp.concatenate([x_1, x_2], axis=0),
                x_variation,
                x_mutation,
            )

        return genotypes, random_key

    @partial(jax.jit, static_argnames=("self",))
    def state_update(
        self,
        emitter_state: Optional[UsageEmitterState],
        repertoire: Optional[Repertoire] = None,
        genotypes: Optional[Genotype] = None,
        fitnesses: Optional[Fitness] = None,
        descriptors: Optional[Descriptor] = None,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Optional[UsageEmitterState]:

        # Update metrics
        usage = jnp.sum(added_repertoire(genotypes, descriptors, repertoire))
        parents_distance = jnp.average(
            jnp.sqrt(
                jnp.sum(
                    jnp.square(emitter_state.parents_descriptors - descriptors),
                    axis=1,
                )
            ),
            axis=0,
        )

        # Sample parents
        parents, parents_descriptors, _, random_key = repertoire.sample_full(
            emitter_state.random_key, self._batch_size
        )

        return emitter_state.replace(
            usage=usage,
            parents=parents,
            parents_descriptors=parents_descriptors,
            parents_distance=parents_distance,
            random_key=random_key,
        )

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._batch_size

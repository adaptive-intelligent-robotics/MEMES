"""Core components of the MAP-Elites algorithm."""
from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.map_elites import MAPElites
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)


class MEMESAll(MAPElites):
    """
    MAP-Elites algorithm with 2 calls to emitter and to add, before and after
    emitter update.
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        center_scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
    ) -> None:
        self._scoring_function = scoring_function
        self._center_scoring_function = center_scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        init_genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[
        MapElitesRepertoire, MapElitesRepertoire, Optional[EmitterState], RNGKey
    ]:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
            init_genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            random_key: a random key used for stochastic operations.

        Returns:
            An initialized MAP-Elite repertoire with the initial state of the emitter,
            and a random key.
        """

        # score initial genotypes
        fitnesses, descriptors, _, random_key = self._scoring_function(
            init_genotypes, random_key
        )

        # init the repertoire
        repertoire = MapElitesRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores={},
        )

        # init the gradient-only repertoire (used to ensure behaviour of emitter reset)
        gradient_repertoire = MapElitesRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores={},
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores={},
        )

        return repertoire, gradient_repertoire, emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: MapElitesRepertoire,
        gradient_repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[
        MapElitesRepertoire,
        MapElitesRepertoire,
        Optional[EmitterState],
        Metrics,
        RNGKey,
    ]:
        """
        Performs one iteration of the MAP-Elites algorithm.
        1. A batch of genotypes is sampled in the archive and the genotypes are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the archive.

        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Results:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """

        # generate first offsprings with the emitter
        genotypes, emitter_state, random_key = self._emitter.emit_first(
            repertoire, emitter_state, random_key
        )

        # scores the first offsprings
        fitnesses, descriptors, _, random_key = self._scoring_function(
            genotypes, random_key
        )

        # add first genotypes in the repertoire
        repertoire = repertoire.add(genotypes, descriptors, fitnesses, {})

        # generate second offsprings with the emitter
        genotypes, emitter_state, random_key = self._emitter.emit_second(
            repertoire=repertoire,
            emitter_state=emitter_state,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores={},
            random_key=random_key,
        )

        # scores the second offsprings
        fitnesses, descriptors, _, random_key = self._center_scoring_function(
            genotypes, random_key
        )

        # add second genotypes in the repertoire
        repertoire = repertoire.add(genotypes, descriptors, fitnesses, {})
        gradient_repertoire = gradient_repertoire.add(
            genotypes,
            descriptors,
            fitnesses,
            {},
        )

        # re-update emitter state after scoring is made
        emitter_state = self._emitter.state_update_second(
            emitter_state=emitter_state,
            repertoire=repertoire,
            gradient_repertoire=gradient_repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores={},
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, gradient_repertoire, emitter_state, metrics, random_key

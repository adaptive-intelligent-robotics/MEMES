from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from chex import ArrayTree
from qdax.core.containers.mapelites_repertoire import get_cells_indices
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


@jax.jit
def added_repertoire(
    genotypes: Genotype, descriptors: Descriptor, repertoire: Repertoire
) -> jnp.ndarray:
    """Compute if the given genotypes have been added to the repertoire in
    corresponding cell.

    Args:
        genotypes: genotypes candidate to addition
        descriptors: corresponding descriptors
        repertoire: repertoire considered for addition
    Returns:
        boolean for each genotype
    """
    cells = get_cells_indices(descriptors, repertoire.centroids)
    repertoire_genotypes = jax.tree_util.tree_map(
        lambda x: x[cells], repertoire.genotypes
    )
    added = jax.tree_util.tree_map(
        lambda x, y: jnp.equal(x, y), genotypes, repertoire_genotypes
    )
    added = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (descriptors.shape[0], -1)), added
    )
    added = jax.tree_util.tree_map(lambda x: jnp.all(x, axis=1), added)
    final_added = jnp.array(jax.tree_util.tree_leaves(added))
    final_added = jnp.all(final_added, axis=0)
    return final_added


@dataclass
class ESConfig:
    """Configuration for the ES emitter.

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


class ESEmitterState(EmitterState):
    """Emitter State for the ES emitter.

    Args:
        initial_optimizer_state: stored to re-initialise when sampling new parent
        optimizer_state: current optimizer state
        offspring: offspring generated through gradient estimate
        generation_count: generation counter
        random_key: key to handle stochastic operations
    """

    initial_optimizer_state: optax.OptState
    optimizer_states: ArrayTree
    offspring: Genotype
    generation_count: int
    random_key: RNGKey


class ESEmitter(Emitter):
    """
    An emitter that uses gradients approximated through sampling and resample
    a new center from the archive every num_generations_sample generations.

    This scan version scans through parents instead of performing all es
    operations in parallell, to avoid memory overload issue.
    """

    def __init__(
        self,
        config: ESConfig,
        batch_size: int,
        scoring_fn: Callable[
            [Genotype, RNGKey],
            Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
        ],
        num_descriptors: int,
        scan_batch_size: int = 1,
    ) -> None:
        """Initialise the emitter.

        Args:
            config
            batch_size: number of individuals generated per generation.
            scoring_fn: used to evaluate the samples for the gradient estimate.
            num_descriptors: dimension of the descriptors, used to initialise
                metrics.
        """
        assert (
            batch_size % scan_batch_size == 0 or scan_batch_size > batch_size
        ), "!!!ERROR!!! Batch-size should be dividible by scan-batch-size."

        # Set up config
        self._config = config

        # Set up other parameters
        self._batch_size = batch_size
        self._scoring_fn = scoring_fn
        self._num_descriptors = num_descriptors
        self._scan_batch_size = (
            scan_batch_size if batch_size > scan_batch_size else batch_size
        )
        self._num_scan = self._batch_size // self._scan_batch_size
        self._non_scan_explore = jnp.zeros(self._batch_size)
        self._explore = jnp.zeros((self._num_scan))

        # Initialise optimizer
        if self._config.adam_optimizer:
            self._optimizer = optax.adam(learning_rate=config.learning_rate)
        else:
            self._optimizer = optax.sgd(learning_rate=config.learning_rate)

    @property
    def batch_size(self) -> int:
        """
        Returns:
            the batch size emitted by the emitter.
        """
        return self._batch_size

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def init(
        self,
        init_genotypes: Genotype,
        random_key: RNGKey,
    ) -> Tuple[ESEmitterState, RNGKey]:
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

        return (
            ESEmitterState(
                initial_optimizer_state=initial_optimizer_state,
                optimizer_states=optimizer_states,
                offspring=init_genotypes,
                generation_count=0,
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
        emitter_state: ESEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """
        Return the offspring generated through gradient update.

        Params:
            repertoire: the MAP-Elites repertoire to sample from
            emitter_state
            random_key: a jax PRNG random key

        Returns:
            a batch of offsprings
            a new jax PRNG key
        """

        assert emitter_state is not None, "\n!!!ERROR!!! No emitter state."
        return emitter_state.offspring, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def _sub_scores(
        self,
        explore: jnp.ndarray,
        emitter_state: ESEmitterState,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> jnp.ndarray:
        """Compute the score from evaluation.
        In this simple case, just use the fitness."""

        return fitnesses

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def _scores(
        self,
        samples: Genotype,
        explore: jnp.ndarray,
        repertoire: Repertoire,
        emitter_state: ESEmitterState,
        random_key: RNGKey,
    ) -> Tuple[jnp.ndarray, RNGKey]:
        """Compute the scores associated with each sample.
        Can be easily overriden for alternative es approaches.

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

        # Get correspondin score
        scores = self._sub_scores(explore, emitter_state, fitnesses, descriptors, {})

        return scores, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def _pre_es_noise(
        self,
        parents: Genotype,
        emitter_state: ESEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Creating sample noises for gradient estimate."""

        random_key, subkey = jax.random.split(random_key)

        # Sampling mirror noise
        if self._config.sample_mirror:
            gradient_noise = jax.tree_util.tree_map(
                lambda x: jax.random.normal(
                    key=subkey,
                    shape=jnp.repeat(x, self._config.sample_number // 2, axis=0).shape,
                ),
                parents,
            )

        # Sampling non-mirror noise
        else:
            gradient_noise = jax.tree_util.tree_map(
                lambda x: jax.random.normal(
                    key=subkey,
                    shape=jnp.repeat(x, self._config.sample_number, axis=0).shape,
                ),
                parents,
            )

        return gradient_noise, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def _pre_es_apply(
        self,
        parents: Genotype,
        gradient_noise: Genotype,
        emitter_state: ESEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Creating samples for gradient estimate."""

        # Applying noise
        total_sample_number = self._config.sample_number
        if self._config.sample_mirror:
            # Splitting noise to apply it in mirror to samples
            sample_noise = jax.tree_util.tree_map(
                lambda x: jnp.concatenate(
                    [jnp.expand_dims(x, axis=1), jnp.expand_dims(-x, axis=1)], axis=1
                ).reshape(jnp.repeat(x, 2, axis=0).shape),
                gradient_noise,
            )

        # Sampling non-mirror noise
        else:
            sample_noise = gradient_noise

        # Expanding dimension to number of samples
        samples = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x, total_sample_number, axis=0),
            parents,
        )

        # Applying noise to each sample
        samples = jax.tree_util.tree_map(
            lambda mean, noise: mean + self._config.sample_sigma * noise,
            samples,
            sample_noise,
        )
        return samples, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def _post_es_emitter(
        self,
        parents: Genotype,
        scores: jnp.ndarray,
        gradient_noise: jnp.ndarray,
        emitter_state: ESEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Computing gradients"""

        total_sample_number = self._config.sample_number
        if self._config.sample_mirror:
            sample_number = total_sample_number // 2
        else:
            sample_number = total_sample_number

        # Computing rank with normalisation
        if self._config.sample_rank_norm:

            # Ranking objective
            ranking_indices = jnp.argsort(scores, axis=1)
            ranks = jnp.argsort(ranking_indices, axis=1)

            # Normalising ranks to [-0.5, 0.5]
            ranks = (ranks / (total_sample_number - 1)) - 0.5

        # Computing rank without normalisation
        else:
            ranks = scores

        # Reshaping rank to match shape of genotype_noise
        if self._config.sample_mirror:
            ranks = jnp.reshape(ranks, (self._scan_batch_size, sample_number, 2))
            ranks = jnp.apply_along_axis(lambda rank: rank[0] - rank[1], 2, ranks)

        ranks = jax.tree_util.tree_map(
            lambda x: jnp.reshape(
                jnp.repeat(ranks.ravel(), x[0].ravel().shape[0], axis=0), x.shape
            ),
            gradient_noise,
        )

        # Computing the gradients
        gradients = jax.tree_util.tree_map(
            lambda noise, rank: jnp.multiply(noise, rank),
            gradient_noise,
            ranks,
        )
        gradients = jax.tree_util.tree_map(
            lambda gradient: jnp.reshape(
                gradient, (self._scan_batch_size, sample_number, -1)
            ),
            gradients,
        )
        gradients = jax.tree_util.tree_map(
            lambda gradient, parent: jnp.reshape(
                -jnp.sum(gradient, axis=1)
                / (total_sample_number * self._config.sample_sigma),
                parent.shape,
            ),
            gradients,
            parents,
        )

        # Adding regularisation
        gradients = jax.tree_util.tree_map(
            lambda gradient, parent: gradient + self._config.l2_coefficient * parent,
            gradients,
            parents,
        )

        return gradients, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def _es_emitter(
        self,
        parents: Genotype,
        explore: jnp.ndarray,
        repertoire: Repertoire,
        emitter_state: ESEmitterState,
        random_key: RNGKey,
    ) -> Tuple[Genotype, RNGKey]:
        """Generate the samples, evaluate them and compute
        the gradients from the evaluations.

        Args:
            parents: parent to mutate
            explore: repartition of explore and exploit emitters
            repertoire: current repertoire, used to compute scores
            emitter_state: used to compute scores
            random_key

        Returns:
            the gradients to apply and a new random key
        """

        # Creating samples for gradient estimate.
        gradient_noise, random_key = self._pre_es_noise(
            parents, emitter_state, random_key
        )
        samples, random_key = self._pre_es_apply(
            parents, gradient_noise, emitter_state, random_key
        )

        # Choosing the score to use for rank
        total_sample_number = self._config.sample_number
        scores, random_key = self._scores(
            samples=samples,
            explore=explore,
            repertoire=repertoire,
            emitter_state=emitter_state,
            random_key=random_key,
        )
        scores = jnp.reshape(scores, (self._scan_batch_size, total_sample_number))

        # Computing gradients
        gradients, random_key = self._post_es_emitter(
            parents, scores, gradient_noise, emitter_state, random_key
        )

        return gradients, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def _apply_optimizer(
        self, parent: Genotype, gradient: Genotype, optimizer_state: ArrayTree
    ) -> Tuple[Genotype, ArrayTree]:
        (offspring_update, optimizer_state) = self._optimizer.update(
            gradient, optimizer_state
        )
        offspring = optax.apply_updates(parent, offspring_update)
        return offspring, optimizer_state

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def _optimizer_step(
        self,
        parents: Genotype,
        optimizer_states: optax.OptState,
        random_key: RNGKey,
        repertoire: Repertoire,
        emitter_state: ESEmitterState,
    ) -> Tuple[Genotype, optax.OptState, RNGKey]:
        """Apply one step of gradient to the parents.

        Args:
            parents: parent to mutate
            optimizer_state: used to update the parents
            random_key
            repertoire: current repertoire, used to compute scores
            emitter_state: used to compute scores

        Returns:
            the gradients to apply and a new random key
        """

        # Reshape parents for scan
        reshape_parents = jax.tree_util.tree_map(
            lambda x: x.reshape((self._num_scan, self._scan_batch_size) + x.shape[1:]),
            parents,
        )

        @jax.jit
        def _compute_gradients_function(
            carry: Tuple[RNGKey, int],
            unused: Tuple[()],
            reshape_parents: Genotype,
            explore: jnp.ndarray,
            repertoire: Repertoire,
            emitter_state: ESEmitterState,
        ) -> Tuple[Tuple[RNGKey, int], Genotype]:
            random_key, counter = carry
            current_parents = jax.tree_util.tree_map(
                lambda x: x[counter], reshape_parents
            )
            current_explore = explore[counter]
            gradients, random_key = self._es_emitter(
                parents=current_parents,
                explore=current_explore,
                repertoire=repertoire,
                emitter_state=emitter_state,
                random_key=random_key,
            )
            return (random_key, counter + 1), (gradients)

        # Applying es using scan
        gradients_fn = partial(
            _compute_gradients_function,
            reshape_parents=reshape_parents,
            explore=self._explore,
            repertoire=repertoire,
            emitter_state=emitter_state,
        )
        (random_key, _), (gradients) = jax.lax.scan(
            gradients_fn, (random_key, 0), (), length=self._num_scan
        )

        # Reshape gradients
        gradients = jax.tree_util.tree_map(
            lambda gradient, parent: jnp.reshape(gradient, parent.shape),
            gradients,
            parents,
        )

        # Applying gradients
        offspring, optimizer_states = jax.vmap(self._apply_optimizer)(
            parents, gradients, optimizer_states
        )

        return offspring, optimizer_states, random_key

    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "batch_size",
        ),
    )
    def _sample_parents(
        self,
        emitter_state: ESEmitterState,
        repertoire: Repertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
        added: jnp.ndarray,
        batch_size: int,
    ) -> Tuple[ESEmitterState, Genotype, optax.OptState, RNGKey]:
        """Sample new parents."""

        random_key = emitter_state.random_key

        # Sample new parents every num_generations_sample generations
        sample_parents = (
            emitter_state.generation_count % self._config.num_generations_sample == 0
        )
        parents, random_key = jax.lax.cond(
            sample_parents,
            lambda random_key: repertoire.sample(random_key, batch_size),
            lambda random_key: (genotypes, random_key),
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

        return emitter_state, parents, optimizer_states, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: ESEmitterState,
        repertoire: Repertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> ESEmitterState:
        """
        Update the generation count from current call.
        Generate the gradient offsprings for the next emitter call.
        """
        assert emitter_state is not None, "\n!!!ERROR!!! No emitter state."

        # Check if offspring have been added to repertoire
        added = added_repertoire(genotypes, descriptors, repertoire)

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
            emitter_state: ESEmitterState,
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
        generation_count = emitter_state.generation_count + 1

        return emitter_state.replace(  # type: ignore
            optimizer_states=optimizer_states,
            offspring=offspring,
            generation_count=generation_count,
            random_key=random_key,
        )

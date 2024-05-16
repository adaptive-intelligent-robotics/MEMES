from dataclasses import dataclass
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import optax
from chex import ArrayTree
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from core.emitters.es_base_emitter import ESEmitter
from core.emitters.es_novelty_archives import NoveltyArchive, SequentialNoveltyArchive


@dataclass
class NSESConfig:
    """Configuration for the ES emitter.

    Args:
        sample_number: num of samples for gradient estimate
        sample_sigma: std to sample the samples for gradient estimate
        sample_mirror: if True, use mirroring sampling
        sample_rank_norm: if True, use normalisation

        learning_rate
        l2_coefficient: coefficient for regularisation

        novelty_nearest_neighbors: number of nearest neighbors for
            novelty computation
        population_size: NSES population size
        fitness_weight: relative weight of fitness and novelty in objective
        adaptive_fitness_weight: if True, adaptively change the fitness_weight value
        adaptive_generations_period: if using adaptive finess weight, generation
            period to update the fitness_weight value
        adaptive_amount: if using adaptive finess weight, amouns used to update
            the fitness_weight value
    """

    sample_number: int = 10000
    sample_sigma: float = 0.02
    sample_mirror: bool = True
    sample_rank_norm: bool = True

    learning_rate: float = 0.01
    l2_coefficient: float = 0.0  # coefficient for regularisation

    novelty_nearest_neighbors: int = 10
    population_size: int = 5
    fitness_weight: float = 0.0
    adaptive_fitness_weight: bool = False
    adaptive_generations_period: int = 10
    adaptive_amount: float = 0.05


class NSESEmitterState(EmitterState):
    """Emitter State for the ES emitter.

    Args:
        population: population used by NSES to select the parents
        population_descriptors: corresponding descriptors, store to compute
            selection probability
        last_selected_index: index of last selected parent in the population,
            store to compute selection probability
        optimizer_state: current optimizer state
        offspring: offspring generated through gradient estimate
        fitness_weight: current fitness weight
        fitness_weight_decrease: store if the fitness_weight was descreased
            at previous update
        num_fitness_stagnate: store how many updates in a row did the fitness
            stagnate
        generation_count: generation counter used to update the novelty archive
        novelty_archive: used to compute novelty for explore
        random_key: key to handle stochastic operations
    """

    population: Genotype
    population_descriptors: Descriptor  # Store to compute selection probability
    last_selected_index: int  # Store to compute selection probability
    optimizer_states: ArrayTree  # One optimizer per member of population
    offspring: Genotype
    fitness_weight: float
    fitness_weight_decrease: bool
    num_fitness_stagnate: int
    fitness_value: float
    generation_count: int
    novelty_archive: NoveltyArchive
    random_key: RNGKey


class NSESEmitter(ESEmitter):
    """
    An emitter that uses novelty gradients approximated through sampling.
    """

    def __init__(
        self,
        config: NSESConfig,
        total_generations: int,
        batch_size: int,
        scoring_fn: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        construction_fn: Callable[[int, RNGKey], Tuple[Genotype, RNGKey]],
        num_descriptors: int,
    ) -> None:
        """Initialise the emitter.

        Args:
            config
            total_generations: total number of generations for which the
                emitter will run, necessary to set up the novelty archive.
            batch_size: number of individuals generated per generation.
            scoring_fn: used to evaluate the samples for the gradient estimate.
            construction_fn: used to initialised the meta population
            num_descriptors: dimension of the descriptors, used to initialise
                the empty novelty archive.

        Returns: /
        """
        self._config = config
        self._config.adam_optimizer = True
        self._config.use_novelty_archive = True
        self._config.use_novelty_fifo = False
        self._config.use_explore = True
        self._config.use_exploit = False

        self._batch_size = batch_size
        self._scoring_fn = scoring_fn
        self._construction_fn = construction_fn
        self._total_generations = total_generations
        self._num_descriptors = num_descriptors
        self._scan_batch_size = 1
        self._num_scan = 1
        self._explore = jnp.zeros((1))
        self._novelty_archive_size = self._total_generations * self._batch_size

        self._optimizer = optax.adam(learning_rate=config.learning_rate)

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def init(
        self, init_genotypes: Genotype, random_key: RNGKey
    ) -> Tuple[NSESEmitterState, RNGKey]:
        """Initializes the emitter state.

        Args:
            init_genotypes: The initial population.
            random_key: A random key.

        Returns:
            The initial state of the emitter, a new random key.
        """

        # Empty Novelty archive
        novelty_archive = SequentialNoveltyArchive.init(
            self._total_generations * self._batch_size, self._num_descriptors
        )

        # Initialise population
        population, random_key = self._construction_fn(
            self._config.population_size, random_key
        )

        # Evaluate to initialise bd and fitness
        fitnesses, descriptors, _, random_key = self._scoring_fn(population, random_key)

        # Initial optimizer state
        params = jax.tree_util.tree_map(lambda x: x[0], init_genotypes)
        initial_optimizer_state = self._optimizer.init(params)
        optimizer_states = jax.tree_util.tree_map(
            lambda x: jnp.repeat(
                jnp.expand_dims(x, axis=0), self._config.population_size, axis=0
            ),
            initial_optimizer_state,
        )

        return (
            NSESEmitterState(
                population=population,
                population_descriptors=descriptors,
                last_selected_index=0,
                optimizer_states=optimizer_states,
                offspring=init_genotypes,
                fitness_weight=self._config.fitness_weight,
                fitness_weight_decrease=False,
                num_fitness_stagnate=0,
                fitness_value=jnp.min(fitnesses),
                generation_count=0,
                novelty_archive=novelty_archive,
                random_key=random_key,
            ),
            random_key,
        )

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def _scores(
        self,
        samples: Genotype,
        explore: jnp.ndarray,
        repertoire: Repertoire,
        emitter_state: NSESEmitterState,
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
        fitnesses, descriptors, extra_scores, random_key = self._scoring_fn(
            samples,
            random_key,
        )

        return (
            1 - emitter_state.fitness_weight
        ) * emitter_state.novelty_archive.novelty(
            descriptors, self._config.novelty_nearest_neighbors
        ) + emitter_state.fitness_weight * fitnesses, random_key

    @partial(
        jax.jit,
        static_argnames=("self",),
    )
    def state_update(
        self,
        emitter_state: NSESEmitterState,
        repertoire: Repertoire,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
    ) -> NSESEmitterState:
        """
        Update the novelty archive, population, adaptive weight and generation
        count from current call. Generate the gradient offsprings for the next
        emitter call.
        """
        assert emitter_state is not None, "\n!!!ERROR!!! No emitter state."

        # Update novelty archive
        novelty_archive = emitter_state.novelty_archive.update(
            descriptors, repertoire.descriptors, repertoire.fitnesses
        )

        # Update descriptor of last selected parent
        population = emitter_state.population
        optimizer_states = emitter_state.optimizer_states
        population_descriptors = emitter_state.population_descriptors.at[
            emitter_state.last_selected_index
        ].set(descriptors[0])

        # Update fitness weight
        fitness_weight = emitter_state.fitness_weight
        fitness_weight_decrease = emitter_state.fitness_weight_decrease
        fitness_value = emitter_state.fitness_value
        num_fitness_stagnate = emitter_state.num_fitness_stagnate
        if self._config.adaptive_fitness_weight:
            num_fitness_stagnate = jax.lax.cond(
                fitnesses[0] <= fitness_value,
                lambda unused: num_fitness_stagnate + 1,
                lambda unused: 0,
                (),
            )
            fitness_weight_decrease, num_fitness_stagnate = jax.lax.cond(
                fitness_weight_decrease,
                lambda unused: jax.lax.cond(
                    fitnesses[0] < fitness_value,
                    lambda unused: (True, 0),
                    lambda unused: (False, 0),
                    (),
                ),
                lambda unused: jax.lax.cond(
                    num_fitness_stagnate > self._config.adaptive_generations_period,
                    lambda unused: (True, 0),
                    lambda unused: (False, num_fitness_stagnate),
                    (),
                ),
                (),
            )
            fitness_weight = jax.lax.cond(
                fitness_weight_decrease,
                lambda unused: jax.lax.max(
                    fitness_weight - self._config.adaptive_amount, 0.0
                ),
                lambda unused: jax.lax.min(
                    fitness_weight + self._config.adaptive_amount, 1.0
                ),
                (),
            )

        # Compute novelty of all population
        novelties = jax.vmap(
            partial(
                novelty_archive._single_novelty,
                num_nearest_neighbors=self._config.novelty_nearest_neighbors,
            )
        )(population_descriptors)

        # Sample new parents based on novelties
        random_key = emitter_state.random_key
        p = novelties / jnp.sum(novelties)
        random_key, subkey = jax.random.split(random_key)
        selected_index = jax.random.choice(
            subkey, jnp.arange(0, self._config.population_size, 1), p=p
        )

        parent = jax.tree_util.tree_map(
            lambda pop: jnp.expand_dims(pop.at[selected_index].get(), axis=0),
            population,
        )
        optimizer_state_parent = jax.tree_util.tree_map(
            lambda optm_pop: jnp.expand_dims(optm_pop.at[selected_index].get(), axis=0),
            optimizer_states,
        )

        # Update emitter state to use it with next functions
        emitter_state = emitter_state.replace(
            population_descriptors=population_descriptors,
            last_selected_index=selected_index,
            novelty_archive=novelty_archive,
            fitness_weight=fitness_weight,
            fitness_weight_decrease=fitness_weight_decrease,
            fitness_value=jax.lax.max(fitnesses[0], fitness_value),
            num_fitness_stagnate=num_fitness_stagnate,
        )

        # Apply one optimizer step to compute the offspring
        (offspring, optimizer_state_offspring, random_key,) = self._optimizer_step(
            parents=parent,
            optimizer_states=optimizer_state_parent,
            random_key=random_key,
            repertoire=repertoire,
            emitter_state=emitter_state,
        )

        # Update population and optimizer state
        population = jax.tree_util.tree_map(
            lambda pop, off: pop.at[selected_index].set(off.squeeze(axis=0)),
            population,
            offspring,
        )
        optimizer_states = jax.tree_util.tree_map(
            lambda optm_pop, opt_off: optm_pop.at[selected_index].set(
                opt_off.squeeze(axis=0)
            ),
            optimizer_states,
            optimizer_state_offspring,
        )

        # Increase generation counter
        generation_count = emitter_state.generation_count + 1

        return emitter_state.replace(  # type: ignore
            population=population,
            optimizer_states=optimizer_states,
            offspring=offspring,
            generation_count=generation_count,
            random_key=random_key,
        )

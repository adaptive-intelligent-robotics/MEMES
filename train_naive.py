import logging
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Tuple

import hydra
import jax
import jax.numpy as jnp
from hydra.core.config_store import ConfigStore
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.types import RNGKey
from qdax.utils.sampling import sampling

from core.containers_metrics.metrics_mapelites_repertoire import (
    MapElitesRepertoire,
    compute_euclidean_centroids,
)
from core.emitters_metrics.metrics_standard_emitters import MixingEmitter
from core.map_elites_metrics import MAPElites
from core.stochasticity_utils import reevaluation_function
from initialisation import set_up_default_metrics_dict, set_up_envs, set_up_metrics
from main_loop import main_loop


@dataclass
class ExperimentConfig:
    """Configuration from this experiment script"""

    alg_name: str
    # Env config
    seed: int
    env_name: str
    episode_length: int
    policy_hidden_layer_sizes: Tuple[int, ...]
    # ME config
    num_evaluations: int
    num_iterations: int
    batch_size: int
    fixed_init_state: bool
    discard_dead: bool
    # Grid config
    grid_shape: Tuple[int, ...]
    # Emitter config
    iso_sigma: float
    line_sigma: float
    crossover_percentage: float
    # others
    log_period: int  # only for timings and metrics
    store_repertoire: bool
    store_repertoire_log_period: int

    # naive
    num_samples: int

    # Stochasticity config
    num_reevals: int
    log_period_reevals: int


@hydra.main(config_path="configs", config_name="naive")
def train(config: ExperimentConfig) -> None:

    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().handlers[0].setLevel(logging.INFO)
    logger = logging.getLogger(f"{__name__}")

    # Choose stopping criteria
    if config.num_iterations > 0 and config.num_evaluations > 0:
        print(
            "!!!WARNING!!! Both num_iterations and num_evaluations are set",
            "choosing num_iterations over num_evaluations",
        )
    if config.num_iterations > 0:
        num_iterations = config.num_iterations
    elif config.num_evaluations > 0:
        num_iterations = config.num_evaluations // config.batch_size + 1

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Setup environment
    (
        env,
        scoring_fn,
        policy_network,
        construction_fn,
        init_variables,
        reward_offset,
        behavior_descriptor_length,
        genotype_dim,
        random_key,
    ) = set_up_envs(config, config.batch_size, random_key)

    # Setup all metrics
    (
        metrics_fn,
        reeval_metrics_fn,
    ) = set_up_metrics(config.episode_length, reward_offset)

    # Wrap the scoring function to do sampling
    me_scoring_fn = partial(
        sampling,
        scoring_fn=scoring_fn,
        num_samples=config.num_samples,
    )

    # Compute the centroids
    logger.warning("--- Compute the CVT centroids ---")
    minval, maxval = env.behavior_descriptor_limits
    init_time = time.time()
    centroids = compute_euclidean_centroids(
        grid_shape=config.grid_shape,
        minval=minval,
        maxval=maxval,
    )
    centroid_time = time.time() - init_time
    logger.warning(f"--- Duration for CVT centroids computation : {centroid_time:.2f}s")

    # Define emitter
    variation_fn = partial(
        isoline_variation, iso_sigma=config.iso_sigma, line_sigma=config.line_sigma
    )
    mixing_emitter = MixingEmitter(
        mutation_fn=None,
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=config.batch_size,
        behavior_descriptor_length=behavior_descriptor_length,
    )

    # Instantiate MAP-Elites
    map_elites = MAPElites(
        scoring_function=me_scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_fn,
    )

    # Init algorithm
    logger.warning("--- Algorithm initialisation ---")
    start_time = time.time()
    repertoire, emitter_state, random_key = map_elites.init(
        init_variables, centroids, random_key
    )
    init_time = time.time() - start_time
    logger.warning("--- Initialised ---")
    logger.warning("--- Starting the algorithm main process ---")

    # Define a reeval function
    metric_repertoire = MapElitesRepertoire.init(
        genotypes=init_variables,
        fitnesses=jnp.zeros(config.batch_size),
        descriptors=jnp.zeros((config.batch_size, env.behavior_descriptor_length)),
        extra_scores={},
        centroids=centroids,
    )
    reevaluation_fn = partial(
        reevaluation_function,
        metric_repertoire=metric_repertoire,
        scoring_fn=scoring_fn,
        num_reevals=config.num_reevals,
        use_median=True,
    )

    # Set up metric dicts
    full_metrics, full_reeval_metrics, timings = set_up_default_metrics_dict(
        init_time=init_time,
        centroid_time=centroid_time,
        num_iterations=num_iterations,
    )

    def additional_metrics_fn(
        metrics: Dict, repertoire: MapElitesRepertoire, emitter_state: Any
    ) -> Dict:
        return metrics

    # Function to count number of evaluations
    count_evals_fn = (
        lambda iteration: iteration * config.batch_size * config.num_samples
    )

    # Main QD loop
    map_elites_update_fn = partial(map_elites.update)

    def update_fn(
        repertoire: MapElitesRepertoire,
        second_repertoire: MapElitesRepertoire,
        emitter_state: Any,
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, MapElitesRepertoire, Any, Dict, RNGKey]:
        repertoire, emitter_state, metrics, random_key = map_elites_update_fn(
            repertoire, emitter_state, random_key
        )
        return repertoire, None, emitter_state, metrics, random_key

    full_metrics, full_reeval_metrics, timings, random_key = main_loop(
        logger=logger,
        init_time=init_time,
        centroid_time=centroid_time,
        behavior_descriptor_length=behavior_descriptor_length,
        num_iterations=num_iterations,
        update_fn=update_fn,
        repertoire=repertoire,
        second_repertoire=None,
        emitter_state=emitter_state,
        count_evals_fn=count_evals_fn,
        reevaluation_fn=reevaluation_fn,
        metrics_fn=metrics_fn,
        full_metrics=full_metrics,
        full_reeval_metrics=full_reeval_metrics,
        timings=timings,
        additional_metrics_fn=additional_metrics_fn,
        reeval_metrics_fn=reeval_metrics_fn,
        log_period=config.log_period,
        num_reevals=config.num_reevals,
        log_period_reevals=config.log_period_reevals,
        store_repertoire=config.store_repertoire,
        store_repertoire_log_period=config.store_repertoire_log_period,
        random_key=random_key,
    )


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="validate_experiment_config", node=ExperimentConfig)
    train()

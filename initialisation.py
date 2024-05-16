from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
from hydra.core.config_store import ConfigStore
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.arm import arm_scoring_function
from qdax.types import Genotype, RNGKey

from set_up_brax import (
    get_behavior_descriptor_length_brax,
    get_environment_brax,
    get_policy_struc_brax,
    get_reward_offset_brax,
    get_scoring_function_brax,
)
from tasks.hexapod import create_default_hexapod_controller


def set_up_envs(
    config: ConfigStore,
    batch_size: int,
    random_key: RNGKey,
) -> Tuple[Any, Callable, Any, Callable, Genotype, float, int, int, RNGKey]:

    # Init environment and population of controllers
    print("Env name: ", config.env_name)

    # Open-loop hexapod control
    if config.env_name == "hexapod_omni":
        (
            env,
            policy_network,
            init_variables,
            scoring_fn,
            construction_fn,
            reward_offset,
            behavior_descriptor_length,
            genotype_dim,
            random_key,
        ) = create_default_hexapod_controller(
            random_key,
            config.episode_length,
            batch_size,
            deterministic=config.fixed_init_state,
        )

    # Non-dynamic 100DoF arm
    elif config.env_name == "arm":
        reward_offset = 1
        behavior_descriptor_length = 2
        genotype_dim = 1000

        class ArmEnv:
            """Placeholder class to make sure all algortihms run."""

            @property
            def behavior_descriptor_length(self) -> int:
                return 2

            @property
            def behavior_descriptor_limits(self) -> Tuple[List[float], List[float]]:
                return ([0, 0], [1, 1])

        def construction_fn(size: int, random_key: RNGKey) -> jnp.ndarray:
            random_key, subkey = jax.random.split(random_key)
            init_variables = jax.random.uniform(
                random_key, shape=(size, genotype_dim), minval=0, maxval=1
            )
            return init_variables, random_key

        env = ArmEnv()
        init_variables, random_key = construction_fn(batch_size, random_key)
        scoring_fn = arm_scoring_function
        policy_network = None

    else:
        # Initialising environment
        env = get_environment_brax(
            config.env_name, config.episode_length, config.fixed_init_state
        )

        # Get network size
        input_size, output_size, policy_layer_sizes, activation = get_policy_struc_brax(
            env, config.policy_hidden_layer_sizes
        )

        # Create the network
        policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=activation,
        )

        # Get the scoring function
        scoring_fn, random_key = get_scoring_function_brax(
            env,
            config.env_name,
            config.episode_length,
            policy_network,
            random_key,
        )

        # Build init variables
        def construction_fn(size: int, random_key: RNGKey) -> jnp.ndarray:
            random_key, subkey = jax.random.split(random_key)
            keys = jax.random.split(subkey, num=size)
            fake_batch = jnp.zeros(shape=(size, output_size))
            init_variables = jax.vmap(policy_network.init)(keys, fake_batch)
            return init_variables, random_key

        # Build all common parts
        reward_offset = get_reward_offset_brax(env, config.env_name)
        behavior_descriptor_length = get_behavior_descriptor_length_brax(
            env, config.env_name
        )
        init_variables, random_key = construction_fn(batch_size, random_key)
        genotype_dim = jnp.prod(jnp.asarray(config.policy_hidden_layer_sizes))

    return (
        env,
        scoring_fn,
        policy_network,
        construction_fn,
        init_variables,
        reward_offset,
        behavior_descriptor_length,
        genotype_dim,
        random_key,
    )


def set_up_metrics(
    episode_length: int,
    reward_offset: float,
) -> Tuple[Callable, Callable]:

    # Define metrics functions
    def metrics_fn(repertoire: MapElitesRepertoire) -> Dict:
        grid_empty = repertoire.fitnesses == -jnp.inf
        qd_score = jnp.sum(repertoire.fitnesses, where=~grid_empty)
        qd_score += reward_offset * episode_length * jnp.sum(1.0 - grid_empty)
        coverage = 100 * jnp.mean(1.0 - grid_empty)
        max_fitness = jnp.max(repertoire.fitnesses)
        min_fitness = jnp.min(
            jnp.where(repertoire.fitnesses > -jnp.inf, repertoire.fitnesses, jnp.inf)
        )
        return {
            "qd_score": jnp.array([qd_score]),
            "max_fitness": jnp.array([max_fitness]),
            "min_fitness": jnp.array([min_fitness]),
            "coverage": jnp.array([coverage]),
        }

    def reeval_metrics_fn(
        reeval_repertoire: MapElitesRepertoire,
        fit_reeval_repertoire: MapElitesRepertoire,
        desc_reeval_repertoire: MapElitesRepertoire,
        fit_var_repertoire: MapElitesRepertoire,
        reeval_fit_var_repertoire: MapElitesRepertoire,
        desc_var_repertoire: MapElitesRepertoire,
        reeval_desc_var_repertoire: MapElitesRepertoire,
    ) -> Dict:
        reeval_metrics = metrics_fn(reeval_repertoire)
        fit_reeval_metrics = metrics_fn(fit_reeval_repertoire)
        desc_reeval_metrics = metrics_fn(desc_reeval_repertoire)
        fit_var_metrics = metrics_fn(fit_var_repertoire)
        reeval_fit_var_metrics = metrics_fn(reeval_fit_var_repertoire)
        desc_var_metrics = metrics_fn(desc_var_repertoire)
        reeval_desc_var_metrics = metrics_fn(reeval_desc_var_repertoire)
        return {
            "reeval_qd_score": reeval_metrics["qd_score"],
            "reeval_max_fitness": reeval_metrics["max_fitness"],
            "reeval_min_fitness": reeval_metrics["min_fitness"],
            "reeval_coverage": reeval_metrics["coverage"],
            "fit_reeval_qd_score": fit_reeval_metrics["qd_score"],
            "fit_reeval_max_fitness": fit_reeval_metrics["max_fitness"],
            "fit_reeval_min_fitness": fit_reeval_metrics["min_fitness"],
            "fit_reeval_coverage": fit_reeval_metrics["coverage"],
            "desc_reeval_qd_score": desc_reeval_metrics["qd_score"],
            "desc_reeval_max_fitness": desc_reeval_metrics["max_fitness"],
            "desc_reeval_min_fitness": desc_reeval_metrics["min_fitness"],
            "desc_reeval_coverage": desc_reeval_metrics["coverage"],
            "fit_var_qd_score": fit_var_metrics["qd_score"],
            "fit_var_max_fitness": fit_var_metrics["max_fitness"],
            "fit_var_min_fitness": fit_var_metrics["min_fitness"],
            "fit_var_coverage": fit_var_metrics["coverage"],
            "reeval_fit_var_qd_score": reeval_fit_var_metrics["qd_score"],
            "reeval_fit_var_max_fitness": reeval_fit_var_metrics["max_fitness"],
            "reeval_fit_var_min_fitness": reeval_fit_var_metrics["min_fitness"],
            "reeval_fit_var_coverage": reeval_fit_var_metrics["coverage"],
            "desc_var_qd_score": desc_var_metrics["qd_score"],
            "desc_var_max_fitness": desc_var_metrics["max_fitness"],
            "desc_var_min_fitness": desc_var_metrics["min_fitness"],
            "desc_var_coverage": desc_var_metrics["coverage"],
            "reeval_desc_var_qd_score": reeval_desc_var_metrics["qd_score"],
            "reeval_desc_var_max_fitness": reeval_desc_var_metrics["max_fitness"],
            "reeval_desc_var_min_fitness": reeval_desc_var_metrics["min_fitness"],
            "reeval_desc_var_coverage": reeval_desc_var_metrics["coverage"],
        }

    return (
        metrics_fn,
        reeval_metrics_fn,
    )


def set_up_default_metrics_dict(
    init_time: float,
    centroid_time: float,
    num_iterations: int,
) -> Tuple[Dict, Dict, Dict]:
    full_metrics = {
        "epoch": jnp.array([0.0]),
        "evals": jnp.array([0.0]),
        "coverage": jnp.array([0.0]),
        "max_fitness": jnp.array([0.0]),
        "min_fitness": jnp.array([0.0]),
        "qd_score": jnp.array([0.0]),
    }
    full_reeval_metrics = {
        "epoch": jnp.array([0.0]),
        "evals": jnp.array([0.0]),
        "reeval_coverage": jnp.array([0.0]),
        "reeval_max_fitness": jnp.array([0.0]),
        "reeval_min_fitness": jnp.array([0.0]),
        "reeval_qd_score": jnp.array([0.0]),
        "fit_reeval_coverage": jnp.array([0.0]),
        "fit_reeval_max_fitness": jnp.array([0.0]),
        "fit_reeval_min_fitness": jnp.array([0.0]),
        "fit_reeval_qd_score": jnp.array([0.0]),
        "desc_reeval_coverage": jnp.array([0.0]),
        "desc_reeval_max_fitness": jnp.array([0.0]),
        "desc_reeval_min_fitness": jnp.array([0.0]),
        "desc_reeval_qd_score": jnp.array([0.0]),
        "fit_var_coverage": jnp.array([0.0]),
        "fit_var_max_fitness": jnp.array([0.0]),
        "fit_var_min_fitness": jnp.array([0.0]),
        "fit_var_qd_score": jnp.array([0.0]),
        "desc_var_coverage": jnp.array([0.0]),
        "desc_var_max_fitness": jnp.array([0.0]),
        "desc_var_min_fitness": jnp.array([0.0]),
        "desc_var_qd_score": jnp.array([0.0]),
        "reeval_fit_var_coverage": jnp.array([0.0]),
        "reeval_fit_var_max_fitness": jnp.array([0.0]),
        "reeval_fit_var_min_fitness": jnp.array([0.0]),
        "reeval_fit_var_qd_score": jnp.array([0.0]),
        "reeval_desc_var_coverage": jnp.array([0.0]),
        "reeval_desc_var_max_fitness": jnp.array([0.0]),
        "reeval_desc_var_min_fitness": jnp.array([0.0]),
        "reeval_desc_var_qd_score": jnp.array([0.0]),
    }
    timings = {
        "epoch": jnp.array([0.0]),
        "evals": jnp.array([0.0]),
        "init_time": init_time,
        "centroids_time": centroid_time,
        "runtime_logs": jnp.zeros([(num_iterations) + 1, 1]),
        "avg_iteration_time": 0.0,
    }
    return full_metrics, full_reeval_metrics, timings


def set_up_explore_exploit_metrics_dict(
    init_time: float,
    centroid_time: float,
    num_iterations: int,
) -> Tuple[Dict, Dict, Dict]:
    full_metrics = {
        "epoch": jnp.array([0.0]),
        "evals": jnp.array([0.0]),
        "coverage": jnp.array([0.0]),
        "max_fitness": jnp.array([0.0]),
        "min_fitness": jnp.array([0.0]),
        "qd_score": jnp.array([0.0]),
        "proportion_explore": jnp.array([0.0]),
        "explore_usage": jnp.array([0.0]),
        "exploit_usage": jnp.array([0.0]),
        "parents_distance": jnp.array([0.0]),
    }
    full_reeval_metrics = {
        "epoch": jnp.array([0.0]),
        "evals": jnp.array([0.0]),
        "reeval_coverage": jnp.array([0.0]),
        "reeval_max_fitness": jnp.array([0.0]),
        "reeval_min_fitness": jnp.array([0.0]),
        "reeval_qd_score": jnp.array([0.0]),
        "fit_reeval_coverage": jnp.array([0.0]),
        "fit_reeval_max_fitness": jnp.array([0.0]),
        "fit_reeval_min_fitness": jnp.array([0.0]),
        "fit_reeval_qd_score": jnp.array([0.0]),
        "desc_reeval_coverage": jnp.array([0.0]),
        "desc_reeval_max_fitness": jnp.array([0.0]),
        "desc_reeval_min_fitness": jnp.array([0.0]),
        "desc_reeval_qd_score": jnp.array([0.0]),
        "fit_var_coverage": jnp.array([0.0]),
        "fit_var_max_fitness": jnp.array([0.0]),
        "fit_var_min_fitness": jnp.array([0.0]),
        "fit_var_qd_score": jnp.array([0.0]),
        "desc_var_coverage": jnp.array([0.0]),
        "desc_var_max_fitness": jnp.array([0.0]),
        "desc_var_min_fitness": jnp.array([0.0]),
        "desc_var_qd_score": jnp.array([0.0]),
        "reeval_fit_var_coverage": jnp.array([0.0]),
        "reeval_fit_var_max_fitness": jnp.array([0.0]),
        "reeval_fit_var_min_fitness": jnp.array([0.0]),
        "reeval_fit_var_qd_score": jnp.array([0.0]),
        "reeval_desc_var_coverage": jnp.array([0.0]),
        "reeval_desc_var_max_fitness": jnp.array([0.0]),
        "reeval_desc_var_min_fitness": jnp.array([0.0]),
        "reeval_desc_var_qd_score": jnp.array([0.0]),
    }
    timings = {
        "epoch": jnp.array([0.0]),
        "evals": jnp.array([0.0]),
        "init_time": init_time,
        "centroids_time": centroid_time,
        "runtime_logs": jnp.zeros([(num_iterations) + 1, 1]),
        "avg_iteration_time": 0.0,
    }
    return full_metrics, full_reeval_metrics, timings


def set_up_explore_exploit_reset_metrics_dict(
    init_time: float,
    centroid_time: float,
    num_iterations: int,
) -> Tuple[Dict, Dict, Dict]:
    full_metrics = {
        "epoch": jnp.array([0.0]),
        "evals": jnp.array([0.0]),
        "coverage": jnp.array([0.0]),
        "max_fitness": jnp.array([0.0]),
        "min_fitness": jnp.array([0.0]),
        "qd_score": jnp.array([0.0]),
        "proportion_explore": jnp.array([0.0]),
        "explore_usage": jnp.array([0.0]),
        "exploit_usage": jnp.array([0.0]),
        "parents_distance": jnp.array([0.0]),
        "explore_max_gen_reset": jnp.array([0.0]),
        "exploit_max_gen_reset": jnp.array([0.0]),
        "explore_mean_gen_reset": jnp.array([0.0]),
        "exploit_mean_gen_reset": jnp.array([0.0]),
        "explore_mean_stagnate": jnp.array([0.0]),
        "exploit_mean_stagnate": jnp.array([0.0]),
    }
    full_reeval_metrics = {
        "epoch": jnp.array([0.0]),
        "evals": jnp.array([0.0]),
        "reeval_coverage": jnp.array([0.0]),
        "reeval_max_fitness": jnp.array([0.0]),
        "reeval_min_fitness": jnp.array([0.0]),
        "reeval_qd_score": jnp.array([0.0]),
        "fit_reeval_coverage": jnp.array([0.0]),
        "fit_reeval_max_fitness": jnp.array([0.0]),
        "fit_reeval_min_fitness": jnp.array([0.0]),
        "fit_reeval_qd_score": jnp.array([0.0]),
        "desc_reeval_coverage": jnp.array([0.0]),
        "desc_reeval_max_fitness": jnp.array([0.0]),
        "desc_reeval_min_fitness": jnp.array([0.0]),
        "desc_reeval_qd_score": jnp.array([0.0]),
        "fit_var_coverage": jnp.array([0.0]),
        "fit_var_max_fitness": jnp.array([0.0]),
        "fit_var_min_fitness": jnp.array([0.0]),
        "fit_var_qd_score": jnp.array([0.0]),
        "desc_var_coverage": jnp.array([0.0]),
        "desc_var_max_fitness": jnp.array([0.0]),
        "desc_var_min_fitness": jnp.array([0.0]),
        "desc_var_qd_score": jnp.array([0.0]),
        "reeval_fit_var_coverage": jnp.array([0.0]),
        "reeval_fit_var_max_fitness": jnp.array([0.0]),
        "reeval_fit_var_min_fitness": jnp.array([0.0]),
        "reeval_fit_var_qd_score": jnp.array([0.0]),
        "reeval_desc_var_coverage": jnp.array([0.0]),
        "reeval_desc_var_max_fitness": jnp.array([0.0]),
        "reeval_desc_var_min_fitness": jnp.array([0.0]),
        "reeval_desc_var_qd_score": jnp.array([0.0]),
    }
    timings = {
        "epoch": jnp.array([0.0]),
        "evals": jnp.array([0.0]),
        "init_time": init_time,
        "centroids_time": centroid_time,
        "runtime_logs": jnp.zeros([(num_iterations) + 1, 1]),
        "avg_iteration_time": 0.0,
    }
    return full_metrics, full_reeval_metrics, timings

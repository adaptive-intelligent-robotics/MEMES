import os
import pickle
import time
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.types import RNGKey


def main_loop(
    logger: Any,
    init_time: float,
    centroid_time: float,
    behavior_descriptor_length: int,
    num_iterations: int,
    update_fn: Callable,
    repertoire: MapElitesRepertoire,
    second_repertoire: MapElitesRepertoire,
    emitter_state: Any,
    count_evals_fn: Callable,
    reevaluation_fn: Callable,
    metrics_fn: Callable,
    reeval_metrics_fn: Callable,
    full_metrics: Dict,
    full_reeval_metrics: Dict,
    timings: Dict,
    additional_metrics_fn: Callable,
    log_period: int,
    num_reevals: int,
    log_period_reevals: int,
    store_repertoire: bool,
    store_repertoire_log_period: int,
    random_key: RNGKey,
) -> Tuple[Dict, Dict, Dict, RNGKey]:

    output_dir = "./"

    # Setup metrics checkpoint save
    _last_metrics_dir = os.path.join(output_dir, "checkpoints", "last_metrics")
    os.makedirs(_last_metrics_dir, exist_ok=True)
    _grid_img_dir = os.path.join(output_dir, "images", "me_grids")
    os.makedirs(_grid_img_dir, exist_ok=True)
    _metrics_img_dir = os.path.join(output_dir, "images", "me_metrics")
    os.makedirs(_metrics_img_dir, exist_ok=True)
    _timings_dir = os.path.join(output_dir, "timings")
    os.makedirs(_timings_dir, exist_ok=True)

    # Setup repertoire checkpoint save
    _last_grid_dir = os.path.join(output_dir, "checkpoints", "last_grid")
    os.makedirs(_last_grid_dir, exist_ok=True)
    _last_reeval_grid_dir = os.path.join(output_dir, "checkpoints", "last_reeval_grid")
    os.makedirs(_last_reeval_grid_dir, exist_ok=True)
    _last_fit_reeval_grid_dir = os.path.join(
        output_dir, "checkpoints", "last_fit_reeval_grid"
    )
    os.makedirs(_last_fit_reeval_grid_dir, exist_ok=True)
    _last_desc_reeval_grid_dir = os.path.join(
        output_dir, "checkpoints", "last_desc_reeval_grid"
    )
    os.makedirs(_last_desc_reeval_grid_dir, exist_ok=True)
    _last_fit_var_grid_dir = os.path.join(
        output_dir, "checkpoints", "last_fit_var_grid"
    )
    os.makedirs(_last_fit_var_grid_dir, exist_ok=True)
    _last_reeval_fit_var_grid_dir = os.path.join(
        output_dir, "checkpoints", "last_reeval_fit_var_grid"
    )
    os.makedirs(_last_reeval_fit_var_grid_dir, exist_ok=True)
    _last_desc_var_grid_dir = os.path.join(
        output_dir, "checkpoints", "last_desc_var_grid"
    )
    os.makedirs(_last_desc_var_grid_dir, exist_ok=True)
    _last_reeval_desc_var_grid_dir = os.path.join(
        output_dir, "checkpoints", "last_reeval_desc_var_grid"
    )
    os.makedirs(_last_reeval_desc_var_grid_dir, exist_ok=True)

    # Main QD Loop
    total_start_time = time.time()
    algorithm_time = 0.0
    total_evals = 0
    for iteration in range(num_iterations):
        logger.warning(
            f"--- Iteration indice : {iteration} out of {num_iterations} ---"
        )

        start_time = time.time()
        (
            repertoire,
            second_repertoire,
            emitter_state,
            metrics,
            random_key,
        ) = update_fn(
            repertoire,
            second_repertoire,
            emitter_state,
            random_key,
        )
        iteration_time = time.time() - start_time
        algorithm_time += iteration_time

        logger.warning(f"--- Current QD Score: {metrics['qd_score'][-1]:.2f}")
        logger.warning(f"--- Current Coverage: {metrics['coverage'][-1]:.2f}%")
        logger.warning(f"--- Current Max Fitness: {metrics['max_fitness'][-1]}")
        logger.warning(f"--- Iteration time: {iteration_time}")

        # Add epoch and evals
        total_evals += count_evals_fn(iteration)
        metrics["epoch"] = jnp.array([iteration])
        metrics["evals"] = jnp.array([total_evals])
        metrics["time"] = jnp.array([algorithm_time])

        # Add emitter_state metrics to metrics
        metrics = additional_metrics_fn(metrics, repertoire, emitter_state)

        # Save metrics
        full_metrics = {
            key: jnp.concatenate((full_metrics[key], metrics[key]))
            for key in full_metrics
        }
        if iteration % log_period == 0:
            with open(
                os.path.join(_last_metrics_dir, "metrics.pkl"), "wb"
            ) as file_to_save:
                pickle.dump(full_metrics, file_to_save)

        # Compute reeval metrics
        if num_reevals > 0 and iteration % log_period_reevals == 0:
            (
                reeval_repertoire,
                fit_reeval_repertoire,
                desc_reeval_repertoire,
                fit_var_repertoire,
                reeval_fit_var_repertoire,
                desc_var_repertoire,
                reeval_desc_var_repertoire,
                random_key,
            ) = reevaluation_fn(
                repertoire=repertoire,
                random_key=random_key,
            )
            reeval_metrics = reeval_metrics_fn(
                reeval_repertoire,
                fit_reeval_repertoire,
                desc_reeval_repertoire,
                fit_var_repertoire,
                reeval_fit_var_repertoire,
                desc_var_repertoire,
                reeval_desc_var_repertoire,
            )

            logger.warning(
                f"--- Current Reeval QD Score: "
                + f"{reeval_metrics['reeval_qd_score'][-1]:.2f}"
            )
            logger.warning(
                f"--- Current Reeval Coverage: "
                + f"{reeval_metrics['reeval_coverage'][-1]:.2f}%"
            )
            logger.warning(
                f"--- Current Reeval Max Fitness: "
                + f"{reeval_metrics['reeval_max_fitness'][-1]}"
            )

            # Add epoch and evals
            reeval_metrics["epoch"] = jnp.array([iteration])
            reeval_metrics["evals"] = jnp.array([total_evals])

            # Save reeval metrics
            full_reeval_metrics = {
                key: jnp.concatenate((full_reeval_metrics[key], reeval_metrics[key]))
                for key in full_reeval_metrics
            }
            with open(
                os.path.join(_last_metrics_dir, "reeval_metrics.pkl"), "wb"
            ) as file_to_save:
                pickle.dump(full_reeval_metrics, file_to_save)

            # Store the latest controllers of the reeval repertoires
            if store_repertoire and iteration % store_repertoire_log_period == 0:
                reeval_repertoire.save(path=_last_reeval_grid_dir + "/")
                fit_reeval_repertoire.save(path=_last_fit_reeval_grid_dir + "/")
                desc_reeval_repertoire.save(path=_last_desc_reeval_grid_dir + "/")
                fit_var_repertoire.save(path=_last_fit_var_grid_dir + "/")
                desc_var_repertoire.save(path=_last_desc_var_grid_dir + "/")
                reeval_fit_var_repertoire.save(path=_last_reeval_fit_var_grid_dir + "/")
                reeval_desc_var_repertoire.save(
                    path=_last_reeval_desc_var_grid_dir + "/"
                )

        # Store the latest controllers of the repertoire
        if store_repertoire and iteration % store_repertoire_log_period == 0:
            repertoire.save(path=_last_grid_dir + "/")

    total_time = time.time() - total_start_time
    logger.warning("--- Final metrics ---")
    logger.warning(f"Total time: {total_time:.2f}s")
    logger.warning(f"Algorithm time: {algorithm_time:.2f}s")
    logger.warning(f"QD Score: {metrics['qd_score'][-1]:.2f}")
    logger.warning(f"Coverage: {metrics['coverage'][-1]:.2f}%")

    # Save final metrics
    with open(os.path.join(_last_metrics_dir, "metrics.pkl"), "wb") as file_to_save:
        pickle.dump(full_metrics, file_to_save)
    # Save final repertoire
    repertoire.save(path=_last_grid_dir + "/")

    # Reeval final repertoire
    if num_reevals > 0:
        (
            reeval_repertoire,
            fit_reeval_repertoire,
            desc_reeval_repertoire,
            fit_var_repertoire,
            reeval_fit_var_repertoire,
            desc_var_repertoire,
            reeval_desc_var_repertoire,
            random_key,
        ) = reevaluation_fn(
            repertoire=repertoire,
            random_key=random_key,
        )
        reeval_metrics = reeval_metrics_fn(
            reeval_repertoire,
            fit_reeval_repertoire,
            desc_reeval_repertoire,
            fit_var_repertoire,
            reeval_fit_var_repertoire,
            desc_var_repertoire,
            reeval_desc_var_repertoire,
        )

        logger.warning(
            f"--- Reeval QD Score: {reeval_metrics['reeval_qd_score'][-1]:.2f}"
        )
        logger.warning(
            f"--- Reeval Coverage: {reeval_metrics['reeval_coverage'][-1]:.2f}%"
        )
        logger.warning(
            f"--- Reeval Max Fitness: {reeval_metrics['reeval_max_fitness'][-1]}"
        )

        # Add epoch and evals
        reeval_metrics["epoch"] = jnp.array([iteration])
        reeval_metrics["evals"] = jnp.array([total_evals])

        # Save reeval metrics
        full_reeval_metrics = {
            key: jnp.concatenate((full_reeval_metrics[key], reeval_metrics[key]))
            for key in full_reeval_metrics
        }
        with open(
            os.path.join(_last_metrics_dir, "reeval_metrics.pkl"), "wb"
        ) as file_to_save:
            pickle.dump(full_reeval_metrics, file_to_save)

        # Store the latest controllers of the reeval repertoires
        reeval_repertoire.save(path=_last_reeval_grid_dir + "/")
        fit_reeval_repertoire.save(path=_last_fit_reeval_grid_dir + "/")
        desc_reeval_repertoire.save(path=_last_desc_reeval_grid_dir + "/")
        fit_var_repertoire.save(path=_last_fit_var_grid_dir + "/")
        desc_var_repertoire.save(path=_last_desc_var_grid_dir + "/")
        reeval_fit_var_repertoire.save(path=_last_reeval_fit_var_grid_dir + "/")
        reeval_desc_var_repertoire.save(path=_last_reeval_desc_var_grid_dir + "/")

    return full_metrics, full_reeval_metrics, algorithm_time, random_key

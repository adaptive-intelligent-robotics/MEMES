import argparse
import copy
import os
import pickle
import traceback
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import yaml
from natsort import natsort_keygen
from qdax.utils.plotting import (
    plot_2d_map_elites_repertoire,
    plot_multidimensional_map_elites_grid,
)
from scipy.stats import ranksums
from yaml.loader import SafeLoader

####################################################
# Some global variable to easily modify the graphs #

parser = argparse.ArgumentParser()
parser.add_argument("--results", default="results", type=str)
parser.add_argument("--plots", default="plots", type=str)
parser.add_argument("--print-median", action="store_true", help="Print median.")
parser.add_argument("--convergence", action="store_true")
parser.add_argument("--final-interval-qd-score", action="store_true")
parser.add_argument("--loss-interval", action="store_true")
parser.add_argument(
    "--paper-metrics",
    action="store_true",
    help="Plot all additional metrics from the paper.",
)
parser.add_argument("--archives", action="store_true", help="Plot paper archives.")
parser.add_argument("--archives-solo", action="store_true", help="Plot all archives.")
parser.add_argument("--p-values", action="store_true", help="Write p-values.")
args = parser.parse_args()

# Display parameters
graph_palette = "colorblind"  # mako
font_size_big = 26
font_size_title = 30
font_size_small = 20
line_width = 4

# Number of columns for the legend at the bottom of the graph
graph_columns = 1

# Size of the margin to put the legend at the bottom of the graph
bottom_size = 0.20

# Environments order for plot
graph_env_order = ["arm", "hexapod_omni", "ant_uni", "anttrap"]

# Environments that are deterministics (for which reeval metrics = metrics)
env_deterministics = ["arm", "hexapod_omni"]

# Environments name correspondances
graph_env_names = {
    "anttrap": "AntTrap",
    "ant_uni": "Ant",
    "hexapod_omni": "Hexapod",
    "arm": "Arm",
}

# Environments max gen
graph_env_max_gen = {
    "anttrap": 10000,
    "ant_uni": 2000,
    "hexapod_omni": 2000,
    "arm": 2000,
}

# Environments num cells
graph_env_num_cells = {
    "anttrap": 2500,
    "ant_uni": 1296,
    "hexapod_omni": 2500,
    "arm": 2500,
}

# Environments BD correspondances for archives
graph_env_bds = {
    "anttrap": [[0, -8], [30, 8]],
    "ant_uni": [jnp.array([0, 0, 0, 0]), jnp.array([1, 1, 1, 1])],
    "hexapod_omni": [[-2, -2], [2, 2]],
    "arm": [[0, 0], [1, 1]],
}

# Environments line for BD-distance plot
graph_env_line = {
    "anttrap": 0.46,  # Average over 2 dimensions
    "ant_uni": 0.17,
    "hexapod_omni": 0.08,
    "arm": 0.02,
    "default": 0.02,
}

# Set up the graph names
new_names = {
    "me": "ME",
    "pga": "PGA-ME",
    "mees": "ME-ES",
    "cmame": "CMA-ME",
    "ns_es": "NS-ES",
    "nsr_es": "NSR-ES",
    "nsra_es": "NSRA-ES",
    "vanilla_es": "ES",
    "naive": "ME-Sampling",
    "memes": "MEMES (ours)",
    "all_memes": "MEMES-all (ours)",
    "ga_memes": "MEMES - GA",
    "memes_adapt_nov_arch": "MEMES - Novelty-archive",
    "memes_adapt_repertoire": "MEMES - Elites-archive",
    "sequential_memes": "MEMES - Sequential",
    "fix_reset_memes": "MEMES - Fix reset",
}
final_new_names = {
    "ME-batch-128.0": "ME - 128",
    "ME-batch-16384.0": "ME - 16384",
    "ME-batch-65536.0": "ME - 65536",
    "ME-Sampling-batch-512.0-smpl-32.0": "ME-Sampling - 32",
    "ME-Sampling-batch-32.0-smpl-512.0": "ME-Sampling - 512",
    "MEMES - Fix reset-num_generations_sample-50.0": "MEMES - Fix reset 50",
    "MEMES - Fix reset-num_generations_sample-20.0": "MEMES - Fix reset 20",
    "MEMES - Fix reset-num_generations_sample-100.0": "MEMES - Fix reset 100",
    "MEMES (ours)-batch-32.0": "MEMES - 32 (ours)",
    "MEMES (ours)-batch-128.0": "MEMES - 128 (ours)",
    "MEMES-all (ours)-batch-16416.0": "MEMES-all - 16384 (ours)",
    "MEMES-all (ours)-batch-32832.0": "MEMES-all - 32768 (ours)",
    "MEMES-all (ours)-batch-65664.0": "MEMES-all - 65536 (ours)",
}

# Order for legend
order = [
    "MEMES (ours)",
    "MEMES - 32 (ours)",
    "MEMES - 128 (ours)",
    "MEMES-all (ours)",
    "MEMES-all - 16384 (ours)",
    "MEMES-all - 32768 (ours)",
    "MEMES-all - 65536 (ours)",
    "ME - 128",
    "ME - 16384",
    "ME - 65536",
    "ME-ES",
    "PGA-ME",
    "CMA-ME",
    "ME-Sampling",
    "ME-Sampling - 32",
    "ME-Sampling - 512",
    "ES",
    "NS-ES",
    "NSR-ES",
    "NSRA-ES",
    "MEMES - Fix reset 100",
    "MEMES - Fix reset 50",
    "MEMES - Fix reset 20",
    "MEMES - Fix reset 10",
    "MEMES - Sequential",
    "MEMES - Novelty-archive",
    "MEMES - Elites-archive",
    "MEMES - GA",
]

# Never considered in the name
not_name = [
    "folder",
    "env_name",
    "plot_grid_log_period",
    "store_repertoire_log_period",
    "scan_batch_size",
    "plot_grid",
    "num_evaluations",
    "alg_name",
    "episode_length",
    "num_steps",
    "model_period",
    "surrogate_model_update_period",
    "grid_shape",
    "scan_novelty",
    "num_iterations",
    "adaptive_reset",
    "num_reevals",
    "log_period",
    "seed",
    "log_period_reevals",
    "fixed_init_state",
    "num_generations_stagnate",
]

# Correspondances to simplify name
name_dict = {
    "batch_size": "batch",
    "num_samples": "smpl",
    "sample_number": " es-smpl",
    "learning_rate": "lr",
    "sample_sigma": "sig",
    "l2_coefficient": "l2",
    "novelty_nearest_neighbors": "knn",
    "use_novelty_archive": "novelty_arch",
    "use_novelty_fifo": "novelty_fifo",
    "use_explore": "expl",
    "num_in_optimizer_steps": "in_opt_steps",
    "num_generations_stagnate": "stag",
}


##############
# Plot utils #


def isnan(num: float) -> bool:
    return num != num


def customize_axis(ax: Any) -> Any:
    """
    Customise axis for plots
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.tick_params(axis="y", length=0)
    # ax.get_yaxis().tick_left()

    # offset the spines
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))
    # put the grid behind
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linestyle="-", linewidth=1.5)
    return ax


def sub_plot(
    x: str,
    y: str,
    data: pd.DataFrame,
    ax: Any,
    xlabel: str,
    ylabel: str,
    scientific: bool,
) -> None:

    # Plot
    sns.lineplot(
        x=x,
        y=y,
        data=data,
        hue="algo",
        estimator=np.median,
        errorbar=("pi", 50),
        style="algo",
        ax=ax,
    )

    # Scientific units
    if scientific:
        if "qd_score" in y or "max_fitness" in y:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # Cosmetics
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    customize_axis(ax)


def sub_interval(
    y: str,
    data: pd.DataFrame,
    ax: Any,
    algos: np.ndarray,
    colors: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    scientific: bool,
) -> None:

    h = 0.6

    # Plot algorithms one by one
    algos = np.flip(algos)
    for alg_idx, algo in enumerate(algos):

        # Get color
        algo_color = colors[colors["Label"] == algo]["Color"].values[0]

        # Get values to plot
        values = np.expand_dims(data[data["algo"] == algo][y].values, axis=1)
        if len(values) == 1 and values[0] == 0.0:
            ax.barh(
                y=alg_idx,
                width=0.0,
                height=h,
                left=0.0,
                color=algo_color,
                alpha=0.0,
                label=algo,
            )
            continue
        aggregate_values = scipy.stats.trim_mean(
            values.squeeze(), proportiontocut=0.25, axis=None
        )
        aggregate_values_cis = scipy.stats.mstats.mquantiles(
            values.squeeze(), prob=[0.25, 0.75]
        )

        # Plot interval estimates
        lower, upper = aggregate_values_cis
        ax.barh(
            y=alg_idx,
            width=upper - lower,
            height=h,
            left=lower,
            color=algo_color,
            alpha=0.8,
            label=algo,
        )

        # Plot point estimates
        ax.vlines(
            x=aggregate_values,
            ymin=alg_idx - (8 * h / 16),
            ymax=alg_idx + (7.98 * h / 16),
            label=algo,
            color="k",
            alpha=1.0,
            linewidth=3,
        )

        # Plot datapoints
        # datapoints = np.ones_like(values.squeeze()) * alg_idx
        # ax.scatter(y=datapoints, x=values.squeeze(), marker='o', color=algo_color, s=50, alpha=0.25)

    # Scientific units
    if scientific:
        if "qd_score" in y or "max_fitness" in y:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # Cosmetics
    ax.set_xlabel(xlabel)
    ax.set_yticks(list(range(len(algos))))
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.tick_params(axis="y", which="both", length=0.0)
    ax.tick_params(axis="x", which="both", length=6)
    if ylabel is not None:
        ax.set_yticklabels(algos)
        # ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])

    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))


def extract_algo(data: pd.DataFrame, algos: str, columns: List) -> pd.DataFrame:
    sub_data = data[(data["algo"] == algos)].reset_index(drop=True)
    if sub_data.empty:
        return sub_data
    sub_data = sub_data.sort_values(columns, key=natsort_keygen(), ignore_index=True)
    return sub_data


def extract_nonalgo(data: pd.DataFrame, algos: str, columns: List) -> pd.DataFrame:
    sub_data = data[~(data["algo"] == algos)].reset_index(drop=True)
    if sub_data.empty:
        return sub_data
    sub_data = sub_data.sort_values(columns, key=natsort_keygen(), ignore_index=True)
    return sub_data


def sort_data(
    data: pd.DataFrame,
    columns: List,
    order: List,
) -> pd.DataFrame:
    final_data = extract_algo(data, order[0], columns=columns)
    left_data = extract_nonalgo(data, order[0], columns=columns)
    added_names = order[0]
    for i in range(1, len(order)):
        final_data = pd.concat(
            [
                final_data,
                extract_algo(left_data, order[i], columns=columns),
            ],
            ignore_index=True,
        )
        added_names += "|" + order[i]
        left_data = extract_nonalgo(data, added_names, columns=columns)
    final_data = pd.concat([final_data, left_data], ignore_index=True)
    return final_data


#######################
# Main plot functions #


def all_env_plot(
    x: str,
    xlabel: str,
    data: pd.DataFrame,
    color_data: pd.DataFrame,
    file_name: str,
    columns: List,
    names: List,
    hline_values: Dict = None,
    split_column=None,
    height=18,
    width=32,
    interval: bool = False,
    scientific: bool = False,
) -> None:

    assert len(columns) == len(
        names
    ), "!!!ERROR!!! columns and names do not have same size."

    # Get all env names
    env_names = data["env_name"].drop_duplicates().values
    ncols = max(2, len(env_names))
    nrows = len(columns)
    if split_column is not None:
        if nrows > 1:
            assert 0, "!!!ERROR!!! more than 1 line on multi lines not implemented yet."
        ncols = ncols // split_column
        nrows = nrows * split_column

    # Order envs
    ordered_env_names = []
    for env_name in graph_env_order:
        if env_name in env_names:
            ordered_env_names.append(env_name)
    for env_name in env_names:
        if env_name not in ordered_env_names:
            print(
                "!!!WARNING!!!",
                env_name,
                "not in environment order, adding it as last.",
            )
            ordered_env_names.append(env_name)
    env_names = ordered_env_names

    # Plot parameters
    params = {
        "lines.linewidth": line_width,
        "axes.titlesize": font_size_title,
        "axes.labelsize": font_size_big,
        "legend.fontsize": font_size_big,
        "xtick.labelsize": font_size_small,
        "ytick.labelsize": font_size_small,
        "text.usetex": False,
    }
    mpl.rcParams.update(params)

    # Create subplots
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(width, height),  # sharex="col"
    )

    # Plot one env per column
    all_handles: List = []
    all_labels: List = []
    for col in range(ncols):
        if col >= len(env_names):
            continue

        # Plot one metric per row
        for subplot in range(nrows):

            # Select axis
            if nrows == 1 or ncols == 1:
                ax = axes[col]
            else:
                ax = axes[subplot, col]

            # Select idx for metric and env
            if split_column is not None:
                n_metric = 0
                n_env = subplot + 2 * col
            else:
                n_metric = subplot
                n_env = col

            env_data = data[data["env_name"] == env_names[n_env]]

            # Set palette
            algos = env_data["algo"].drop_duplicates().values
            env_color_data = color_data[color_data["Label"].isin(algos)]
            env_palette = env_color_data["Color"].values
            sns.set_palette(env_palette)

            # Plot
            if interval:
                sub_interval(
                    y=columns[n_metric],
                    data=env_data,
                    ax=ax,
                    algos=algos,
                    colors=env_color_data,
                    xlabel=names[n_metric]
                    if not split_column
                    else (names[n_metric] if subplot == 1 else None),
                    ylabel=names[n_metric] if col == 0 else None,
                    scientific=scientific,
                )
            else:
                sub_plot(
                    x=x,
                    y=columns[n_metric],
                    data=env_data,
                    ax=ax,
                    xlabel=xlabel,
                    ylabel=names[n_metric] if col == 0 else None,
                    scientific=scientific,
                )

            # Accumulate all the legends
            if not interval:
                handles, labels = ax.get_legend_handles_labels()
                for i in range(len(labels)):
                    if labels[i] not in all_labels:
                        all_handles.append(handles[i])
                        all_labels.append(labels[i])
                ax.legend_.remove()

            # Add env as title to first subplot
            if n_metric == 0:
                title = env_names[n_env]
                if title in graph_env_names.keys():
                    title = graph_env_names[title]
                else:
                    print(
                        "!!!WARNING!!!",
                        title,
                        "is not in graph_env_names, keeping this name.",
                    )
                ax.set_title(title)

            # Add hline
            if hline_values is not None:
                env_line = hline_values[env_names[n_env]]
                ax.axhline(env_line, c="r", linestyle="--", linewidth=3)

    # Spacing between subplots
    if not interval:
        plt.tight_layout(h_pad=1.70)
    else:
        plt.tight_layout(h_pad=2.0, w_pad=3.0)

    # Add legend below the plot
    if not interval:
        fig.subplots_adjust(bottom=bottom_size)
        fig.legend(
            handles=all_handles,
            labels=all_labels,
            loc="lower center",
            frameon=False,
            ncol=graph_columns if ncols > 1 else 2,
        )

    # Save plot
    plt.savefig(file_name)
    plt.close()


############
# P-values #


def p_value_ranksum(
    frame: pd.DataFrame, reference_label: str, compare_label: str, stat: str
) -> Any:
    """Compute one p-value for one reference and one compare label for a given stat."""

    reference_frame = frame[frame["algo"] == reference_label]
    reference_max_gen = reference_frame["gen"].max()
    reference_frame = reference_frame[reference_frame["gen"] == reference_max_gen]

    compare_frame = frame[frame["algo"] == compare_label]
    compare_max_gen = compare_frame["gen"].max()
    compare_frame = compare_frame[compare_frame["gen"] == compare_max_gen]

    _, p = ranksums(
        reference_frame[stat].to_numpy(),
        compare_frame[stat].to_numpy(),
    )
    return p


def compute_p_values(
    frame: pd.DataFrame,
    file_name: str,
    stat: str,
) -> pd.DataFrame:
    """Write p-value of stat in a table."""

    p_frame = pd.DataFrame(columns=["Reference label", "Label", "p-value"])
    labels = frame["algo"].drop_duplicates().values

    # For each labels-couple
    for reference_label in labels:
        for compare_label in labels:
            p_frame = pd.concat(
                [
                    p_frame,
                    pd.DataFrame.from_dict(
                        {
                            "Reference label": [reference_label],
                            "Label": [compare_label],
                            "p-value": [
                                p_value_ranksum(
                                    frame, reference_label, compare_label, stat
                                )
                            ],
                        }
                    ),
                ],
                ignore_index=True,
            )
    # When writting in frame, writting it as double entry table
    written_p_frame = p_frame.pivot(
        index="Reference label", columns="Label", values="p-value"
    )
    p_file = open(file_name, "a")
    p_file.write(written_p_frame.to_markdown())
    p_file.close()

    # Still returning the frame just in case
    return p_frame


################
# Find results #

# Opening all config files in the results folder
print("\n\nOpening config files")
folders = [
    root
    for root, dirs, files in os.walk(args.results)
    for name in files
    if "config.yaml" in name
]
assert len(folders) > 0, "\n!!!ERROR!!! No config files in result folder.\n"

# Go through folders to remove .hydra from path
for i in range(len(folders)):
    folders[i] = folders[i][: -len(".hydra")]

# Create a dataframe with the parameters of the config files
config_frame = pd.DataFrame()
for folder in folders:
    with open(os.path.join(folder, ".hydra/config.yaml")) as f:
        config = yaml.load(f, Loader=SafeLoader)
        for key in config.keys():
            config[key] = [config[key]]
        config["folder"] = [folder]

        config_frame = pd.concat(
            [config_frame, pd.DataFrame.from_dict(config)], ignore_index=True
        )
print("\nFound", config_frame.shape[0], "results folder")

################
# Name results #

# Create results folder if needed
try:
    if not os.path.exists(args.plots):
        os.mkdir(args.plots)
    if not os.path.exists(f"{args.plots}_csv"):
        os.mkdir(f"{args.plots}_csv")
except Exception:
    if not args.no_traceback:
        print("\n!!!WARNING!!! Cannot create folders for plots.")
        print(traceback.format_exc(-1))

print("\nSetting up algorithms names")

# First use name from new_name
algos = []
for line in range(config_frame.shape[0]):
    original_name = config_frame["alg_name"][line]
    if original_name in new_names.keys():
        algo = new_names[original_name]
    else:
        algo = original_name
    algos.append(algo)
config_frame["algo"] = algos

# Second get parameters that are different
use_in_name_dict = {}
for name in config_frame["algo"].drop_duplicates().values:
    sub_config_frame = config_frame[config_frame["algo"] == name]
    use_in_name = []
    for column in sub_config_frame.columns:
        if column not in not_name and not sub_config_frame[column].dropna().empty:
            ref = str(sub_config_frame[column].dropna().values[0])
            if any(
                [str(val) != ref for val in sub_config_frame[column].dropna().values]
            ):
                use_in_name.append(column)
    use_in_name_dict[name] = use_in_name

# Third add parameters to name
algos = []
for line in range(config_frame.shape[0]):
    algo = config_frame["algo"][line]

    # Build name for parameters that change across baselines
    use_in_name = use_in_name_dict[algo]
    for name in use_in_name:
        # Only if parameters is not nan
        if not isnan(config_frame[name][line]):
            name_simpl = name_dict[name] if name in name_dict.keys() else name
            if type(config_frame[name][line]) != bool:
                algo += "-" + name_simpl + "-" + str(config_frame[name][line])
            elif type(config_frame[name][line]) == bool and config_frame[name][line]:
                algo += "-" + name_simpl
    algos.append(algo)

# Fourth check that this does not corresponds to final_new_names
for idx in range(len(algos)):
    if algos[idx] in final_new_names:
        algos[idx] = final_new_names[algos[idx]]
config_frame["algo"] = algos

print("\n    Final names for graphs:")
print(config_frame["algo"].drop_duplicates())

########################
# Opening metric files #

print("\n    Opening metric files")
metrics_frame = pd.DataFrame()
reeval_metrics_frame = pd.DataFrame()
final_metrics_frame = pd.DataFrame()
loss_metrics_frame = pd.DataFrame()
var_metrics_frame = pd.DataFrame()
for line in range(config_frame.shape[0]):
    metrics_file = os.path.join(
        config_frame["folder"][line], "checkpoints/last_metrics/metrics.pkl"
    )
    reeval_metrics_file = os.path.join(
        config_frame["folder"][line], "checkpoints/last_metrics/reeval_metrics.pkl"
    )

    try:

        # Load metrics
        with open(metrics_file, "rb") as f:
            metrics = pickle.load(f)
        metrics = pd.DataFrame.from_dict(metrics)

        # Add x axis
        num_gen = jnp.arange(jnp.shape(metrics["coverage"])[0])
        metrics["gen"] = num_gen

        # Add necessary informations
        algo = config_frame["algo"][line]
        env_name = config_frame["env_name"][line]
        metrics["env_name"] = env_name
        metrics["algo"] = algo
        metrics["line"] = line

        # If xlim given remove all points after
        if env_name in graph_env_max_gen.keys():
            metrics = metrics[metrics["gen"] <= graph_env_max_gen[env_name]]

        # Add to overall metrics frame
        metrics_frame = pd.concat([metrics_frame, metrics], ignore_index=True)

        if env_name in env_deterministics:

            reeval_metrics = metrics
            reeval_metrics["reeval_qd_score"] = metrics["qd_score"]
            reeval_metrics["reeval_coverage"] = metrics["coverage"]
            reeval_metrics["reeval_max_fitness"] = metrics["max_fitness"]

            reeval_metrics["desc_var_qd_score"] = jnp.zeros_like(num_gen)
            reeval_metrics["desc_var_coverage"] = jnp.zeros_like(num_gen)
            reeval_metrics["desc_var_max_fitness"] = jnp.zeros_like(num_gen)
            reeval_metrics["desc_var_min_fitness"] = jnp.zeros_like(num_gen)

            reeval_metrics["reeval_desc_var_qd_score"] = jnp.zeros_like(num_gen)
            reeval_metrics["reeval_desc_var_coverage"] = jnp.zeros_like(num_gen)
            reeval_metrics["reeval_desc_var_max_fitness"] = jnp.zeros_like(num_gen)
            reeval_metrics["reeval_desc_var_min_fitness"] = jnp.zeros_like(num_gen)

            reeval_metrics["fit_var_qd_score"] = jnp.zeros_like(num_gen)
            reeval_metrics["fit_var_coverage"] = jnp.zeros_like(num_gen)
            reeval_metrics["fit_var_max_fitness"] = jnp.zeros_like(num_gen)
            reeval_metrics["fit_var_min_fitness"] = jnp.zeros_like(num_gen)

            reeval_metrics["reeval_fit_var_qd_score"] = jnp.zeros_like(num_gen)
            reeval_metrics["reeval_fit_var_coverage"] = jnp.zeros_like(num_gen)
            reeval_metrics["reeval_fit_var_max_fitness"] = jnp.zeros_like(num_gen)
            reeval_metrics["reeval_fit_var_min_fitness"] = jnp.zeros_like(num_gen)

            # Add to overall reeval metrics frame
            reeval_metrics_frame = pd.concat(
                [reeval_metrics_frame, reeval_metrics], ignore_index=True
            )

        # Do the same for reeval metrics
        elif (
            "num_reevals" in config_frame.columns
            and not isnan(config_frame["num_reevals"][line])
            and config_frame["num_reevals"][line] > 0
        ):

            # Load reeval metrics
            with open(reeval_metrics_file, "rb") as f:
                reeval_metrics = pickle.load(f)
            reeval_metrics = pd.DataFrame.from_dict(reeval_metrics)

            # Add x axis to reeval
            num_gen = (
                jnp.arange(jnp.shape(reeval_metrics["reeval_coverage"])[0])
                * config_frame["log_period_reevals"][line]
            )
            reeval_metrics["gen"] = num_gen

            # Add necessary informations
            reeval_metrics["env_name"] = env_name
            reeval_metrics["algo"] = algo
            reeval_metrics["line"] = line

            # If xlim given remove all points after
            if env_name in graph_env_max_gen.keys():
                reeval_metrics = reeval_metrics[
                    reeval_metrics["gen"] <= graph_env_max_gen[env_name]
                ]

            # Add to overall reeval metrics frame
            reeval_metrics_frame = pd.concat(
                [reeval_metrics_frame, reeval_metrics], ignore_index=True
            )

        else:
            print(f"WARNING {algo} in {env_name} has no reeval.")

            reeval_metrics = metrics
            reeval_metrics["reeval_qd_score"] = metrics["qd_score"]
            reeval_metrics["reeval_coverage"] = metrics["coverage"]
            reeval_metrics["reeval_max_fitness"] = metrics["max_fitness"]

            reeval_metrics["desc_var_qd_score"] = jnp.zeros_like(num_gen)
            reeval_metrics["desc_var_coverage"] = jnp.zeros_like(num_gen)
            reeval_metrics["desc_var_max_fitness"] = jnp.zeros_like(num_gen)
            reeval_metrics["desc_var_min_fitness"] = jnp.zeros_like(num_gen)

            reeval_metrics["reeval_desc_var_qd_score"] = jnp.zeros_like(num_gen)
            reeval_metrics["reeval_desc_var_coverage"] = jnp.zeros_like(num_gen)
            reeval_metrics["reeval_desc_var_max_fitness"] = jnp.zeros_like(num_gen)
            reeval_metrics["reeval_desc_var_min_fitness"] = jnp.zeros_like(num_gen)

            reeval_metrics["fit_var_qd_score"] = jnp.zeros_like(num_gen)
            reeval_metrics["fit_var_coverage"] = jnp.zeros_like(num_gen)
            reeval_metrics["fit_var_max_fitness"] = jnp.zeros_like(num_gen)
            reeval_metrics["fit_var_min_fitness"] = jnp.zeros_like(num_gen)

            reeval_metrics["reeval_fit_var_qd_score"] = jnp.zeros_like(num_gen)
            reeval_metrics["reeval_fit_var_coverage"] = jnp.zeros_like(num_gen)
            reeval_metrics["reeval_fit_var_max_fitness"] = jnp.zeros_like(num_gen)
            reeval_metrics["reeval_fit_var_min_fitness"] = jnp.zeros_like(num_gen)

            # Add to overall reeval metrics frame
            reeval_metrics_frame = pd.concat(
                [reeval_metrics_frame, reeval_metrics], ignore_index=True
            )

        max_gen = max(metrics["gen"])
        reeval_max_gen = max(reeval_metrics["gen"])

        # Get the final metrics
        final_metrics: Dict = {}
        final_metrics["qd_score"] = metrics[metrics["gen"] == max_gen][
            "qd_score"
        ].values[0]
        final_metrics["coverage"] = metrics[metrics["gen"] == max_gen][
            "coverage"
        ].values[0]
        final_metrics["max_fitness"] = metrics[metrics["gen"] == max_gen][
            "max_fitness"
        ].values[0]

        # Reeval metrics
        final_metrics["reeval_qd_score"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["reeval_qd_score"].values[0]
        final_metrics["reeval_coverage"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["reeval_coverage"].values[0]
        final_metrics["reeval_max_fitness"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["reeval_max_fitness"].values[0]

        # Add necessary informations
        final_metrics["env_name"] = [env_name]
        final_metrics["algo"] = [algo]
        final_metrics["line"] = [line]

        # Add to overall final metrics frame
        final_metrics = pd.DataFrame.from_dict(final_metrics)
        final_metrics_frame = pd.concat(
            [final_metrics_frame, final_metrics], ignore_index=True
        )

        # Get the loss metrics
        loss_metrics: Dict = {}

        loss_metrics["loss_qd_score"] = (
            (final_metrics["qd_score"] - final_metrics["reeval_qd_score"])
            / final_metrics["qd_score"]
            * 100
        )
        loss_metrics["loss_coverage"] = (
            (final_metrics["coverage"] - final_metrics["reeval_coverage"])
            / final_metrics["coverage"]
            * 100
        )
        loss_metrics["loss_max_fitness"] = (
            (final_metrics["max_fitness"] - final_metrics["reeval_max_fitness"])
            / final_metrics["max_fitness"]
            * 100
        )

        # Add necessary informations
        loss_metrics["env_name"] = [env_name]
        loss_metrics["algo"] = [algo]
        loss_metrics["line"] = [line]

        # Add to overall loss metrics frame
        loss_metrics = pd.DataFrame.from_dict(loss_metrics)
        loss_metrics_frame = pd.concat(
            [loss_metrics_frame, loss_metrics], ignore_index=True
        )

        # Get the var metrics
        var_metrics: Dict = {}

        var_metrics["desc_var_qd_score"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["desc_var_qd_score"].values[0]
        var_metrics["desc_var_coverage"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["desc_var_coverage"].values[0]
        var_metrics["desc_var_max_fitness"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["desc_var_max_fitness"].values[0]
        var_metrics["desc_var_min_fitness"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["desc_var_min_fitness"].values[0]

        var_metrics["reeval_desc_var_qd_score"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["reeval_desc_var_qd_score"].values[0]
        var_metrics["reeval_desc_var_coverage"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["reeval_desc_var_coverage"].values[0]
        var_metrics["reeval_desc_var_max_fitness"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["reeval_desc_var_max_fitness"].values[0]
        var_metrics["reeval_desc_var_min_fitness"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["reeval_desc_var_min_fitness"].values[0]

        var_metrics["fit_var_qd_score"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["fit_var_qd_score"].values[0]
        var_metrics["fit_var_coverage"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["fit_var_coverage"].values[0]
        var_metrics["fit_var_max_fitness"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["fit_var_max_fitness"].values[0]
        var_metrics["fit_var_min_fitness"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["fit_var_min_fitness"].values[0]

        var_metrics["reeval_fit_var_qd_score"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["reeval_fit_var_qd_score"].values[0]
        var_metrics["reeval_fit_var_coverage"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["reeval_fit_var_coverage"].values[0]
        var_metrics["reeval_fit_var_max_fitness"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["reeval_fit_var_max_fitness"].values[0]
        var_metrics["reeval_fit_var_min_fitness"] = reeval_metrics[
            reeval_metrics["gen"] == reeval_max_gen
        ]["reeval_fit_var_min_fitness"].values[0]

        # Add necessary informations
        var_metrics["env_name"] = [env_name]
        var_metrics["algo"] = [algo]
        var_metrics["line"] = [line]

        # Add to overall var metrics frame
        var_metrics = pd.DataFrame.from_dict(var_metrics)
        var_metrics_frame = pd.concat(
            [var_metrics_frame, var_metrics], ignore_index=True
        )

    except Exception:
        print("\n!!!WARNING!!! Could not read:", metrics_file, reeval_metrics_file)
        print(traceback.format_exc(-1))

# Add 0 values for non-existing algos
print(" \n!!!WARNING!!! Adding empty lines for non-existing replications.")
all_algos = metrics_frame["algo"].drop_duplicates().values
line_idx = metrics_frame["line"].max() + 1
for env_name in metrics_frame["env_name"].drop_duplicates().values:
    sub_metrics_frame = metrics_frame[metrics_frame["env_name"] == env_name]
    env_algos = sub_metrics_frame["algo"].drop_duplicates().values
    env_gen = sub_metrics_frame["gen"].max()
    for algo in all_algos:
        if algo not in env_algos:
            new_line = pd.Series(0, index=metrics_frame.columns)
            new_line["env_name"] = env_name
            new_line["algo"] = algo
            new_line["line"] = line_idx
            new_line["gen"] = env_gen
            metrics_frame = pd.concat(
                [metrics_frame, new_line.to_frame().T], ignore_index=True
            )

            new_line = pd.Series(0, index=reeval_metrics_frame.columns)
            new_line["env_name"] = env_name
            new_line["algo"] = algo
            new_line["line"] = line_idx
            reeval_metrics_frame = pd.concat(
                [reeval_metrics_frame, new_line.to_frame().T], ignore_index=True
            )

            new_line = pd.Series(0, index=final_metrics_frame.columns)
            new_line["env_name"] = env_name
            new_line["algo"] = algo
            new_line["line"] = line_idx
            final_metrics_frame = pd.concat(
                [final_metrics_frame, new_line.to_frame().T], ignore_index=True
            )

            new_line = pd.Series(0, index=loss_metrics_frame.columns)
            new_line["env_name"] = env_name
            new_line["algo"] = algo
            new_line["line"] = line_idx
            loss_metrics_frame = pd.concat(
                [loss_metrics_frame, new_line.to_frame().T], ignore_index=True
            )

            new_line = pd.Series(0, index=var_metrics_frame.columns)
            new_line["env_name"] = env_name
            new_line["algo"] = algo
            new_line["line"] = line_idx
            var_metrics_frame = pd.concat(
                [var_metrics_frame, new_line.to_frame().T], ignore_index=True
            )

            line_idx = line_idx + 1

# Sort
print("\n    Sorting metrics frames")
config_frame = sort_data(config_frame, ["env_name", "algo"], order)
metrics_frame = sort_data(metrics_frame, ["env_name", "algo", "line"], order)
reeval_metrics_frame = sort_data(
    reeval_metrics_frame, ["env_name", "algo", "line"], order
)
final_metrics_frame = sort_data(
    final_metrics_frame, ["env_name", "algo", "line"], order
)
loss_metrics_frame = sort_data(loss_metrics_frame, ["env_name", "algo", "line"], order)
var_metrics_frame = sort_data(var_metrics_frame, ["env_name", "algo", "line"], order)

# Create color palette
labels = metrics_frame["algo"].drop_duplicates().values
colors = sns.color_palette(graph_palette, len(labels))
color_frame = pd.DataFrame(data={"Label": labels, "Color": colors})


######################################
# Displaying median values for table #

if args.print_median:
    for env_name in metrics_frame["env_name"].drop_duplicates().values:
        env_metrics_frame = metrics_frame[metrics_frame["env_name"] == env_name]
        env_reeval_metrics_frame = reeval_metrics_frame[
            reeval_metrics_frame["env_name"] == env_name
        ]
        for algo in env_metrics_frame["algo"].drop_duplicates().values:
            env_algo_frame = env_metrics_frame[(env_metrics_frame["algo"] == algo)]
            env_algo_frame = env_algo_frame.sort_values(
                ["gen"], ignore_index=True, key=natsort_keygen()
            )
            reeval_env_algo_frame = env_reeval_metrics_frame[
                (env_reeval_metrics_frame["algo"] == algo)
            ]
            reeval_env_algo_frame = reeval_env_algo_frame.sort_values(
                ["gen"], ignore_index=True, key=natsort_keygen()
            )

            # Main metrics
            max_gen = max(env_algo_frame["gen"])
            median_qd_score = env_algo_frame[env_algo_frame["gen"] == max_gen][
                "qd_score"
            ].median()
            median_coverage = env_algo_frame[env_algo_frame["gen"] == max_gen][
                "coverage"
            ].median()
            median_max_fitness = env_algo_frame[env_algo_frame["gen"] == max_gen][
                "max_fitness"
            ].median()

            # Reeval metrics
            max_gen = max(reeval_env_algo_frame["gen"])
            reeval_median_qd_score = reeval_env_algo_frame[
                reeval_env_algo_frame["gen"] == max_gen
            ]["reeval_qd_score"].median()
            reeval_median_coverage = reeval_env_algo_frame[
                reeval_env_algo_frame["gen"] == max_gen
            ]["reeval_coverage"].median()
            reeval_median_max_fitness = reeval_env_algo_frame[
                reeval_env_algo_frame["gen"] == max_gen
            ]["reeval_max_fitness"].median()

            # Print
            print("For", env_name, "and", algo, ":")
            print("Median qd_score:", median_qd_score)
            print("Median reeval qd_score:", reeval_median_qd_score)
            print("Median coverage:", median_coverage)
            print("Median reeval coverage:", reeval_median_coverage)
            print("Median max_fitness:", median_max_fitness)
            print("Median reeval max_fitness:", reeval_median_max_fitness)
            print("\n")


########################
# Plotting metric files #

# Plot all final value graphs
print("\n    Plotting final metrics as interval.")
try:
    file_name = f"{args.plots}/all_final.svg"
    all_env_plot(
        x="env_name",
        xlabel="",
        data=final_metrics_frame,
        color_data=color_frame,
        file_name=file_name,
        columns=["qd_score", "coverage", "max_fitness"],
        names=["QD-Score", "Coverage (%)", "Max-Fitness"],
        interval=True,
        scientific=True,
        height=18,
        width=28,
    )
    file_name = f"{args.plots}/all_reeval_final.svg"
    all_env_plot(
        x="env_name",
        xlabel="",
        data=final_metrics_frame,
        color_data=color_frame,
        file_name=file_name,
        columns=["reeval_qd_score", "reeval_coverage", "reeval_max_fitness"],
        names=["QD-Score", "Coverage (%)", "Max-Fitness"],
        interval=True,
        scientific=True,
        height=18,
        width=28,
    )
except Exception:
    print("\n!!!WARNING!!! Cannot plot final graphs.")
    print(traceback.format_exc(-1))

# Plot all convergence graphs
if args.convergence:
    print("\n    Plotting convergence metrics.")
    try:
        file_name = f"{args.plots}/all_generations.svg"
        all_env_plot(
            x="gen",
            xlabel="Generations",
            data=metrics_frame,
            color_data=color_frame,
            file_name=file_name,
            columns=["qd_score", "coverage", "max_fitness"],
            names=["QD-Score", "Coverage (%)", "Max-Fitness"],
        )
        file_name = f"{args.plots}/all_reeval_generations.svg"
        all_env_plot(
            "gen",
            "Generations",
            reeval_metrics_frame,
            color_frame,
            file_name,
            columns=["reeval_qd_score", "reeval_coverage", "reeval_max_fitness"],
            names=[
                "QD-Score",
                "Coverage (%)",
                "Max-Fitness",
            ],
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot generation graphs.")
        print(traceback.format_exc(-1))

# Plot all final value qd-score only graphs
if args.final_interval_qd_score:
    print("\n    Plotting final qd-score as interval.")
    try:
        file_name = f"{args.plots}/all_final_qd_score.svg"
        all_env_plot(
            x="env_name",
            xlabel="",
            data=final_metrics_frame,
            color_data=color_frame,
            file_name=file_name,
            columns=["qd_score"],
            names=["QD-Score"],
            interval=True,
            scientific=True,
            height=8,
            width=24,
        )
        file_name = f"{args.plots}/all_final_qd_score_square.svg"
        all_env_plot(
            x="env_name",
            xlabel="",
            data=final_metrics_frame,
            color_data=color_frame,
            file_name=file_name,
            columns=["qd_score"],
            names=["QD-Score"],
            interval=True,
            scientific=True,
            split_column=2,
            height=10,
            width=15,
        )
        file_name = f"{args.plots}/all_final_reeval_qd_score.svg"
        all_env_plot(
            x="env_name",
            xlabel="",
            data=final_metrics_frame,
            color_data=color_frame,
            file_name=file_name,
            columns=["reeval_qd_score"],
            names=["QD-Score"],
            interval=True,
            scientific=True,
            height=8,
            width=24,
        )
        file_name = f"{args.plots}/all_final_reeval_qd_score_square.svg"
        all_env_plot(
            x="env_name",
            xlabel="",
            data=final_metrics_frame,
            color_data=color_frame,
            file_name=file_name,
            columns=["reeval_qd_score"],
            names=["QD-Score"],
            interval=True,
            scientific=True,
            split_column=2,
            height=10,
            width=15,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot final qd-score graphs.")
        print(traceback.format_exc(-1))

# Plot all loss graphs
if args.loss_interval:
    print("\n    Plotting loss as interval.")
    valid_algos = [
        "MEMES",
        "MEMES (ours)",
        "MEMES-all",
        "MEMES-all (ours)",
        "ME - 128",
        "ME - 16384",
        "ME-Sampling",
    ]
    try:
        sub_metrics_frame = loss_metrics_frame[
            ~loss_metrics_frame["env_name"].isin(env_deterministics)
        ]
        sub_metrics_frame = sub_metrics_frame[
            sub_metrics_frame["algo"].isin(valid_algos)
        ]
        file_name = f"{args.plots}/all_loss_interval.svg"
        all_env_plot(
            x="env_name",
            xlabel="",
            data=sub_metrics_frame,
            color_data=color_frame,
            file_name=file_name,
            columns=["loss_qd_score", "loss_coverage", "loss_max_fitness"],
            names=["QD-Score loss (%)", "Coverage loss (%)", "Max-Fitness loss (%)"],
            interval=True,
            width=16,
        )
        file_name = f"{args.plots}/all_qd_score_loss_interval.svg"
        all_env_plot(
            x="env_name",
            xlabel="",
            data=sub_metrics_frame,
            color_data=color_frame,
            file_name=file_name,
            columns=["loss_qd_score"],
            names=["QD-Score loss (%)"],
            interval=True,
            height=6,
            width=16,
        )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot loss graphs.")
        print(traceback.format_exc(-1))

# Plot variation usage
if args.paper_metrics:
    print("\n    Plotting proportion explore metrics.")
    if "proportion_explore" in metrics_frame.columns:
        try:
            # Create moving average frame
            frame = pd.DataFrame()
            for env_name in metrics_frame["env_name"].drop_duplicates().values:
                env_metrics_frame = metrics_frame[metrics_frame["env_name"] == env_name]
                for algo in env_metrics_frame["algo"].drop_duplicates().values:
                    sub_frame = env_metrics_frame[(env_metrics_frame["algo"] == algo)]
                    for line in sub_frame["line"].drop_duplicates().values:
                        sub_sub_frame = sub_frame[(sub_frame["line"] == line)][
                            [
                                "gen",
                                "env_name",
                                "algo",
                                "proportion_explore",
                                "explore_usage",
                                "exploit_usage",
                            ]
                        ]
                        sub_sub_frame = sub_sub_frame.sort_values(
                            ["gen"], ignore_index=True, key=natsort_keygen()
                        )
                        add_frame = copy.copy(sub_sub_frame)
                        add_frame.dropna(inplace=True)
                        add_frame["explore_usage_smooth"] = (
                            add_frame["explore_usage"].rolling(20).mean()
                        )
                        add_frame["exploit_usage_smooth"] = (
                            add_frame["exploit_usage"].rolling(20).mean()
                        )
                        add_frame.dropna(inplace=True)

                        frame = pd.concat([frame, add_frame], ignore_index=True)

            # Plot
            file_name = f"{args.plots}/variation_usage_generations.svg"
            all_env_plot(
                "gen",
                "Generations",
                frame,
                color_frame,
                file_name,
                columns=["exploit_usage_smooth", "explore_usage_smooth"],
                names=["Explore emitter addition", "Exploit emitter addition"],
            )  # "proportion_explore" "Proportion of explore / exploit"

        except Exception:
            print("\n!!!WARNING!!! Cannot plot variation graphs.")
            print(traceback.format_exc(-1))

# Plot parent-offspring BD-distance
if args.paper_metrics:
    print("\n    Plotting parent distance metrics.")
    if "parents_distance" in metrics_frame.columns:
        valid_algos = [
            "MEMES",
            "MEMES (ours)",
            "PGA-ME",
            "ME - 128",
            "ME - 16384",
            "ME-Sampling",
        ]
        try:
            hline_values: Dict = {}
            frame = pd.DataFrame()
            for env_name in metrics_frame["env_name"].drop_duplicates().values:
                env_metrics_frame = metrics_frame[metrics_frame["env_name"] == env_name]
                if env_name in graph_env_line:
                    hline_values[env_name] = graph_env_line[env_name]
                else:
                    print(
                        "!!!WARNING!!!",
                        env_name,
                        "is not in env list for distance, using default.",
                    )
                    hline_values[env_name] = graph_env_line["default"]
                for algo in env_metrics_frame["algo"].drop_duplicates().values:
                    if algo not in valid_algos:
                        continue
                    sub_frame = env_metrics_frame[(env_metrics_frame["algo"] == algo)]
                    for line in sub_frame["line"].drop_duplicates().values:
                        sub_sub_frame = sub_frame[(sub_frame["line"] == line)][
                            ["gen", "env_name", "algo", "parents_distance"]
                        ]
                        sub_sub_frame = sub_sub_frame.sort_values(
                            ["gen"], ignore_index=True, key=natsort_keygen()
                        )
                        add_frame = copy.copy(sub_sub_frame)
                        add_frame.dropna(inplace=True)
                        add_frame["parents_distance_smooth"] = (
                            add_frame["parents_distance"].rolling(20).mean()
                        )
                        add_frame.dropna(inplace=True)

                        frame = pd.concat([frame, add_frame], ignore_index=True)

            # Plot
            file_name = f"{args.plots}/parents_distance_generations.svg"
            all_env_plot(
                "gen",
                "Generations",
                frame,
                color_frame,
                file_name,
                columns=["parents_distance"],
                names=["Parent-offspring BD distance"],
                hline_values=hline_values,
                split_column=2
                if (len(metrics_frame["env_name"].drop_duplicates().values) > 2)
                else None,
                height=20,
                width=20,
            )

        except Exception:
            print("\n!!!WARNING!!! Cannot plot parent-offspring BD-distance graphs.")
            print(traceback.format_exc(-1))

# Plot emitter reset stats
if args.paper_metrics:
    print("\n    Plotting reset metrics.")
    if "explore_mean_gen_reset" in metrics_frame.columns:
        try:
            frame = pd.DataFrame()
            for env_name in metrics_frame["env_name"].drop_duplicates().values:
                env_metrics_frame = metrics_frame[metrics_frame["env_name"] == env_name]
                for algo in env_metrics_frame["algo"].drop_duplicates().values:
                    sub_frame = env_metrics_frame[(env_metrics_frame["algo"] == algo)]
                    for line in sub_frame["line"].drop_duplicates().values:
                        sub_sub_frame = sub_frame[(sub_frame["line"] == line)][
                            [
                                "gen",
                                "env_name",
                                "algo",
                                "explore_mean_gen_reset",
                                "exploit_mean_gen_reset",
                                "explore_max_gen_reset",
                                "exploit_max_gen_reset",
                            ]
                        ]
                        sub_sub_frame = sub_sub_frame.sort_values(
                            ["gen"], ignore_index=True, key=natsort_keygen()
                        )
                        add_frame = copy.copy(sub_sub_frame)
                        add_frame.dropna(inplace=True)
                        if not (
                            (add_frame["explore_mean_gen_reset"] == 0).all()
                            and (add_frame["exploit_mean_gen_reset"] == 0).all()
                            and (add_frame["explore_max_gen_reset"] == 0).all()
                            and (add_frame["exploit_max_gen_reset"] == 0).all()
                        ):
                            frame = pd.concat([frame, add_frame], ignore_index=True)

            # Plot
            file_name = f"{args.plots}/emitter_reset_generations.svg"
            all_env_plot(
                "gen",
                "Generations",
                frame,
                color_frame,
                file_name,
                columns=["exploit_mean_gen_reset", "explore_mean_gen_reset"],
                names=[
                    "Mean num gens for explore emitters",
                    "Mean num gens for exploit emitters",
                ],
            )

            if args.print_median:
                for env_name in frame["env_name"].drop_duplicates().values:
                    sub_frame = frame[frame["env_name"] == env_name]
                    for algo in sub_frame["algo"].drop_duplicates().values:
                        sub_sub_frame = sub_frame[sub_frame["algo"] == algo]
                        sub_sub_frame = sub_sub_frame.sort_values(
                            ["gen"], ignore_index=True
                        )
                        max_gen = max(sub_sub_frame["gen"])
                        sub_sub_frame = sub_sub_frame[sub_sub_frame["gen"] == max_gen]
                        median_explore_mean = sub_sub_frame[
                            "explore_mean_gen_reset"
                        ].median()
                        median_exploit_mean = sub_sub_frame[
                            "exploit_mean_gen_reset"
                        ].median()
                        median_explore_max = sub_sub_frame[
                            "explore_max_gen_reset"
                        ].median()
                        median_exploit_max = sub_sub_frame[
                            "exploit_max_gen_reset"
                        ].median()
                        print("For", env_name, "and", algo, ":")
                        print("median_explore_mean", median_explore_mean)
                        print("median_exploit_mean", median_exploit_mean)
                        print("median_explore_max", median_explore_max)
                        print("median_exploit_max", median_exploit_max)

        except Exception:
            print("\n!!!WARNING!!! Cannot plot emitter reset graphs.")
            print(traceback.format_exc(-1))

# Plot emitter stagnate stats
if args.paper_metrics:
    print("\n    Plotting stagnate metrics.")
    if "explore_mean_stagnate" in metrics_frame.columns:
        try:
            frame = pd.DataFrame()
            for env_name in metrics_frame["env_name"].drop_duplicates().values:
                env_metrics_frame = metrics_frame[metrics_frame["env_name"] == env_name]
                for algo in env_metrics_frame["algo"].drop_duplicates().values:
                    sub_frame = env_metrics_frame[(env_metrics_frame["algo"] == algo)]
                    for line in sub_frame["line"].drop_duplicates().values:
                        sub_sub_frame = sub_frame[(sub_frame["line"] == line)][
                            [
                                "gen",
                                "env_name",
                                "algo",
                                "explore_mean_stagnate",
                                "exploit_mean_stagnate",
                            ]
                        ]
                        sub_sub_frame = sub_sub_frame.sort_values(
                            ["gen"], ignore_index=True, key=natsort_keygen()
                        )
                        add_frame = copy.copy(sub_sub_frame)
                        add_frame.dropna(inplace=True)
                        if not (
                            (add_frame["explore_mean_stagnate"] == 0).all()
                            and (add_frame["exploit_mean_stagnate"] == 0).all()
                        ):
                            frame = pd.concat([frame, add_frame], ignore_index=True)

            # Plot
            file_name = f"{args.plots}/emitter_stagnate_generations.svg"
            all_env_plot(
                "gen",
                "Generations",
                frame,
                color_frame,
                file_name,
                columns=["explore_mean_stagnate", "exploit_mean_stagnate"],
                names=[
                    "Mean stagnate for explore emitters",
                    "Mean stagnate for exploit emitters",
                ],
            )

        except Exception:
            print("\n!!!WARNING!!! Cannot plot emitter stagnate graphs.")
            print(traceback.format_exc(-1))

#####################
# Plotting p-values #

if args.p_values:

    print("\n    Plotting p-values.")

    for env_name in metrics_frame["env_name"].drop_duplicates().values:

        print("\n    Computing p-values for", env_name)
        env_metrics_frame = metrics_frame[metrics_frame["env_name"] == env_name]
        env_reeval_metrics_frame = reeval_metrics_frame[
            reeval_metrics_frame["env_name"] == env_name
        ]

        # Write p-value
        try:
            file_name = f"{args.plots}/{env_name}_qd_score_p_values.md"
            compute_p_values(env_metrics_frame, file_name, "qd_score")
            file_name = f"{args.plots}/{env_name}_coverage_p_values.md"
            compute_p_values(env_metrics_frame, file_name, "coverage")
            file_name = f"{args.plots}/{env_name}_max_fitness_p_values.md"
            compute_p_values(env_metrics_frame, file_name, "max_fitness")

            file_name = f"{args.plots}/{env_name}_reeval_qd_score_p_values.md"
            compute_p_values(env_reeval_metrics_frame, file_name, "qd_score")
            file_name = f"{args.plots}/{env_name}_reeval_coverage_p_values.md"
            compute_p_values(env_reeval_metrics_frame, file_name, "coverage")
            file_name = f"{args.plots}/{env_name}_reeval_max_fitness_p_values.md"
            compute_p_values(env_reeval_metrics_frame, file_name, "max_fitness")
        except Exception:
            print("\n!!!WARNING!!! Cannot compute p-value for", env_name)
            print(traceback.format_exc(-1))


##################################################
# Finding min and max for plotting archive files #

if args.archives or args.archives_solo:
    print("\n    Getting min-max for archives.")

    env_min_max = pd.DataFrame()

    # Find min and max fitness for each env
    for env_name in config_frame["env_name"].drop_duplicates().values:
        sub_config_frame = config_frame[
            config_frame["env_name"] == env_name
        ].reset_index()
        min_fitness = jnp.inf
        max_fitness = -jnp.inf

        for line in range(sub_config_frame.shape[0]):

            try:
                # Find last repertoire file
                folder = os.path.join(
                    sub_config_frame["folder"][line], "checkpoints/last_grid"
                )

                # Load fitnesses file
                fitnesses = jnp.load(os.path.join(folder, "fitnesses.npy"))
                if jnp.sum(fitnesses == jnp.inf) > 0:
                    print("Some infinite fitnesses:", fitnesses)
                    continue
                fitnesses_inf = jnp.where(fitnesses == -jnp.inf, jnp.inf, fitnesses)
                min_fitness = min(min_fitness, float(min(fitnesses_inf)))
                max_fitness = max(max_fitness, float(max(fitnesses)))
            except Exception:
                print("\n!!!WARNING!!! Cannot open repertoire in", folder)
                print(traceback.format_exc(-1))

            try:
                # Find last reeval repertoire file
                folder = os.path.join(
                    sub_config_frame["folder"][line], "checkpoints/last_reeval_grid"
                )

                # Load fitnesses file
                fitnesses = jnp.load(os.path.join(folder, "fitnesses.npy"))
                if jnp.sum(fitnesses == jnp.inf) > 0:
                    print("Some infinite fitnesses:", fitnesses)
                    continue
                fitnesses_inf = jnp.where(fitnesses == -jnp.inf, jnp.inf, fitnesses)
                min_fitness = min(min_fitness, float(min(fitnesses_inf)))
                max_fitness = max(max_fitness, float(max(fitnesses)))
            except Exception:
                print("\n!!!WARNING!!! Cannot open repertoire in", folder)
                print(traceback.format_exc(-1))

        # Add to the env min and max fitness dataframe
        assert (
            min_fitness > -jnp.inf
        ), f"!!ERROR!!! Incorrect min fit for {env_name}: {min_fitness}."
        assert (
            max_fitness < jnp.inf
        ), f"!!ERROR!!! Incorrect max fit for {env_name}: {max_fitness}."
        new_min_max = pd.DataFrame(
            [[env_name, min_fitness, max_fitness]],
            columns=["env_name", "min_fitness", "max_fitness"],
        )
        env_min_max = pd.concat([env_min_max, new_min_max], ignore_index=True)

####################################
# Printing the archives one by one #

if args.archives_solo:

    print("\n    Plotting archive one by one")

    for line in range(config_frame.shape[0]):
        try:
            # Find last repertoire file
            folder = os.path.join(config_frame["folder"][line], "checkpoints/last_grid")
            algo = config_frame["algo"][line]
            env_name = config_frame["env_name"][line]
            sub_env_min_max = env_min_max[
                env_min_max["env_name"] == env_name
            ].reset_index()
            min_fitness = sub_env_min_max["min_fitness"][0]
            max_fitness = sub_env_min_max["max_fitness"][0]

            # Load files
            fitnesses = jnp.load(os.path.join(folder, "fitnesses.npy"))
            descriptors = jnp.load(os.path.join(folder, "descriptors.npy"))
            centroids = jnp.load(os.path.join(folder, "centroids.npy"))

            # Find last reeval repertoire file
            folder = os.path.join(
                config_frame["folder"][line], "checkpoints/last_reeval_grid"
            )
            reeval_fitnesses = jnp.load(os.path.join(folder, "fitnesses.npy"))
            reeval_descriptors = jnp.load(os.path.join(folder, "descriptors.npy"))
            reeval_centroids = jnp.load(os.path.join(folder, "centroids.npy"))

            # Setting bd values
            if env_name in graph_env_bds.keys():
                min_bd = graph_env_bds[env_name][0]
                max_bd = graph_env_bds[env_name][1]
            else:
                print(
                    "!!!WARNING!!!",
                    env_name,
                    "is not in graph_env_bds so setting arbitrary max and min bd values to [0, 0] and [1, 1].",
                )
                min_bd = [0, 0]
                max_bd = [1, 1]

            # Plot
            file_name = f"{args.plots}/{env_name}_{algo}_{line}_archive.png"
            fig, ax = plot_2d_map_elites_repertoire(
                centroids=centroids,
                repertoire_fitnesses=fitnesses,
                minval=min_bd,
                maxval=max_bd,
                vmin=min_fitness,
                vmax=max_fitness,
                repertoire_descriptors=descriptors,
            )
            plt.tight_layout()
            plt.savefig(file_name, bbox_inches="tight")
            plt.close()

            # Plot reeval
            file_name = f"{args.plots}/{env_name}_{algo}_{line}_reeval_archive.png"
            fig, ax = plot_2d_map_elites_repertoire(
                centroids=reeval_centroids,
                repertoire_fitnesses=reeval_fitnesses,
                minval=min_bd,
                maxval=max_bd,
                vmin=min_fitness,
                vmax=max_fitness,
                repertoire_descriptors=reeval_descriptors,
            )
            plt.tight_layout()
            plt.savefig(file_name, bbox_inches="tight")
            plt.close()

        except Exception:
            print("\n!!!WARNING!!! Cannot plot repertoire in", folder)
            print(traceback.format_exc(-1))


########################################################
# Printing one replication per algorithm in a big grid #

if args.archives:

    print("\n    Plotting archive in a common figure")
    print("        !!!WARNING!!! Taking one replication per algo")

    for env_name in config_frame["env_name"].drop_duplicates().values:
        try:
            sub_config_frame = config_frame[
                config_frame["env_name"] == env_name
            ].reset_index()
            sub_env_min_max = env_min_max[
                env_min_max["env_name"] == env_name
            ].reset_index()
            min_fitness = sub_env_min_max["min_fitness"][0]
            max_fitness = sub_env_min_max["max_fitness"][0]

            # Setting bd values
            if env_name in graph_env_bds.keys():
                min_bd = graph_env_bds[env_name][0]
                max_bd = graph_env_bds[env_name][1]
            else:
                print(
                    "!!!WARNING!!!",
                    env_name,
                    "is not in graph_env_bds so setting arbitrary max and min bd values to [0, 0] and [1, 1].",
                )
                min_bd = [0, 0]
                max_bd = [1, 1]

            # Creating file
            num_algos = len(sub_config_frame["algo"].drop_duplicates().values)
            num_lines = 1 if env_name in env_deterministics else 2
            num_columns = num_algos
            # num_lines = 3 if num_algos > 8 else (2 if num_algos > 4 else 1)
            # num_columns = ceil(num_algos / num_lines)

            file_name = f"{args.plots}/{env_name}_archives.png"
            fig, ax = plt.subplots(
                nrows=num_lines,
                ncols=num_columns,
                figsize=(num_columns * 10, num_lines * 8),
            )
            index = 0

            for algo in sub_config_frame["algo"].drop_duplicates().values:
                sub_sub_config_frame = sub_config_frame[
                    sub_config_frame["algo"] == algo
                ].reset_index()

                try:
                    for folder in (
                        sub_sub_config_frame["folder"].drop_duplicates().values
                    ):
                        folder = os.path.join(folder, "checkpoints/last_grid")
                        genotypes = jnp.load(os.path.join(folder, "genotypes.npy"))
                        max_gen = jax.tree_util.tree_map(
                            lambda x: jnp.amax(x), genotypes
                        )
                        min_gen = jax.tree_util.tree_map(
                            lambda x: jnp.amin(x), genotypes
                        )

                    # Find last repertoire file
                    folder = os.path.join(
                        sub_sub_config_frame["folder"][0], "checkpoints/last_grid"
                    )

                    # Load files
                    fitnesses = jnp.load(os.path.join(folder, "fitnesses.npy"))
                    descriptors = jnp.load(os.path.join(folder, "descriptors.npy"))
                    centroids = jnp.load(os.path.join(folder, "centroids.npy"))

                    # Get titles
                    title = env_name
                    if title in graph_env_names.keys():
                        title = graph_env_names[title]
                    else:
                        print(
                            "!!!WARNING!!!",
                            title,
                            "is not in graph_env_names, keeping this name.",
                        )
                    algo_title = algo
                    if algo in new_names.keys():
                        algo_title = new_names[original_name]

                    # Deterministic environment - print only archive
                    if env_name in env_deterministics:

                        # Plot
                        if num_columns > 1:
                            ax_index = ax[index]
                        else:
                            ax_index = ax

                        if descriptors.shape[-1] <= 2:
                            _, _ = plot_2d_map_elites_repertoire(
                                centroids=centroids,
                                repertoire_fitnesses=fitnesses,
                                minval=min_bd,
                                maxval=max_bd,
                                vmin=min_fitness,
                                vmax=max_fitness,
                                repertoire_descriptors=descriptors,
                                ax=ax_index,
                            )
                            ax_index.set_xlabel(None)
                            ax_index.set_ylabel(None)
                        else:
                            plot_multidimensional_map_elites_grid(
                                fitnesses=fitnesses,
                                descriptors=descriptors,
                                minval=min_bd,
                                maxval=max_bd,
                                grid_shape=(6, 6, 6, 6),
                                vmin=min_fitness,
                                vmax=max_fitness,
                                ax=ax_index,
                            )

                        ax_index.set_title(f"{title} \n{algo_title}", fontsize=40)

                    # Stochastic environment - print both archive and reeval archive
                    else:

                        # Find last reeval repertoire file
                        reeval_folder = os.path.join(
                            sub_sub_config_frame["folder"][0],
                            "checkpoints/last_reeval_grid",
                        )

                        # Load files
                        reeval_fitnesses = jnp.load(
                            os.path.join(reeval_folder, "fitnesses.npy")
                        )
                        reeval_descriptors = jnp.load(
                            os.path.join(reeval_folder, "descriptors.npy")
                        )
                        reeval_centroids = jnp.load(
                            os.path.join(reeval_folder, "centroids.npy")
                        )

                        # Plot
                        if num_columns > 1:
                            ax_index = ax[0][index]
                        else:
                            ax_index = ax[0]
                        if num_columns > 1:
                            ax_reeval_index = ax[1][index]
                        else:
                            ax_reeval_index = ax[1]

                        if descriptors.shape[-1] <= 2:
                            _, _ = plot_2d_map_elites_repertoire(
                                centroids=centroids,
                                repertoire_fitnesses=fitnesses,
                                minval=min_bd,
                                maxval=max_bd,
                                vmin=min_fitness,
                                vmax=max_fitness,
                                repertoire_descriptors=descriptors,
                                ax=ax_index,
                            )
                            ax_index.set_xlabel(None)
                            ax_index.set_ylabel(None)
                            _, _ = plot_2d_map_elites_repertoire(
                                centroids=reeval_centroids,
                                repertoire_fitnesses=reeval_fitnesses,
                                minval=min_bd,
                                maxval=max_bd,
                                vmin=min_fitness,
                                vmax=max_fitness,
                                repertoire_descriptors=reeval_descriptors,
                                ax=ax_reeval_index,
                            )
                            ax_reeval_index.set_xlabel(None)
                            ax_reeval_index.set_ylabel(None)
                        else:
                            plot_multidimensional_map_elites_grid(
                                fitnesses=fitnesses,
                                descriptors=descriptors,
                                minval=min_bd,
                                maxval=max_bd,
                                grid_shape=(6, 6, 6, 6),
                                vmin=min_fitness,
                                vmax=max_fitness,
                                ax=ax_index,
                            )
                            plot_multidimensional_map_elites_grid(
                                fitnesses=reeval_fitnesses,
                                descriptors=reeval_descriptors,
                                minval=min_bd,
                                maxval=max_bd,
                                grid_shape=(6, 6, 6, 6),
                                vmin=min_fitness,
                                vmax=max_fitness,
                                ax=ax_reeval_index,
                            )

                        ax_index.set_title(f"{title} \n{algo_title}", fontsize=40)
                        ax_reeval_index.set_title(
                            f"Corrected {algo_title}", fontsize=40
                        )

                except Exception:
                    print("\n!!!WARNING!!! Cannot plot repertoire in", folder)
                    print(traceback.format_exc(-1))

                index += 1

            plt.tight_layout()
            plt.savefig(file_name, bbox_inches="tight")
            plt.close()
        except Exception:
            print("\n!!!WARNING!!! Cannot plot repertoires for", env_name)
            print(traceback.format_exc(-1))

print("\nFinished all analysis.")

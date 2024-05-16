
# MEMES: MAP-Elites-Multi-ES

This repository contains the code for the paper [Enhancing MAP-Elites with Multiple Parallel Evolution Strategies](https://arxiv.org/abs/2303.06137), introducing the MAP-Elites-Multi-ES (MEMES) algorithm:

This code is based on the [`QDax`](https://github.com/adaptive-intelligent-robotics/QDax) library.

## Installation 🛠️

This repository requires `python>=3.9` because of the `brax` dependency. The list of the required libraries can be found in the `requirements.txt` file, and installed using the following command:
```bash
pip install -r requirements.txt
```

## Running the code 🚀

We provide the code for MEMES, as well as all the baselines, all the ablations, and all the metrics displayed in the paper. We provide one main file and one config file per algorithm. The config files are using [Hydra](https://hydra.cc/docs/intro/) and contain all the hyper-parameters used in the paper. We also provide the code to reproduce the plots of the paper in `analysis.py`.

For example to run the main MEMES algorithm on the Arm task (set as default task):
```bash
python3 train_memes.py
```

Alternatively to run the NS-ES baseline on the Arm task:
```bash
python3 train_nses.py
```

**Important note on metrics:** The paper displays additional metrics to analyse the behaviour of the proposed algorithms (e.g. average lifespan, parent-offspring feature-distance) that cannot easily be integrated into the default QDax framework. Thus, we provide two main files per algorithm: one named `metrics_train_algo.py`, that trains the algorithm `algo` and saves all the metrics shown in the paper, and one named `train_algo.py` that trains the algorithm `algo`  but does not save those metrics. For the baselines, we only provide the version with the metrics, as the other one can be found in the QDax repository.

For example to run the main MEMES algorithm on the Arm task with all the paper metrics:
```bash
python3 metrics_train_memes.py
```

## Understanding the code 💻

If you are not familiar with QDax, we recommend the [QDax documentation](https://qdax.readthedocs.io/en/latest/) to understand the general structure of this repository.

The repository follows the same structure as QDax:
- `core` contains all the files to build the algorithms, in particular:
	- `core` itself contains the main file for the MEMES-All algorithm and a modified version of MAP-Elites for the case with the metrics (see notes on metrics above).
	- `core/emitter` defines MEMES, all its ablation as well as the ES, NS-ES, NSR-ES, NSRA-ES baselines as emitters that can be used as emitter for any QDax algorithm.
	- `core/emitter_metrics` defines MEMES and all its ablation for the case with the metrics (see notes on metrics above).  These versions can only be used with modified versions of the QDax algorithms.
	- `core/container_metrics` contains one file redefining the MAP-Elites Container for the case with the metrics (see notes on metrics above).
- `tasks` contains the definition of the Hexapod task that is not provided by default in the QDax library but is used in the MEMES paper.
- `configs` contains the [Hydra](https://hydra.cc/docs/intro/) config files for all the algorithms.


## Citing MEMES ✏️

If you use MEMES in your research, please cite the following paper:
```
@article{flageat2023multiple,
  title={Enhancing Quality and Diversity using MAP-Elites with Multiple Parallel Evolution Strategies},
  author={Flageat, Manon and Lim, Bryan and Cully, Antoine},
  journal={arXiv e-prints},
  pages={arXiv--2303},
  year={2023}
}
```

`QDax` citation:
```
@article{lim2022accelerated,
  title={Accelerated Quality-Diversity through Massive Parallelism},
  author={Lim, Bryan and Allard, Maxime and Grillotti, Luca and Cully, Antoine},
  journal={Transactions on Machine Learning Research},
  year={2022}
}
@article{chalumeau2024qdax,
  title={Qdax: A library for quality-diversity and population-based algorithms with hardware acceleration},
  author={Chalumeau, Felix and Lim, Bryan and Boige, Raphael and Allard, Maxime and Grillotti, Luca and Flageat, Manon and Mac{\'e}, Valentin and Richard, Guillaume and Flajolet, Arthur and Pierrot, Thomas and others},
  journal={Journal of Machine Learning Research},
  volume={25},
  number={108},
  pages={1--16},
  year={2024}
}
```

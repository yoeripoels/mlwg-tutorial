# MLWG tutorial 6

Tutorial materials on inductive biases, ensembling and confidence calibration, for the 6th session of the [MIT PSFC Machine Learning Working Group](https://github.com/MIT-PSFC/mlwg-store).

> **Disclaimer!** The material was made for illustrative purposes, with its fair share of "vibe coding" in the notebooks: it's checked for correctness, but the implementations are not necessarily the best. For calibration-related functionality, I'd recommend [netcal](https://github.com/EFS-OpenSource/calibration-framework).

## Install

The requirements are set up using [uv](https://docs.astral.sh/uv/). If it is not installed:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```sh
uv sync
uv run jupyter notebook
```

## `synthetic/`

Synthetic data playground, intended in this order:

1. `inductive_bias_boundary.ipynb` - trees vs neural networks on 2D classification with smooth vs sharp decision boundaries
2. `inductive_bias_conv.ipynb` - static MLP vs windowed MLP vs Conv1D on time-series segmentation
3. `calibration_ensembling.ipynb` - reliability diagrams, ECE, temperature scaling, and ensembling

Data lives in `synthetic/data/` (already committed to the repository as `.npz` - can regenerate with `make_*_data.ipynb` if wanted).

## `TCV/`

Per-timestep confinement-state classification (L / D / H) on TCV shots, with XGBoost + 1D CNN ensembling and uncertainty calibration.

1. `download_dataset.ipynb` - run first; downloads a subset from [Zenodo](https://zenodo.org/records/16631053) into `TCV/parquets/`.
2. `tcv_conf_state.ipynb` - the tutorial.

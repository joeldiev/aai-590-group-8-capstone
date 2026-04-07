# Notebooks Guide

This folder contains the notebook-based workflow for the Adaptive Guardrail Layer (AGL) project.

For the full project overview, repository structure, dataset requirements, model architecture, and web application setup, start with the root [README](../README.md).

## Purpose

The notebooks directory is organized around three main activities:

- data preparation
- model training
- experiment tracking

The recommended workflow is:

1. run the notebooks in `data_pipeline/` to fetch, clean, inspect, and engineer features for the dataset
2. run the primary training notebooks in the top-level `notebooks/` folder
3. use the `experimentation/` folder to review prior experiments, prototypes, and intermediate modeling or feature-engineering work

## Folder Layout

### `data_pipeline/`

These notebooks are the primary data preparation pipeline and should generally be run in order:

1. `0_data_fetching.ipynb`
2. `1_data_cleaning.ipynb`
3. `2_eda.ipynb`
4. `3_feature_engineering.ipynb`

These notebooks are responsible for assembling the source datasets, cleaning and standardizing them, exploring distributions, and producing the processed artifacts used later for training.

Important notes:

- some datasets are pulled automatically through the Hugging Face API
- some datasets must be downloaded manually into `data/raw/`
- the root [README](../README.md) documents the Hugging Face token setup and lists the manually downloaded datasets

### Top-Level Training Notebooks

The main training notebooks live directly under `notebooks/`:

- `AGL_denoising_autoencoder.ipynb`
  - trains the denoising autoencoder used for anomaly detection
- `train_roberta_heavy_with_api_exports.ipynb`
  - trains the RoBERTa classifier
  - evaluates the trained classifier
  - exports API-ready classifier artifacts for the FastAPI application

These are the main notebooks to use for final model training workflows.

### `experimentation/`

The `experimentation/` folder is used to keep track of exploratory work, modeling experiments, feature-engineering trials, dataset investigations, local training variants, demos, and other intermediate notebook-based research.

This folder is useful for:

- comparing different modeling approaches
- documenting feature-engineering ideas before they are finalized
- keeping earlier or alternative training workflows
- preserving exploratory analysis that informed the final pipeline

In other words, `experimentation/` is the project’s working lab notebook, while the top-level training notebooks and `data_pipeline/` notebooks reflect the more standardized workflow.

## Suggested Usage

- use the root [README](../README.md) as the source of truth for setup and data requirements
- use `data_pipeline/` to prepare the dataset and features
- use the top-level training notebooks for the main model builds
- use `experimentation/` for reference, comparison, and historical context

## Outputs

Depending on the notebook, outputs may be written to:

- `data/processed/`
- `models/`
- `results/`
- `artifacts/`

If a notebook depends on existing processed data or model artifacts, that dependency should be satisfied by running the earlier pipeline notebooks first.

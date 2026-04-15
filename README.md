# SAXSFormer: Deep Learning for Inverse SAXS Scattering

**SAXSFormer** is a deep learning pipeline designed to solve the inverse scattering problem in structural biology. It automatically extracts 3D macroscopic physical properties—such as Radius of Gyration (Rg), Maximum Diameter (Dmax), and Excluded Volume—directly from 1D Small-Angle X-ray Scattering (SAXS) profiles.

## The Scientific Problem

While X-ray crystallography provides static 3D snapshots of proteins, SAXS measures proteins in their natural, dynamic, liquid state. However, converting the resulting 1D SAXS intensity curve I(q) back into 3D structural parameters is a mathematically ill-posed inverse problem.

Currently, extracting these parameters requires slow, subjective, manual curve-fitting by human experts (e.g., Guinier approximations). This project will try to replace that manual bottleneck with a highly optimized data engineering pipeline and a Deep Learning Transformer model capable of automated, instantaneous property extraction.

## Pipeline Architecture

This repository contains the **data engineering stack** used to build supervised training data for the model, and a **training notebook** that fits the SAXSFormer Transformer on the prepared archive.

1. **Acquisition (`src/data_acquisition.py`):** Multithreaded downloader that queries the [RCSB Search API v2](https://search.rcsb.org/). It applies strict filters: protein-only, monomeric entries, resolution ≤ 2.0 Å, and polymer sequence length 100–400 residues. Structures are saved as `data/raw_pdb/{pdb_id}.ent` (content from the standard PDB download endpoint). Progress uses a concurrent pool with an overall progress bar. The cap on structures is controlled by `MAX_ROWS` in that module (lower it for quick tests).

2. **Simulation (`src/data_preprocessing.py`):** Runs the ATSAS **CRYSOL** engine in parallel over all CPU cores to produce theoretical SAXS curves and reads Rg, Dmax, and volume from the output.

3. **Aggregation (`main.py`):** Collects `.abs` / `.log` outputs, drops atomic coordinates to avoid leakage, and writes the compressed archive `data/processed/saxs_dataset.npz`.

4. **Preparation (`src/data_prepare.py`, optional):** Reads the raw archive **without modifying it** and writes `data/processed/saxs_dataset_prepared.npz`: uniform q-grid (128 points from 0.01–0.50 Å⁻¹), log-transform of intensities, and removal of samples whose labels fall outside 3×IQR per label. Use this file for training when you want a fixed q resolution and cleaned labels.

5. **QA & EDA:** `src/view_dataset.py` (`view-dataset`) validates the raw `.npz`. `src/compare_npz.py` (`data-compare`) prints side-by-side summaries of raw vs prepared datasets. `src/data_visualization.py` (`data-vis`) saves figures under `reports/figures/`.

6. **Training (`src/saxsformer_train.ipynb`):** End-to-end **Phase 3** workflow: load `saxs_dataset_prepared.npz`, verify GPU, train a multi-head Transformer to predict Rg, Dmax, and volume jointly, evaluate with MAE / RMSE / R² per target, and save checkpoints plus training plots. The notebook is written for **Google Colab** (dataset upload via `files.upload()`, GPU runtime). For local use, place the prepared `.npz` at `data/processed/saxs_dataset_prepared.npz`, install **PyTorch** and **scikit-learn**, and run the cells—skip or replace the Colab-specific upload cell as needed.

## Repository Layout

| Path | Role |
|------|------|
| `main.py` | Run CRYSOL over `data/raw_pdb/` and build `saxs_dataset.npz` |
| `src/data_acquisition.py` | PDB search + download (`get-pdb`) |
| `src/data_preprocessing.py` | CRYSOL wrapper and parsing |
| `src/data_prepare.py` | Raw → prepared `.npz` (`data-prepare`, `--input` / `--output`) |
| `src/view_dataset.py` | Inspect raw archive (`view-dataset`) |
| `src/compare_npz.py` | Compare raw vs prepared (`data-compare`) |
| `src/data_visualization.py` | EDA plots (`data-vis`) |
| `src/saxsformer_train.ipynb` | Train / evaluate SAXSFormer on `saxs_dataset_prepared.npz` (Colab-oriented) |

## Installation & Prerequisites

### 1. System Dependencies

You must have the **ATSAS Suite** installed, and the `crysol` command must be on your `PATH`.

- [Download ATSAS](https://www.embl-hamburg.de/biosaxs/download.html)

### 2. Python Environment

Requires **Python ≥ 3.13**. The project uses `uv` for dependency management.

```bash
git clone https://github.com/abhinayshrestha/saxsformer.git
cd saxsformer

uv pip install -e .
```

Console scripts (from `pyproject.toml`): `get-pdb`, `view-dataset`, `data-vis`, `data-prepare`, `data-compare`. You can still run modules directly (e.g. `python src/data_prepare.py`) if you prefer.

## Data Directory Structure

Create a top-level `data/` directory if it is missing. The pipeline uses:

- **`data/raw_pdb/`** — Downloaded coordinate files (`*.ent`).
- **`data/simulated_saxs/`** — CRYSOL `.abs` and `.log` outputs (kept for inspection and idempotent re-runs).
- **`data/processed/`** — `saxs_dataset.npz` (raw bundle) and, after preparation, `saxs_dataset_prepared.npz`.
- **`reports/figures/`** — PNG figures from `data-vis` or from `saxsformer_train.ipynb` (directory is created automatically; `reports/` is gitignored except what you choose to track).
- **`checkpoints/`** — Created by `saxsformer_train.ipynb` for model weights and the fitted label scaler.

## Usage Guide

The pipeline is idempotent: existing downloads and successful simulations are skipped on re-run.

### Step 1: Fetch structures

```bash
get-pdb
```

Adjust `MAX_ROWS` in `src/data_acquisition.py` for smaller batches.

### Step 2: Simulate SAXS and build the raw dataset

```bash
python main.py
```

### Step 3: Validate the raw archive

```bash
view-dataset
```

### Step 4 (optional): Build the training-ready archive

```bash
data-prepare
# or custom paths:
data-prepare --input data/processed/saxs_dataset.npz --output data/processed/saxs_dataset_prepared.npz
```

### Step 5 (optional): Compare raw vs prepared

```bash
data-compare
```

### Step 6 (optional): EDA figures

```bash
data-vis
```

Figures are written to `reports/figures/`. Edit the `CONFIG` section at the top of `src/data_visualization.py` to change the input path, output directory, or whether plots are shown interactively.

### Step 7 (optional): Train the model

Open `src/saxsformer_train.ipynb` in Jupyter or Colab after you have run `data-prepare`. The notebook only needs `saxs_dataset_prepared.npz`; outputs go under `checkpoints/` and `reports/figures/` by default. Training dependencies (**torch**, **scikit-learn**) are used in the notebook but are not listed in `pyproject.toml`—install them in the environment where you run the notebook.

## Dataset Structure

Both archives use the same **keys**: `x`, `q`, `y_rg`, `y_dmax`, `y_volume`, `ids`.

| Artifact | `x` (curves) | `q` | Notes |
|----------|----------------|-----|--------|
| **`saxs_dataset.npz`** | `(N, M)` linear intensity | `(M,)` or `(N, M)` — CRYSOL sampling | Written by `main.py`; `M` depends on CRYSOL settings (often on the order of hundreds of points). |
| **`saxs_dataset_prepared.npz`** | `(N, 128)` log intensity | `(128,)` uniform 0.01–0.50 Å⁻¹ | Outliers removed; suitable as a fixed-size tensor input for training. |

Example shapes for a large raw run (before preparation): `x` might look like `(10000, 512)` with `q` shaped `(512,)`, depending on your simulation grid.

## Author

**Abhinay Shrestha**

- GitHub: [@abhinayshrestha](https://github.com/abhinayshrestha)

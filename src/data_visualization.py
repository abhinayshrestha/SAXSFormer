#!/usr/bin/env python3
"""
data_visualization.py

Utility script for visualizing and validating the serialized SAXS dataset.

This module loads the compressed NumPy dataset archive produced by the
SAXSFormer preprocessing pipeline and generates a set of standard
exploratory data analysis (EDA) plots for Phase 1 reporting.

Expected dataset keys
---------------------
- x         : SAXS intensity curves, shape (N, M)
- q         : q-values, shape (M,) or (N, M)
- y_rg      : radius of gyration labels, shape (N,)
- y_dmax    : maximum dimension labels, shape (N,)
- y_volume  : excluded volume labels, shape (N,)
- ids       : protein identifiers, shape (N,)

Generated figures
-----------------
1. Example SAXS profiles
2. Distribution of Rg
3. Distribution of Dmax
4. Distribution of excluded volume
5. Scatter plot of Rg vs Dmax
6. Mean SAXS curve
7. Mean SAXS curve ± standard deviation

Usage
-----
python data_visualization.py

Optional arguments can be modified in the CONFIG section below.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATASET_PATH = "data/processed/saxs_dataset.npz"
OUTPUT_DIR = "reports/figures"

NUM_EXAMPLE_CURVES = 10
HIST_BINS = 20
SAVE_DPI = 300
SHOW_PLOTS = True


def load_dataset(file_path: str) -> dict:
    """Load the SAXS dataset from a compressed .npz archive."""
    with np.load(file_path, allow_pickle=True) as data:
        dataset = {
            "x": data["x"],
            "q": data["q"],
            "y_rg": data["y_rg"],
            "y_dmax": data["y_dmax"],
            "y_volume": data["y_volume"],
            "ids": data["ids"],
        }
    return dataset


def get_q_vector(q: np.ndarray, sample_index: int = 0) -> np.ndarray:
    """
    Return a 1D q-vector.

    Handles both:
    - global q shape: (M,)
    - per-sample q shape: (N, M)
    """
    if q.ndim == 1:
        return q
    if q.ndim == 2:
        return q[sample_index]
    raise ValueError(f"Unsupported q shape: {q.shape}")


def validate_dataset(dataset: dict) -> None:
    """Print basic dataset diagnostics."""
    x = dataset["x"]
    q = dataset["q"]
    y_rg = dataset["y_rg"]
    y_dmax = dataset["y_dmax"]
    y_volume = dataset["y_volume"]
    ids = dataset["ids"]

    print("\n--- DATASET STRUCTURE ---")
    print("Curves (x):", x.shape)
    print("q values:", q.shape)
    print("Rg:", y_rg.shape)
    print("Dmax:", y_dmax.shape)
    print("Volume:", y_volume.shape)
    print("IDs:", ids.shape)

    print("\n--- DATASET HEALTH CHECK ---")
    print("NaN in curves:", np.isnan(x).sum())
    print("NaN in q:", np.isnan(q).sum())
    print("NaN in Rg:", np.isnan(y_rg).sum())
    print("NaN in Dmax:", np.isnan(y_dmax).sum())
    print("NaN in Volume:", np.isnan(y_volume).sum())
    print("Inf in curves:", np.isinf(x).sum())

    curve_length = x.shape[1]
    q_length = len(q) if q.ndim == 1 else q.shape[1]

    print("\nCurve length:", curve_length)
    print("q length:", q_length)

    if curve_length != q_length:
        print("WARNING: q length does not match curve length!")

    print("\n--- LABEL STATISTICS ---")
    print(
        f"Rg     -> min: {np.min(y_rg):.4f}, max: {np.max(y_rg):.4f}, "
        f"mean: {np.mean(y_rg):.4f}, std: {np.std(y_rg):.4f}"
    )
    print(
        f"Dmax   -> min: {np.min(y_dmax):.4f}, max: {np.max(y_dmax):.4f}, "
        f"mean: {np.mean(y_dmax):.4f}, std: {np.std(y_dmax):.4f}"
    )
    print(
        f"Volume -> min: {np.min(y_volume):.4f}, max: {np.max(y_volume):.4f}, "
        f"mean: {np.mean(y_volume):.4f}, std: {np.std(y_volume):.4f}"
    )


def save_and_optionally_show(fig: plt.Figure, output_path: Path, show: bool) -> None:
    """Save a matplotlib figure and optionally display it."""
    fig.tight_layout()
    fig.savefig(output_path, dpi=SAVE_DPI, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_example_saxs_profiles(
    x: np.ndarray,
    q: np.ndarray,
    output_dir: Path,
    num_curves: int = NUM_EXAMPLE_CURVES,
    show: bool = SHOW_PLOTS,
) -> None:
    """Plot several example SAXS curves."""
    fig = plt.figure(figsize=(8, 5))
    num_to_plot = min(num_curves, len(x))

    for i in range(num_to_plot):
        q_vals = get_q_vector(q, i)
        plt.plot(q_vals, x[i], alpha=0.8, label=f"Sample {i + 1}")

    plt.yscale("log")
    plt.xlabel(r"q ($\AA^{-1}$)")
    plt.ylabel("Intensity I(q)")
    plt.title("Example SAXS Profiles")

    if num_to_plot <= 10:
        plt.legend(fontsize=8)

    save_and_optionally_show(
        fig,
        output_dir / "figure1_example_saxs_profiles.png",
        show,
    )


def plot_rg_distribution(
    y_rg: np.ndarray,
    output_dir: Path,
    show: bool = SHOW_PLOTS,
) -> None:
    """Plot histogram of Rg values."""
    fig = plt.figure(figsize=(7, 5))
    plt.hist(y_rg, bins=HIST_BINS)
    plt.xlabel(r"Radius of Gyration $R_g$ ($\AA$)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Radius of Gyration")
    save_and_optionally_show(fig, output_dir / "figure2_rg_distribution.png", show)


def plot_dmax_distribution(
    y_dmax: np.ndarray,
    output_dir: Path,
    show: bool = SHOW_PLOTS,
) -> None:
    """Plot histogram of Dmax values."""
    fig = plt.figure(figsize=(7, 5))
    plt.hist(y_dmax, bins=HIST_BINS)
    plt.xlabel(r"Maximum Dimension $D_{max}$ ($\AA$)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Maximum Particle Dimension")
    save_and_optionally_show(fig, output_dir / "figure3_dmax_distribution.png", show)


def plot_volume_distribution(
    y_volume: np.ndarray,
    output_dir: Path,
    show: bool = SHOW_PLOTS,
) -> None:
    """Plot histogram of excluded volume values."""
    fig = plt.figure(figsize=(7, 5))
    plt.hist(y_volume, bins=HIST_BINS)
    plt.xlabel(r"Excluded Volume ($\AA^3$)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Excluded Volume")
    save_and_optionally_show(fig, output_dir / "figure4_volume_distribution.png", show)


def plot_rg_vs_dmax(
    y_rg: np.ndarray,
    y_dmax: np.ndarray,
    output_dir: Path,
    show: bool = SHOW_PLOTS,
) -> None:
    """Plot scatter of Rg versus Dmax."""
    fig = plt.figure(figsize=(7, 5))
    plt.scatter(y_rg, y_dmax, s=25, alpha=0.7)
    plt.xlabel(r"Radius of Gyration $R_g$ ($\AA$)")
    plt.ylabel(r"Maximum Dimension $D_{max}$ ($\AA$)")
    plt.title(r"Relationship Between $R_g$ and $D_{max}$")
    save_and_optionally_show(fig, output_dir / "figure5_rg_vs_dmax.png", show)


def plot_mean_saxs_curve(
    x: np.ndarray,
    q: np.ndarray,
    output_dir: Path,
    show: bool = SHOW_PLOTS,
) -> None:
    """Plot the mean SAXS curve across all samples."""
    q_vals = get_q_vector(q, 0)
    mean_curve = np.mean(x, axis=0)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(q_vals, mean_curve)
    plt.yscale("log")
    plt.xlabel(r"q ($\AA^{-1}$)")
    plt.ylabel("Mean Intensity")
    plt.title("Average SAXS Curve")
    save_and_optionally_show(fig, output_dir / "figure6_mean_saxs_curve.png", show)


def plot_mean_plus_std_saxs_curve(
    x: np.ndarray,
    q: np.ndarray,
    output_dir: Path,
    show: bool = SHOW_PLOTS,
) -> None:
    """Plot the mean SAXS curve with a standard deviation envelope."""
    q_vals = get_q_vector(q, 0)
    mean_curve = np.mean(x, axis=0)
    std_curve = np.std(x, axis=0)

    lower = np.clip(mean_curve - std_curve, 1e-12, None)
    upper = np.clip(mean_curve + std_curve, 1e-12, None)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(q_vals, mean_curve, label="Mean curve")
    plt.fill_between(q_vals, lower, upper, alpha=0.3, label="± 1 std")
    plt.yscale("log")
    plt.xlabel(r"q ($\AA^{-1}$)")
    plt.ylabel("Intensity I(q)")
    plt.title("Average SAXS Curve with Variability")
    plt.legend()
    save_and_optionally_show(
        fig,
        output_dir / "figure7_mean_std_saxs_curve.png",
        show,
    )


def main() -> None:
    """Load dataset, validate it, and generate all visualization figures."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_PATH)
    validate_dataset(dataset)

    x = dataset["x"]
    q = dataset["q"]
    y_rg = dataset["y_rg"]
    y_dmax = dataset["y_dmax"]
    y_volume = dataset["y_volume"]

    plot_example_saxs_profiles(x, q, output_dir)
    plot_rg_distribution(y_rg, output_dir)
    plot_dmax_distribution(y_dmax, output_dir)
    plot_volume_distribution(y_volume, output_dir)
    plot_rg_vs_dmax(y_rg, y_dmax, output_dir)
    plot_mean_saxs_curve(x, q, output_dir)
    plot_mean_plus_std_saxs_curve(x, q, output_dir)

    print(f"\nSaved all figures to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
data_prepare.py

SAXSFormer Phase 3 - Preprocessing Script

Loads the raw SAXS dataset produced by the simulation pipeline and applies
three preprocessing steps to produce a clean, training-ready dataset.

Steps
-----
1. q-grid standardization  — interpolate all curves onto a fixed uniform
                              q-grid of 128 points from 0.01 to 0.50 Å⁻¹
2. Log-transform           — x = log(x + 1e-12) to compress the 4-order
                              magnitude range of SAXS intensities
3. Outlier removal         — drop samples where any label (Rg, Dmax, Volume)
                              falls beyond 3×IQR of that label's distribution

Input
-----
    data/processed/saxs_dataset.npz          ← raw file, never touched

Output
------
    data/processed/saxs_dataset_prepared.npz ← training-ready file

Dataset keys (unchanged from raw)
----------------------------------
    x          → SAXS intensity curves      (N, 128)  log-transformed
    q          → fixed uniform q-grid       (128,)
    y_rg       → radius of gyration         (N,)
    y_dmax     → maximum dimension          (N,)
    y_volume   → excluded volume            (N,)
    ids        → PDB identifiers            (N,)

Usage
-----
    data-prepare

    # Override input / output paths
    data-prepare --input path/to/saxs_dataset.npz \\
                 --output path/to/saxs_dataset_prepared.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_PATH = "data/processed/saxs_dataset.npz"
OUTPUT_PATH = "data/processed/saxs_dataset_prepared.npz"

Q_MIN = 0.01  # Å⁻¹  — matches real-data beamstop cutoff
Q_MAX = 0.50  # Å⁻¹
Q_POINTS = 128  # number of uniform q-sampling points
LOG_EPS = 1e-12  # additive offset before log to avoid log(0)
IQR_FACTOR = 3.0  # outlier threshold multiplier


def _npz_float32_array(entry: object) -> np.ndarray:
    """Float32 ndarray from an npz value (avoids NpzFile / Pylint false positives)."""
    return np.asarray(entry, dtype=np.float32)


# ─────────────────────────────────────────────
# STEP 1 — q-grid standardization
# ─────────────────────────────────────────────


def standardize_q_grid(
    curves: np.ndarray,
    q: np.ndarray,
    q_uniform: np.ndarray,
) -> np.ndarray:
    """
    Interpolate every curve onto a fixed uniform q-grid.

    Parameters
    ----------
    curves    : raw SAXS intensity matrix, shape (N, M)
    q         : original q-values, shape (M,) or (N, M)
    q_uniform : target uniform q-grid, shape (Q_POINTS,)

    Returns
    -------
    curves_interp : shape (N, Q_POINTS)
    """
    N = curves.shape[0]
    curves_interp = np.zeros((N, len(q_uniform)), dtype=np.float32)

    for i in range(N):
        # Support both a single shared q-vector and per-sample q-vectors
        q_i = q[i] if q.ndim == 2 else q

        # Clip interpolation range to the overlap between q_i and q_uniform
        q_lo = max(q_i.min(), q_uniform.min())
        q_hi = min(q_i.max(), q_uniform.max())

        mask_src = (q_i >= q_lo) & (q_i <= q_hi)
        mask_tgt = (q_uniform >= q_lo) & (q_uniform <= q_hi)

        f = interp1d(
            q_i[mask_src],
            curves[i][mask_src],
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        curves_interp[i][mask_tgt] = f(q_uniform[mask_tgt])

    return curves_interp


# ─────────────────────────────────────────────
# STEP 2 — log-transform
# ─────────────────────────────────────────────


def log_transform(curves: np.ndarray, eps: float = LOG_EPS) -> np.ndarray:
    """
    Apply log(x + eps) to compress the 4-order magnitude range of
    SAXS intensities. Curves must already be interpolated (positive values).
    """
    return np.log(curves + eps).astype(np.float32)


# ─────────────────────────────────────────────
# STEP 3 — outlier removal
# ─────────────────────────────────────────────


def remove_outliers(
    curves: np.ndarray,
    y_rg: np.ndarray,
    y_dmax: np.ndarray,
    y_volume: np.ndarray,
    ids: np.ndarray,
    factor: float = IQR_FACTOR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove samples where any label falls beyond factor × IQR.

    A sample is dropped if Rg, Dmax, OR Volume is an outlier.
    All arrays are filtered with the same boolean mask so they
    stay aligned.

    Returns
    -------
    Filtered (curves, y_rg, y_dmax, y_volume, ids)
    """
    N = len(y_rg)
    keep = np.ones(N, dtype=bool)

    for label, name in zip(
        [y_rg, y_dmax, y_volume],
        ["Rg", "Dmax", "Volume"],
    ):
        Q1 = np.percentile(label, 25)
        Q3 = np.percentile(label, 75)
        IQR = Q3 - Q1
        lo = Q1 - factor * IQR
        hi = Q3 + factor * IQR
        mask = (label < lo) | (label > hi)
        n_out = mask.sum()

        print(
            f"  [{name}]  Q1={Q1:.2f}  Q3={Q3:.2f}  IQR={IQR:.2f}  "
            f"range=[{lo:.2f}, {hi:.2f}]  outliers={n_out}"
        )

        keep &= ~mask

    n_dropped = N - keep.sum()
    print(f"\n  Dropped {n_dropped} outlier samples  ({N} → {keep.sum()} remaining)")

    return (
        curves[keep],
        y_rg[keep],
        y_dmax[keep],
        y_volume[keep],
        ids[keep],
    )


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────


def main(args) -> None:
    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    # ── Load raw dataset ──────────────────────────────────────────────────────
    print(f"\n[load] Reading: {input_path}")
    with np.load(input_path, allow_pickle=True) as data:
        curves = _npz_float32_array(data["x"])  # (N, M)
        q = _npz_float32_array(data["q"])  # (M,) or (N, M)
        y_rg = _npz_float32_array(data["y_rg"])  # (N,)
        y_dmax = _npz_float32_array(data["y_dmax"])  # (N,)
        y_volume = _npz_float32_array(data["y_volume"])  # (N,)
        ids = data["ids"]  # (N,)

    N, M = curves.shape
    print(f"[load] {N} samples  |  curve length = {M}")
    print(f"[load] q shape = {q.shape}")

    # ── Step 1: q-grid standardization ───────────────────────────────────────
    print(
        f"\n[step 1] Interpolating curves onto uniform q-grid "
        f"({Q_POINTS} pts, {Q_MIN}–{Q_MAX} Å⁻¹) ..."
    )

    q_uniform = np.linspace(Q_MIN, Q_MAX, Q_POINTS, dtype=np.float32)
    curves = standardize_q_grid(curves, q, q_uniform)

    print(f"[step 1] Done  →  curves shape: {curves.shape}")

    # ── Step 2: log-transform ─────────────────────────────────────────────────
    print(f"\n[step 2] Applying log-transform  (eps={LOG_EPS}) ...")
    curves = log_transform(curves)
    print(
        f"[step 2] Done  →  min={curves.min():.4f}  max={curves.max():.4f}  "
        f"mean={curves.mean():.4f}"
    )

    # ── Step 3: outlier removal ───────────────────────────────────────────────
    print(f"\n[step 3] Removing outliers  (threshold = {IQR_FACTOR}×IQR) ...")
    curves, y_rg, y_dmax, y_volume, ids = remove_outliers(
        curves, y_rg, y_dmax, y_volume, ids
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[save] Writing: {output_path}")
    np.savez_compressed(
        output_path,
        x=curves,  # log-transformed, interpolated curves  (N, 128)
        q=q_uniform,  # fixed uniform q-grid                   (128,)
        y_rg=y_rg,  # radius of gyration                     (N,)
        y_dmax=y_dmax,  # maximum dimension                      (N,)
        y_volume=y_volume,  # excluded volume                        (N,)
        ids=ids,  # PDB identifiers                        (N,)
    )

    print("\n[done] ─────────────────────────────────────────────────────")
    print(f"  Input  : {input_path}   (untouched)")
    print(f"  Output : {output_path}")
    print(f"  Samples: {len(y_rg)}")
    print(f"  Curves : {curves.shape}")
    print(
        f"  q-grid : {q_uniform.shape}  [{q_uniform[0]:.3f} … {q_uniform[-1]:.3f}] Å⁻¹"
    )
    print("────────────────────────────────────────────────────────────")


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="SAXSFormer data preparation")
    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_PATH,
        help=f"Path to raw .npz dataset (default: {INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_PATH,
        help=f"Path for prepared .npz output (default: {OUTPUT_PATH})",
    )
    main(parser.parse_args())


if __name__ == "__main__":
    cli_main()

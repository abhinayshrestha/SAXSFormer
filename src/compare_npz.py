#!/usr/bin/env python3
"""
compare_npz.py

Side-by-side comparison of the raw and prepared SAXSFormer datasets.
Prints the first 10 rows of each .npz file in a human-readable table.

Usage
-----
    python compare_npz.py
"""

import numpy as np

RAW_PATH = "data/processed/saxs_dataset.npz"
PREPARED_PATH = "data/processed/saxs_dataset_prepared.npz"
N_ROWS = 10


def _npz_float_array(entry: object) -> np.ndarray:
    """Float64 ndarray from an npz archive value (avoids NpzFile typing false positives)."""
    return np.asarray(entry, dtype=np.float64)


def load(path):
    with np.load(path, allow_pickle=True) as d:
        return {
            "ids": d["ids"],
            "y_rg": _npz_float_array(d["y_rg"]),
            "y_dmax": _npz_float_array(d["y_dmax"]),
            "y_volume": _npz_float_array(d["y_volume"]),
            "x": _npz_float_array(d["x"]),
            "q": _npz_float_array(d["q"]),
        }


def summarize_curve(curve):
    """Return min, max, mean of a single curve as a compact string."""
    return f"min={curve.min():.4f}  max={curve.max():.4f}  mean={curve.mean():.4f}"


def print_table(label, data, n=N_ROWS):
    ids = data["ids"]
    y_rg = data["y_rg"]
    y_dmax = data["y_dmax"]
    y_volume = data["y_volume"]
    x = data["x"]
    q = data["q"]

    # q info
    q_1d = q if q.ndim == 1 else q[0]
    q_info = f"{q_1d.min():.4f} … {q_1d.max():.4f} Å⁻¹  ({len(q_1d)} pts)"

    print(f"\n{'=' * 100}")
    print(f"  {label}")
    print(f"  Total samples : {len(ids)}")
    print(f"  Curve shape   : {x.shape}")
    print(f"  q-grid        : {q_info}")
    print(f"{'=' * 100}")

    # Header
    col_w = [8, 8, 8, 12, 44]
    header = (
        f"  {'#':<{col_w[0]}}"
        f"{'PDB ID':<{col_w[1]}}"
        f"{'Rg (Å)':<{col_w[2]}}"
        f"{'Dmax (Å)':<{col_w[3]}}"
        f"{'Volume (Å³)':<{col_w[3]}}"
        f"{'Curve (min / max / mean)':<{col_w[4]}}"
    )
    print(header)
    print(f"  {'-' * 96}")

    for i in range(min(n, len(ids))):
        curve_summary = summarize_curve(x[i])
        print(
            f"  {i + 1:<{col_w[0]}}"
            f"{str(ids[i]):<{col_w[1]}}"
            f"{y_rg[i]:<{col_w[2]}.3f}"
            f"{y_dmax[i]:<{col_w[3]}.3f}"
            f"{y_volume[i]:<{col_w[3]}.1f}"
            f"{curve_summary}"
        )

    print(f"  {'-' * 96}")


def print_diff_summary(raw, prepared):
    print(f"\n{'=' * 100}")
    print("  DIFF SUMMARY")
    print(f"{'=' * 100}")

    print(
        f"  Samples      : {len(raw['ids'])}  →  {len(prepared['ids'])}  "
        f"({len(raw['ids']) - len(prepared['ids'])} dropped as outliers)"
    )

    print(
        f"  Curve length : {raw['x'].shape[1]}  →  {prepared['x'].shape[1]}  "
        f"(interpolated to 128 uniform pts)"
    )

    raw_q = raw["q"] if raw["q"].ndim == 1 else raw["q"][0]
    prep_q = prepared["q"]
    print(
        f"  q-range      : [{raw_q.min():.4f}, {raw_q.max():.4f}]  →  "
        f"[{prep_q.min():.4f}, {prep_q.max():.4f}]"
    )

    print("  Curve values : raw (linear intensity)  →  prepared (log-transformed)")
    print(
        f"    raw    min={raw['x'].min():.6f}  max={raw['x'].max():.6f}  "
        f"mean={raw['x'].mean():.6f}"
    )
    print(
        f"    prep   min={prepared['x'].min():.4f}  max={prepared['x'].max():.4f}  "
        f"mean={prepared['x'].mean():.4f}"
    )
    print(f"{'=' * 100}\n")


def main():
    print("\nLoading datasets ...")
    raw = load(RAW_PATH)
    prepared = load(PREPARED_PATH)

    print_table(
        "RAW      →  saxs_dataset.npz         (linear intensity, original q-grid)", raw
    )
    print_table(
        "PREPARED →  saxs_dataset_prepared.npz (log-transformed, uniform q-grid)",
        prepared,
    )
    print_diff_summary(raw, prepared)


if __name__ == "__main__":
    main()

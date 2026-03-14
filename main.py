#!/usr/bin/env python3
"""
main.py

Entry point for the SAXS dataset generation pipeline.

This script orchestrates the full preprocessing workflow that converts raw
protein structures into a machine-learning-ready dataset of simulated SAXS
profiles and associated structural descriptors.

Pipeline Overview
-----------------
1. Input Structures
   Protein structures downloaded from the Protein Data Bank are expected in:

       data/raw_pdb/

   Each file should be a PDB/ENT structure containing atomic coordinates.

2. SAXS Simulation
   The script calls the preprocessing pipeline implemented in:

       src/data_preprocessing.py

   That module performs the following steps for each protein:

       • Runs the CRYSOL physics engine (ATSAS suite)
       • Computes a theoretical SAXS scattering curve
       • Extracts structural parameters from CRYSOL output

   Specifically, the following information is extracted:

       q            → momentum transfer values
       I(q)         → scattering intensity curve
       Rg           → radius of gyration
       Dmax         → maximum particle dimension
       Volume       → excluded volume

3. Parallel Processing
   CRYSOL simulations are executed in parallel across all available CPU cores
   using Python’s ProcessPoolExecutor. This enables efficient processing of
   large protein datasets (10k–100k structures).

4. Dataset Assembly
   Once all valid simulations finish, the returned records are aggregated into
   NumPy arrays representing the full dataset:

       x         → SAXS intensity curves
       q         → q-vector (momentum transfer)
       y_rg      → radius of gyration labels
       y_dmax    → maximum dimension labels
       y_volume  → excluded volume labels
       ids       → protein PDB identifiers

5. Serialization
   The dataset is stored as a compressed NumPy archive:

       data/processed/saxs_dataset.npz

   This single-file representation is optimized for machine learning pipelines
   because it eliminates the need to repeatedly parse thousands of text files.

Scientific Context
------------------
The resulting dataset provides supervised training pairs mapping SAXS
scattering profiles to global structural descriptors of proteins.

The dataset can be used to train machine learning models—such as Transformer
architectures—to directly infer structural properties from SAXS curves,
potentially accelerating automated analysis of experimental SAXS data.

Dependencies
------------
- NumPy
- ATSAS / CRYSOL
- Python multiprocessing utilities (via src.data_preprocessing)

Typical Dataset Size
--------------------
Designed for datasets of approximately:

    10,000 – 100,000 proteins

Each protein contributes one SAXS intensity curve and its corresponding
structural parameters.
"""

import os

import numpy as np

from src.data_preprocessing import run_simulation_pipeline


def main():
    # 1. Configuration
    INPUT_PDB_DIR = "data/raw_pdb"
    SIM_OUTPUT_DIR = "data/simulated_saxs"  # Permanent storage for .abs/.log
    OUTPUT_DATASET = "data/processed/saxs_dataset.npz"  # Final serialized file

    os.makedirs(SIM_OUTPUT_DIR, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # 2. Run Simulation Pipeline
    dataset = run_simulation_pipeline(INPUT_PDB_DIR, SIM_OUTPUT_DIR)

    if not dataset:
        print("Pipeline failed.")
        return

    # 3. Aggregate into Numpy arrays
    curves = np.array([item["curve"] for item in dataset])
    qs = np.array([item["q"] for item in dataset])
    rgs = np.array([item["rg"] for item in dataset])
    dmaxs = np.array([item["dmax"] for item in dataset])
    volumes = np.array([item["volume"] for item in dataset])
    ids = np.array([item["id"] for item in dataset])

    # 4. Save archive
    np.savez_compressed(
        OUTPUT_DATASET,
        x=curves,
        q=qs,
        y_rg=rgs,
        y_dmax=dmaxs,
        y_volume=volumes,
        ids=ids,
    )

    print("\n--- SUCCESS ---")
    print(f"Individual SAXS data saved to: {SIM_OUTPUT_DIR}")
    print(f"Serialized dataset saved to: {OUTPUT_DATASET}")


if __name__ == "__main__":
    main()

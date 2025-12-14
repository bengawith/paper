"""
create_ablation_datasets.py

Create five datasets for ablation study:
1. NoMask: CST parameters + uniform mask (all 1's)
2. BinaryMask: CST parameters + binary validity indicators (0 or 1)
3. ExponentialMask (τ=1.0): Continuous distance indicators (exponential distance mask)
4. ExponentialMask (τ=0.2): Continuous distance with τ=0.2
5. ExponentialMask (τ=0.5): Continuous distance with τ=0.5

All use 12 CST parameters (5 weights per side + TE + LE) for consistency.
This allows testing:
- Necessity of distance weighting (NoMask vs Exponential)
- Hard vs soft masking (Binary vs Exponential)
- Sensitivity to τ parameter (different exponential decay rates)
"""

import sys
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("%s", "=" * 70)
logger.info("CREATING ABLATION STUDY DATASETS")
logger.info("%s", "=" * 70)

# Load base data
logger.info("Loading base aerodynamic data with CST parameters...")
csv_path = "data/csv/dataset_12CST_params.csv"
df_base = pd.read_csv(csv_path)
logger.info("Loaded %d samples", len(df_base))

# Extract original alpha/CL data
cl_cols_existing = [f"CL_{i}" for i in range(48)]

standardised_grid = np.linspace(-20, 20, 48)

# CST parameter columns
# 5 lower + 5 upper + TE + LE = 12 params
cst_cols = []
for i in range(5):  # Changed from 6 to 5
    cst_cols.append(f"lower_weight_{i}")
for i in range(5):  # Changed from 6 to 5
    cst_cols.append(f"upper_weight_{i}")
cst_cols.extend(["TE_thickness", "leading_edge_weight"])

# Verify CST columns exist
missing_cols = [col for col in cst_cols if col not in df_base.columns]
if missing_cols:
    logger.error("Missing CST columns: %s", missing_cols)
    logger.info(
        "Available columns: %s",
        [
            col
            for col in df_base.columns
            if "weight" in col.lower() or "TE" in col or "leading" in col
        ],
    )
    sys.exit(1)

X_cols = ["aerofoil_name"] + cst_cols
logger.info("Using %d CST parameters", len(cst_cols))

# Get existing dist columns to identify valid ranges
dist_cols_existing = [f"dist_{i}" for i in range(48)]
if not all(col in df_base.columns for col in dist_cols_existing):
    logger.info("Original dataset missing distance columns, will use heuristic")
    # Fallback: assume edges are extrapolated if CL is constant
    has_dist = False
else:
    has_dist = True
    logger.info("Using existing distance information to identify valid ranges")


# ==================== Dataset 1: NoMask ====================
logger.info("%s", "\n" + "=" * 70)
logger.info("DATASET 1: NoMask (uniform weighting, all mask = 1)")
logger.info("%s", "=" * 70)

clean_data_nomask = []

for index, row in df_base.iterrows():
    # Get CL values from existing dataset
    cl_values = row[cl_cols_existing].values.astype(float)

    # Create uniform mask (all 1's - treats all points equally)
    uniform_mask = np.ones(48, dtype=float)

    # Create row with CST params + CL values + uniform mask
    row_dict = {col: row[col] for col in X_cols}
    for i, cl_val in enumerate(cl_values):
        row_dict[f"CL_{i}"] = cl_val
    for i, mask_val in enumerate(uniform_mask):
        row_dict[f"mask_{i}"] = mask_val  # All 1.0

    clean_data_nomask.append(row_dict)

df_nomask = pd.DataFrame(clean_data_nomask)
output_path_nomask = "data/csv/ablation_nomask_12CST_params.csv"
df_nomask.to_csv(output_path_nomask, index=False)
logger.info("Created: %s", output_path_nomask)
logger.info("Samples: %d", len(df_nomask))
logger.info("Columns: %d (CST + CL + mask_0..mask_47, all = 1.0)", len(df_nomask.columns))


# ==================== Dataset 2: BinaryMask ====================
logger.info("%s", "\n" + "=" * 70)
logger.info("DATASET 2: BinaryMask (binary validity indicators)")
logger.info("%s", "=" * 70)

clean_data_binary = []

for index, row in df_base.iterrows():
    # Get CL and distance values
    cl_values = row[cl_cols_existing].values.astype(float)

    if has_dist:
        # Use existing distance info: dist > 0.99 means valid
        dist_values = row[dist_cols_existing].values.astype(float)
        binary_mask = (dist_values > 0.99).astype(int)
    else:
        # Fallback: assume middle 70% is valid
        binary_mask = np.zeros(48, dtype=int)
        start_valid = int(48 * 0.15)
        end_valid = int(48 * 0.85)
        binary_mask[start_valid:end_valid] = 1

    # Create row with CST params + CL values + BINARY mask columns
    row_dict = {col: row[col] for col in X_cols}
    for i, cl_val in enumerate(cl_values):
        row_dict[f"CL_{i}"] = cl_val
    for i, mask_val in enumerate(binary_mask):
        row_dict[f"mask_{i}"] = mask_val  # 0 or 1

    clean_data_binary.append(row_dict)

df_binary = pd.DataFrame(clean_data_binary)
output_path_binary = "data/csv/ablation_binary_12CST_params.csv"
df_binary.to_csv(output_path_binary, index=False)
logger.info("Created: %s", output_path_binary)
logger.info("Samples: %d", len(df_binary))
logger.info("Columns: %d (CST + CL + mask_0..mask_47, binary 0/1)", len(df_binary.columns))


# ==================== Datasets 3-5: ExponentialMask with different τ ====================


def create_exponential_mask_dataset(tau: float, output_suffix: str):
    """Create dataset with exponential distance mask at specified τ."""
    logger.info("%s", "\n" + "=" * 70)
    logger.info("DATASET: ExponentialMask (τ=%s)", tau)
    logger.info("%s", "=" * 70)

    clean_data_exp = []

    for index, row in df_base.iterrows():
        # Get CL and existing distance values
        cl_values = row[cl_cols_existing].values.astype(float)

        if has_dist:
            # Recalculate distance with specified τ
            dist_orig = row[dist_cols_existing].values.astype(float)
            distance_indicators = np.ones(48, dtype=float)

            for i in range(48):
                if dist_orig[i] < 0.99:
                    d_norm = -np.log(max(dist_orig[i], 0.01))
                    distance_indicators[i] = np.exp(-d_norm / tau)
                else:
                    distance_indicators[i] = 1.0
        else:
            # Fallback: use simple geometric decay from edges
            distance_indicators = np.ones(48, dtype=float)
            edge_decay = 8  # Points from each edge to decay
            for i in range(edge_decay):
                d_norm = (edge_decay - i) / edge_decay
                distance_indicators[i] = np.exp(-d_norm / tau)
                distance_indicators[47 - i] = np.exp(-d_norm / tau)

        # Create row with distance indicators
        row_dict = {col: row[col] for col in X_cols}
        for i, cl_val in enumerate(cl_values):
            row_dict[f"CL_{i}"] = cl_val
        for i, dist_val in enumerate(distance_indicators):
            row_dict[f"mask_{i}"] = float(dist_val)  # Continuous [0,1]

        clean_data_exp.append(row_dict)

    df_exp = pd.DataFrame(clean_data_exp)
    output_path = f"data/csv/ablation_exp_tau{output_suffix}_12CST_params.csv"
    df_exp.to_csv(output_path, index=False)
    logger.info("Created: %s", output_path)
    logger.info("Samples: %d", len(df_exp))
    logger.info("Columns: %d (CST + CL + mask_0..mask_47, continuous)", len(df_exp.columns))

    # Show example distance values
    sample_dists = df_exp[["mask_0", "mask_23", "mask_47"]].iloc[0].values
    logger.info(
        "Example masks (τ=%s): [%.4f, %.4f, %.4f]",
        tau,
        sample_dists[0],
        sample_dists[1],
        sample_dists[2],
    )

    return output_path


# Create exponential datasets with different τ values
path_tau02 = create_exponential_mask_dataset(tau=0.2, output_suffix="0.2")
path_tau05 = create_exponential_mask_dataset(tau=0.5, output_suffix="0.5")
path_tau10 = create_exponential_mask_dataset(tau=1.0, output_suffix="1.0")

logger.info("%s", "=" * 70)
logger.info("DATASET CREATION COMPLETE")
logger.info("%s", "=" * 70)

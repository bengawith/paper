from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURE_COLUMNS = [f"lower_weight_{i}" for i in range(5)] + [f"upper_weight_{i}" for i in range(5)] + ["leading_edge_weight", "TE_thickness", "alpha_deg"]
TARGET_COLUMNS = ["cl", "log_cd", "cm"]


@dataclass(frozen=True)
class ScalingBundle:
    feature_center: np.ndarray
    feature_scale: np.ndarray
    target_center: dict[str, float]
    target_scale: dict[str, float]


def fit_scaling(frame: pd.DataFrame) -> ScalingBundle:
    features = frame[FEATURE_COLUMNS].to_numpy(float)
    feature_center = np.nanmedian(features, axis=0)
    q75, q25 = np.nanpercentile(features, [75, 25], axis=0)
    feature_scale = q75 - q25
    feature_scale[feature_scale < 1e-9] = 1.0
    target_center: dict[str, float] = {}
    target_scale: dict[str, float] = {}
    for target in TARGET_COLUMNS:
        valid = frame.loc[frame[f"mask_{target}"].astype(bool), target].to_numpy(float)
        target_center[target] = float(np.median(valid))
        scale = float(np.percentile(valid, 75) - np.percentile(valid, 25))
        target_scale[target] = scale if scale >= 1e-9 else 1.0
    return ScalingBundle(feature_center, feature_scale, target_center, target_scale)


class PolarPointDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, frame: pd.DataFrame, scaling: ScalingBundle):
        self.airfoil_ids = frame["airfoil_id"].astype(str).to_numpy()
        self.features = ((frame[FEATURE_COLUMNS].to_numpy(float) - scaling.feature_center) / scaling.feature_scale).astype(np.float32)
        self.targets = {
            target: ((frame[target].fillna(scaling.target_center[target]).to_numpy(float) - scaling.target_center[target]) / scaling.target_scale[target]).astype(np.float32)
            for target in TARGET_COLUMNS
        }
        self.masks = {target: frame[f"mask_{target}"].to_numpy(bool) for target in TARGET_COLUMNS}
        self.weights = frame["airfoil_weight"].to_numpy(np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.features[index]),
            "airfoil_weight": torch.tensor(self.weights[index]),
            **{target: torch.tensor(values[index]) for target, values in self.targets.items()},
            **{f"mask_{target}": torch.tensor(values[index]) for target, values in self.masks.items()},
        }

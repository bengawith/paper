from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler


class UniformAirfoilPointSampler(Sampler[int]):
    def __init__(self, airfoil_ids: np.ndarray, num_samples: int | None = None, seed: int = 20260824):
        self.airfoil_ids = np.asarray(airfoil_ids)
        self.num_samples = int(num_samples or len(self.airfoil_ids))
        self.seed = seed
        self.epoch = 0
        self._groups = {name: np.flatnonzero(self.airfoil_ids == name) for name in np.unique(self.airfoil_ids)}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        names = np.asarray(sorted(self._groups))
        sampled_names = rng.choice(names, size=self.num_samples, replace=True)
        return iter([int(rng.choice(self._groups[name])) for name in sampled_names])

    def __len__(self) -> int:
        return self.num_samples

"""Genuine inference-latency benchmark for the frozen surrogate ensemble.

Measures wall-clock prediction latency on this machine so the manuscript's
"fast surrogate" claim rests on a real, reproducible number rather than an
assumption. Timings use CUDA synchronisation where relevant.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from robust_airfoil.constants import ROOT
from robust_airfoil.modelling.dataset import FEATURE_COLUMNS
from robust_airfoil.modelling.ensemble import LoadedEnsemble

LINEAGE = "robust-v2-current-20260908"
RESULTS = ROOT / "results/robust_v2/lineages" / LINEAGE
PARAM_COLUMNS = FEATURE_COLUMNS[:-1]


def _frame(n_states: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(rng.normal(0.0, 0.1, size=(n_states, len(PARAM_COLUMNS))), columns=pd.Index(PARAM_COLUMNS))
    data["alpha_deg"] = rng.uniform(-5, 15, size=n_states)
    data["airfoil_id"] = "bench"
    return data


def _time(ensemble: LoadedEnsemble, frame: pd.DataFrame, repeats: int, warmup: int) -> float:
    for _ in range(warmup):
        ensemble.predict(frame)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        ensemble.predict(frame)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ensemble = LoadedEnsemble(RESULTS / "full_ensemble")
    results = {"device": device, "ensemble_members": len(ensemble.members)}
    for label, n_states, repeats, warmup in [
        ("single_state", 1, 200, 50),
        ("one_polar_48", 48, 200, 50),
        ("batch_6144", 6144, 50, 10),
    ]:
        frame = _frame(n_states)
        per_call = _time(ensemble, frame, repeats, warmup)
        results[label] = {
            "states": n_states,
            "seconds_per_call": per_call,
            "ms_per_state": 1e3 * per_call / n_states,
            "states_per_second": n_states / per_call,
        }
        print(label, f"{1e3 * per_call:.3f} ms/call", f"{1e3 * per_call / n_states:.5f} ms/state")

    data_path = ROOT / "paper/data/paper_data.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    data["inference_benchmark"] = results
    data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("updated", data_path)


if __name__ == "__main__":
    main()

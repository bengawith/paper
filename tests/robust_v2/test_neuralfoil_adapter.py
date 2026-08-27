import builtins

import numpy as np

from robust_airfoil.benchmarks.neuralfoil import benchmark_neuralfoil


def test_missing_neuralfoil_is_a_diagnostic_skip(monkeypatch):
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == "neuralfoil":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    result = benchmark_neuralfoil({}, np.array([0.0]))
    assert result.status == "unavailable"

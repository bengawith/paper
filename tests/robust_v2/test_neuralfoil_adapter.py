import builtins

import numpy as np

from robust_airfoil.benchmarks.neuralfoil import (
    benchmark_neuralfoil,
    benchmark_neuralfoil_from_coordinates,
)


def test_missing_neuralfoil_is_a_diagnostic_skip(monkeypatch):
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == "neuralfoil":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    result = benchmark_neuralfoil({}, np.array([0.0]))
    assert result.status == "unavailable"


def test_coordinate_adapter_refits_to_neuralfoil_weight_count(monkeypatch):
    captured = {}

    def fake_benchmark(parameters, alpha_deg, reynolds_number=1e6):
        captured.update(parameters)
        return "sentinel"

    monkeypatch.setattr("robust_airfoil.benchmarks.neuralfoil.benchmark_neuralfoil", fake_benchmark)
    x = np.linspace(0.0, 1.0, 21)
    coordinates = np.vstack([
        np.column_stack([x[::-1], 0.1 * np.sin(np.pi * x[::-1])]),
        np.column_stack([x[1:], -0.1 * np.sin(np.pi * x[1:])]),
    ])
    assert benchmark_neuralfoil_from_coordinates(coordinates, np.array([0.0])) == "sentinel"
    assert np.asarray(captured["upper_weights"]).shape == (8,)
    assert np.asarray(captured["lower_weights"]).shape == (8,)

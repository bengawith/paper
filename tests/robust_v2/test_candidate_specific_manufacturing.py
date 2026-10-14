"""Common latent scenarios must act on each candidate's own surface normals."""

from __future__ import annotations

import numpy as np

from robust_airfoil.config import UncertaintyConfig, load_yaml
from robust_airfoil.constants import ROOT
from robust_airfoil.optimisation.candidate_uncertainty import candidate_surface_scenarios


def test_candidate_specific_scenarios_are_distinct_and_deterministic() -> None:
    cfg = UncertaintyConfig.model_validate(load_yaml(ROOT / "configs/robust_v2/uncertainty.yaml"))
    latent = np.array(
        [
            [
                0.05,
                -0.10,
                0.3,
                0.2,
                -0.3,
                0.2,
                0.1,
                -0.15,
                0.22,
                0.1,
                -0.2,
                -0.1,
                0.12,
                -0.3,
                0.15,
                -0.11,
            ],
            [
                -0.10,
                0.2,
                -0.15,
                0.1,
                0.1,
                -0.2,
                0.3,
                -0.1,
                -0.2,
                0.1,
                0.15,
                0.2,
                0.2,
                -0.1,
                0.1,
                -0.15,
            ],
        ]
    )
    # Plausible but geometrically distinct NACA-like CST sections.
    p1 = np.array([-0.13, -0.13, -0.1, -0.08, -0.05, 0.15, 0.17, 0.16, 0.1, 0.07, 0.0, 0.0])
    p2 = np.array([-0.14, -0.12, -0.1, -0.08, -0.05, 0.25, 0.27, 0.25, 0.2, 0.14, 0.0, 0.0])
    a = candidate_surface_scenarios(p1, latent, cfg)
    b = candidate_surface_scenarios(p2, latent, cfg)
    again = candidate_surface_scenarios(p1, latent, cfg)
    np.testing.assert_allclose(a.parameters, again.parameters)
    np.testing.assert_array_equal(a.invalid, again.invalid)
    assert a.parameters.shape == (2, 12)
    assert not np.all(a.invalid) and not np.all(b.invalid)
    # Same shared latent draw does NOT imply same CST delta on different
    # shapes: normals, displacement and CST refit are candidate-specific.
    delta_a = a.parameters[~a.invalid][0] - p1
    delta_b = b.parameters[~b.invalid][0] - p2
    assert np.max(np.abs(delta_a - delta_b)) > 1e-7


def test_candidate_specific_uncertainty_fails_closed_on_bad_shapes() -> None:
    cfg = UncertaintyConfig.model_validate(load_yaml(ROOT / "configs/robust_v2/uncertainty.yaml"))
    with np.testing.assert_raises(ValueError):
        candidate_surface_scenarios(np.zeros(11), np.zeros((2, 16)), cfg)


def test_robust_selection_rejects_all_infeasible_population() -> None:
    from robust_airfoil.optimisation.study import _full_selection_pool

    trial = {
        "solutions": [[0.0] * 12, [0.1] * 12],
        "objectives": [[0.01, 0.02], [0.008, 0.03]],
        "constraint_values": [[0.1] + [-0.1] * 5, [0.2] + [-0.1] * 5],
    }
    with np.testing.assert_raises_regex(RuntimeError, "No feasible full-budget"):
        _full_selection_pool([trial])


def test_candidate_latent_requires_matching_sample_shape() -> None:
    cfg = UncertaintyConfig.model_validate(load_yaml(ROOT / "configs/robust_v2/uncertainty.yaml"))
    with np.testing.assert_raises(ValueError):
        candidate_surface_scenarios(np.zeros(12), np.zeros((3, 15)), cfg)

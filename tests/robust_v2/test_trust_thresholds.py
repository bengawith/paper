import numpy as np
import pandas as pd

from robust_airfoil.modelling.calibrate import apply_trust_model, trust_thresholds
from robust_airfoil.modelling.dataset import FEATURE_COLUMNS, TARGET_COLUMNS


def _calibration():
    return {
        "support_distance_quantiles": {
            "0.5": 0.9,
            "0.9": 1.9,
            "0.95": 2.4,
            "0.99": 4.37,
            "trusted_threshold_from_residual_tolerance": 1.247,
        },
        "ensemble_disagreement_quantiles": {
            "cl": {"0.99": 0.30, "trusted_threshold_from_residual_tolerance": 0.10},
            "log_cd": {"0.99": 0.80, "trusted_threshold_from_residual_tolerance": 0.26},
            "cm": {"0.99": 0.05, "trusted_threshold_from_residual_tolerance": 0.015},
        },
        "alpha_coverage_deg": [-19.75, 19.25],
    }


def test_trust_thresholds_use_strict_residual_tolerance_not_loose_quantile():
    support, disagreement, (lo, hi) = trust_thresholds(_calibration())
    # The authoritative thresholds must be the strict residual-tolerance values,
    # never the permissive 0.99 quantile that previously leaked into the optimiser.
    assert support == 1.247
    assert support < 4.37
    assert disagreement == {"cl": 0.10, "log_cd": 0.26, "cm": 0.015}
    assert (lo, hi) == (-19.75, 19.25)


def test_optimiser_evaluator_shares_the_same_thresholds():
    # The optimisation evaluator must gate trust with exactly the helper values,
    # so a design the online trust model would reject cannot be optimised into.
    import inspect

    from robust_airfoil.optimisation import study as opt_study

    source = inspect.getsource(opt_study.RobustCandidateEvaluator.__init__)
    assert "trust_thresholds(" in source
    assert '["0.99"]' not in source


class _Scaling:
    feature_center = np.zeros(len(FEATURE_COLUMNS))
    feature_scale = np.ones(len(FEATURE_COLUMNS))


def test_apply_trust_model_rejects_support_between_strict_and_loose_thresholds():
    calibration = _calibration()
    development = pd.DataFrame(np.zeros((8, len(FEATURE_COLUMNS))), columns=pd.Index(FEATURE_COLUMNS))
    # One query point at the development manifold (well supported) and one pushed to a
    # support distance of ~2 (inside the old 0.99 gate of 4.37 but outside strict 1.247).
    query = pd.DataFrame(np.zeros((2, len(FEATURE_COLUMNS))), columns=pd.Index(FEATURE_COLUMNS))
    query.loc[1, "lower_weight_0"] = 2.0
    for target in TARGET_COLUMNS:
        query[f"ensemble_std_{target}"] = 0.0
    out = apply_trust_model(development, query, _Scaling(), calibration)
    assert bool(out["trusted_domain"].iloc[0]) is True
    assert bool(out["trusted_domain"].iloc[1]) is False

"""A post-hoc choice of normalising scale cannot satisfy a prospective gate."""
import numpy as np

from robust_airfoil.optimisation.problems import DesignBounds
from robust_airfoil.optimisation.study import _agreement


def test_posthoc_pooled_range_is_diagnostic_not_acceptance():
    runs = [
        {'solutions': [[0., 0.], [1., 1.]], 'objectives': [[1., 2.], [2., 1.]], 'constraint_values': [[-1.], [-1.]]},
        {'solutions': [[0., 0.], [1., 1.]], 'objectives': [[1., 2.], [2., 1.]], 'constraint_values': [[-1.], [-1.]]},
    ]
    result = _agreement(runs, DesignBounds(np.zeros(2), np.ones(2)))
    assert result['pairwise_objective_distance_mean'] == 0.
    assert result['agreement_status'] == 'diagnostic_posthoc_scale_not_accepted'
    assert not result['objective_agreement']['passed']
    assert result['objective_agreement']['descriptive_threshold_satisfied']

"""Predeclared, deterministic selection from verified finite feasible designs.

Inputs are common-context surrogate re-evaluations. No fresh XFOIL or CFD
outcome enters selection. Ties use lexicographic geometry coefficients, not
execution order or subsequent verification performance.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def _checked_rows(parameters: np.ndarray, objectives: np.ndarray, constraints: np.ndarray,
                  objective_count: int) -> tuple[np.ndarray,np.ndarray,np.ndarray,int]:
    x,f,g = np.asarray(parameters,float),np.asarray(objectives,float),np.asarray(constraints,float)
    if objective_count == 1:
        f = f.reshape(-1,1)
    if x.ndim != 2 or x.shape[1] != 12 or f.shape != (len(x),objective_count) or g.ndim != 2 or len(g) != len(x):
        raise ValueError('Finalist parameters/objectives/constraints must have aligned twelve-parameter rows')
    finite = np.isfinite(x).all(axis=1) & np.isfinite(f).all(axis=1) & np.isfinite(g).all(axis=1)
    feasible = finite & np.all(g <= 0,axis=1) & np.all(f > 0,axis=1)
    count = int(feasible.sum())
    if not count:
        raise ValueError('No finite feasible design exists under the unmodified constraints')
    x,f,g = x[feasible],f[feasible],g[feasible]
    order = np.lexsort(tuple(x[:,index] for index in reversed(range(x.shape[1]))))
    x,f,g = x[order],f[order],g[order]
    # Common-context reevaluation should give a single objective/constraint
    # vector for each exact geometry. Refuse contradictory duplicates.
    retain = []
    for index in range(len(x)):
        if index and np.array_equal(x[index],x[index-1]):
            if not np.allclose(f[index],f[index-1],rtol=0,atol=1e-10) or not np.allclose(g[index],g[index-1],rtol=0,atol=1e-10):
                raise ValueError('Duplicate geometry has inconsistent common-context evaluation')
        else:
            retain.append(index)
    return x[retain],f[retain],g[retain],count


def select_controlled_risk_finalists(parameters: np.ndarray, objectives: np.ndarray,
                                     constraints: np.ndarray, objective_scales: np.ndarray) -> dict[str, Any]:
    scales = np.asarray(objective_scales,float)
    if scales.shape != (2,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('Predeclared reference objective scales must be finite and positive')
    x,f,g,count = _checked_rows(parameters,objectives,constraints,2)
    nondominated = np.asarray(NonDominatedSorting().do(f,only_non_dominated_front=True),int)
    front_x,front_f,front_g = x[nondominated],f[nondominated],g[nondominated]
    order = np.lexsort(tuple(front_x[:,index] for index in reversed(range(front_x.shape[1]))))
    front_x,front_f,front_g = front_x[order],front_f[order],front_g[order]
    ideal = front_f.min(axis=0)
    distance = np.linalg.norm((front_f-ideal)/scales,axis=1)
    choices = {'risk_knee':int(np.argmin(distance)),
               'minimum_mean':int(np.argmin(front_f[:,0])),
               'minimum_tail':int(np.argmin(front_f[:,1]))}
    return {
        'schema':'predeclared-risk-finalist-selection-v1',
        'input_count':len(np.asarray(parameters)), 'feasible_count':count,
        'distinct_feasible_geometries':len(x),'nondominated_count':len(front_x),
        'objective_scales':scales.tolist(),'pooled_feasible_ideal':ideal.tolist(),
        'selected':{name:{'parameters':front_x[index].tolist(),'objectives':front_f[index].tolist(),
                          'constraint_values':front_g[index].tolist(),
                          'reference_scaled_ideal_distance':float(distance[index])}
                    for name,index in choices.items()},
        'front_parameters':front_x.tolist(),'front_objectives':front_f.tolist(),
        'front_constraint_values':front_g.tolist(),
        'tie_break':'Lexicographic twelve-coefficient geometry ordering',
        'rule':'Minimum reference-scaled Euclidean distance to the pooled feasible ideal; separate minimum-mean and minimum-tail diagnostics',
        'solver_outcomes_used_for_selection':False,
        'claim_boundary':'Selected finite-search candidates, not a unique or global Pareto optimum',
    }


def select_controlled_nominal_finalist(parameters: np.ndarray, objectives: np.ndarray,
                                       constraints: np.ndarray) -> dict[str, Any]:
    x,f,g,count = _checked_rows(parameters,objectives,constraints,1)
    index = int(np.argmin(f[:,0]))
    return {'parameters':x[index].tolist(),'nominal_weighted_cd':float(f[index,0]),
            'constraint_values':g[index].tolist(),'feasible_count':count,
            'distinct_feasible_geometries':len(x),
            'selection':'Minimum nominal weighted drag among finite feasible final control populations; lexicographic geometry tie-break',
            'solver_outcomes_used_for_selection':False}

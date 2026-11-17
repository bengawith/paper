"""Matched-budget nominal-drag control with explicit feasibility and checkpoints.

This changes only the objective being optimised. Its real evaluator supplies
exactly the same nominal geometry, service-root, moment and screening checks as
the risk-aware campaign. No solver verification outcome selects the control.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem

from robust_airfoil.hashing import hash_object, sha256_file
from robust_airfoil.optimisation.checkpointing import (
    assert_campaign_resources,
    load_algorithm_checkpoint,
    save_algorithm_checkpoint,
)
from robust_airfoil.optimisation.problems import DesignBounds
from robust_airfoil.provenance import write_json_atomic


class NominalControlProblem(Problem):
    def __init__(self, evaluator: Any, bounds: DesignBounds):
        super().__init__(n_var=len(bounds.lower), n_obj=1, n_ieq_constr=6,
                         xl=bounds.lower, xu=bounds.upper)
        self.candidate_evaluator = evaluator

    def _evaluate(self, candidates: np.ndarray, out: dict[str, np.ndarray], *args: object, **kwargs: object) -> None:
        constraints, diagnostics = self.candidate_evaluator._nominal_constraints(candidates)
        constraints = np.asarray(constraints, float).copy()
        objectives = np.ones((len(candidates), 1), float)
        for index, diagnostic in enumerate(diagnostics):
            value = diagnostic.get('nominal_weighted_drag')
            usable = (diagnostic.get('nominal_required_lift_feasible', False)
                      and value is not None and np.isfinite(value) and value > 0
                      and np.isfinite(constraints[index]).all())
            if usable:
                objectives[index, 0] = float(value)
            else:
                constraints[index, ~np.isfinite(constraints[index])] = 1.0
                constraints[index, -1] = max(1.0, constraints[index, -1])
        out['F'] = objectives
        out['G'] = constraints


def initial_nominal_population(evaluator: Any, bounds: DesignBounds, reference: np.ndarray,
                               observed: np.ndarray, population: int, seed: int) -> np.ndarray:
    """The shared reference/observed/random seeding rule, without duplicate loss.

    A duplicate reference among observed designs must not shrink generation one
    below the advertised population when pymoo eliminates duplicates.
    """
    reference = np.asarray(reference, float)
    observed = np.asarray(observed, float)
    low, high = np.asarray(bounds.lower, float), np.asarray(bounds.upper, float)
    if reference.shape != low.shape or low.shape != high.shape or population < 2:
        raise ValueError('Population and reference/bounds dimensions must align')
    if not np.isfinite(np.r_[reference, low, high]).all() or np.any(high < low) or not np.any(high > low):
        raise ValueError('Initial population requires finite nondegenerate bounds')
    if np.any(reference < low) or np.any(reference > high):
        raise ValueError('The reference must lie inside the declared bounds')
    rng = np.random.default_rng(seed)
    values = rng.uniform(low, high, size=(population, len(reference)))
    values[0] = reference
    if observed.ndim != 2 or observed.shape[1] != len(reference):
        raise ValueError('Observed geometry matrix has an incompatible shape')
    admitted = np.isfinite(observed).all(axis=1) & np.all((observed >= low) & (observed <= high), axis=1)
    observed = observed[admitted]
    if len(observed):
        _, first = np.unique(observed, axis=0, return_index=True)
        observed = observed[np.sort(first)]
        observed = observed[np.linalg.norm(observed-reference, axis=1) > 1e-12]
    if len(observed):
        constraints, _ = evaluator._nominal_constraints(observed)
        violations = np.sum(np.maximum(constraints, 0), axis=1)
        violations[~np.isfinite(violations)] = np.inf
        distance = np.linalg.norm((observed-reference)/np.maximum(high-low, 1e-12), axis=1)
        order = np.lexsort((distance, violations))
        selected = min(len(order), max(0, population//2-1))
        values[1:selected+1] = observed[order[:selected]]
    for index in range(1, population):
        attempts = 0
        while np.any(np.linalg.norm(values[:index]-values[index], axis=1) <= 1e-12):
            values[index] = rng.uniform(low, high)
            attempts += 1
            if attempts > 10000:
                raise ValueError('Declared bounds cannot support the requested distinct initial population')
    return values


class NominalControlHistory(Callback):
    def __init__(self):
        super().__init__()
        self.history: list[dict[str, Any]] = []

    def notify(self, algorithm: Any) -> None:
        objectives = np.asarray(algorithm.pop.get('F'), float).reshape(-1)
        constraints = np.asarray(algorithm.pop.get('G'), float)
        feasible = np.isfinite(objectives) & np.isfinite(constraints).all(axis=1) & np.all(constraints <= 0, axis=1)
        self.history.append({'generation':int(algorithm.n_gen),
                             'evaluations':int(algorithm.evaluator.n_eval),
                             'population_count':len(objectives),
                             'feasible_count':int(feasible.sum()),
                             'minimum_feasible_nominal_drag':float(objectives[feasible].min()) if feasible.any() else None,
                             'maximum_constraint_violation':float(np.maximum(constraints,0).max())})


def run_nominal_control_profile(evaluator: Any, bounds: DesignBounds, reference: np.ndarray,
                                observed: np.ndarray, seeds: list[int], population: int,
                                generations: int, output: Path, scientific_context: dict[str, Any]) -> dict[str, Any]:
    if not seeds or len(set(seeds)) != len(seeds) or generations < 1:
        raise ValueError('Every nominal-control seed must be distinct and have a positive generation budget')
    output.mkdir(parents=True, exist_ok=True)
    context = {'scientific_context':scientific_context,
               'operator_sha256':sha256_file(Path(__file__)),
               'population':population, 'generations':generations, 'seeds':seeds,
               'reference':np.asarray(reference, float).tolist(),
               'bounds':{'lower':bounds.lower.tolist(),'upper':bounds.upper.tolist()},
               'observed_parameters_sha256':hash_object(np.asarray(observed,float).tolist())}
    context_hash = hash_object(context)
    protocol_path = output/'NOMINAL_CONTROL_PROTOCOL.json'
    if protocol_path.exists():
        stored = json.loads(protocol_path.read_text(encoding='utf-8'))
        if stored['context_hash'] != context_hash or stored['context'] != context:
            raise ValueError('Nominal-control scientific context mismatch')
    else:
        write_json_atomic(protocol_path, {'schema':'matched-budget-nominal-control-v1','context_hash':context_hash,'context':context})
    started = time.monotonic()
    runs = []
    for seed in seeds:
        target = output/f'nominal_seed_{seed}.json'
        receipt_path = output/f'nominal_seed_{seed}.receipt.json'
        if target.exists():
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            if receipt['context_hash'] != context_hash or receipt['sha256'] != sha256_file(target):
                raise ValueError('Stored nominal-control result context/hash mismatch')
            run = json.loads(target.read_text(encoding='utf-8'))
            if len(run['history']) != generations or run['seed'] != seed:
                raise ValueError('Stored nominal-control run is not the requested complete profile')
            runs.append(run)
            continue
        problem = NominalControlProblem(evaluator, bounds)
        checkpoint_root = output/'checkpoints'/f'seed_{seed}'
        checkpoint_context = {'seed':seed,'context_hash':context_hash,'population':population,'generations':generations}
        if (checkpoint_root/'algorithm_checkpoint.json').exists():
            algorithm = load_algorithm_checkpoint(checkpoint_root,checkpoint_context,problem=problem,evaluator=evaluator)
        else:
            initial = initial_nominal_population(evaluator,bounds,reference,observed,population,seed)
            # Use the same NSGA-II operators as the robust campaign, with a
            # single nominal-drag objective. Its global tournament implementation
            # is serialisable; the alternative GA comparator is a local closure.
            algorithm = NSGA2(pop_size=population,sampling=initial)
            algorithm.setup(problem,termination=('n_gen',generations),seed=seed,
                            callback=NominalControlHistory(),verbose=False)
        while algorithm.has_next():
            assert_campaign_resources(output)
            algorithm.next()
            save_algorithm_checkpoint(algorithm,checkpoint_root,checkpoint_context)
            history = algorithm.callback.history
            write_json_atomic(checkpoint_root/'GENERATION_PROGRESS.json',
                              {'seed':seed,'completed_generations':len(history),'required_generations':generations,'last':history[-1]})
            if len(history) % 25 == 0 or len(history) == 1:
                print(f'NOMINAL seed={seed} generation={len(history)}/{generations} feasible={history[-1]["feasible_count"]}/{population}',flush=True)
        if len(algorithm.callback.history) != generations:
            raise AssertionError('Nominal control terminated at an incomplete generation count')
        if any(row['population_count'] != population for row in algorithm.callback.history):
            raise AssertionError('Nominal control did not retain the declared full population')
        candidates = np.asarray(algorithm.pop.get('X'),float)
        checked: dict[str, np.ndarray] = {}
        problem._evaluate(candidates,checked)
        objectives, constraints = checked['F'],checked['G']
        feasible = np.all(constraints <= 0,axis=1) & np.isfinite(objectives[:,0])
        run = {'seed':seed,'context_hash':context_hash,'solutions':candidates.tolist(),
               'objectives':objectives.tolist(),'constraint_values':constraints.tolist(),
               'feasible_count':int(feasible.sum()),'history':algorithm.callback.history,
               'minimum_feasible_nominal_drag':float(objectives[feasible,0].min()) if feasible.any() else None,
               'final_recheck_max_objective_difference':float(np.max(abs(objectives-np.asarray(algorithm.pop.get('F'),float)))),
               'selection_semantics':'All final individuals retained; only explicitly feasible rows may be selected as the nominal control'}
        write_json_atomic(target,run)
        write_json_atomic(receipt_path,{'context_hash':context_hash,'sha256':sha256_file(target)})
        runs.append(run)
        print(f'NOMINAL_SEED_COMPLETE seed={seed} feasible={run["feasible_count"]}',flush=True)
    return {'schema':'matched-budget-nominal-control-result-v1','context_hash':context_hash,
            'population':population,'generations':generations,'independent_seeds':len(seeds),
            'runs':runs,'invocation_seconds':time.monotonic()-started,
            'scope':'Nominal-only objective with the same hard nominal constraints and initial seeding policy; not a robust optimum or independent solver validation'}

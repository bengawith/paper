"""Integrate the reproduced scientific corrections and checkpointed execution path."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
p=ROOT/'src/robust_airfoil/optimisation/study.py'
t=p.read_text(encoding='utf-8')

def replace(old,new):
    global t
    assert t.count(old)==1,(old[:120],t.count(old))
    t=t.replace(old,new)

replace('from robust_airfoil.optimisation.candidate_uncertainty import candidate_surface_scenarios',
        'from robust_airfoil.optimisation.candidate_uncertainty import PreparedCandidateScenarios\nfrom robust_airfoil.optimisation.checkpointing import load_algorithm_checkpoint, save_algorithm_checkpoint')
replace('        self.physical_uncertainty = uncertainty\n',
        '        self.physical_uncertainty = uncertainty\n        self.prepared_scenarios = (PreparedCandidateScenarios(candidate_latent, uncertainty)\n                                   if candidate_latent is not None and uncertainty is not None else None)\n')
replace('        self.x = x\n','        self.x = x\n        self.validation_x = (1 - np.cos(np.linspace(0, np.pi, 2001))) / 2\n')
replace('            validity = validate_geometry(upper, lower)\n            metrics = geometry_metrics(upper, lower)',
        '            dense_upper, dense_lower = reconstruct_cst(candidate, self.validation_x)\n            validity = validate_geometry(np.column_stack([self.validation_x, dense_upper]),\n                                         np.column_stack([self.validation_x, dense_lower]))\n            metrics = geometry_metrics(upper, lower)')
replace('''            scenario_rows = [
                candidate_surface_scenarios(c, self.candidate_latent, self.physical_uncertainty)
                for c in candidates
            ]''','''            if self.prepared_scenarios is None:
                raise ValueError("Missing prepared candidate-specific uncertainty operator")
            scenario_rows = [self.prepared_scenarios.evaluate(c) for c in candidates]''')
replace('        missing_roots: list[int] = []\n','        missing_roots: list[int] = []\n        usable_samples: list[np.ndarray] = []\n')
replace('            violations = missing | untrusted | candidate_invalid[candidate_index]\n',
        '            violations = missing | untrusted | candidate_invalid[candidate_index]\n            usable_samples.append(~violations)\n')
replace('''            diagnostic["mean_lift_by_alpha"] = np.mean(cl[index], axis=0).tolist()
            diagnostic["lift_std_by_alpha"] = np.std(cl[index], axis=0, ddof=1).tolist()
            diagnostic["mean_cm_by_alpha"] = np.mean(cm[index], axis=0).tolist()''',
        '''            diagnostic.update(summarise_valid_aerodynamics(cl[index], cm[index], usable_samples[index]))''')
start=t.index('    def ava_objective(')
end=t.index('\n\nclass _RobustProblem',start)
t=t[:start]+'''    def ava_objective(self, candidate: np.ndarray, sensitivity_lambda: float) -> float:
        candidate = np.asarray(candidate, dtype=float)
        if self.candidate_latent is not None:
            if self.prepared_scenarios is None:
                raise ValueError("Missing candidate-specific baseline uncertainty")
            scenarios = self.prepared_scenarios.evaluate(candidate)
            perturbed, invalid = scenarios.parameters, scenarios.invalid
        else:
            perturbed = candidate[None, :] + self.deltas
            invalid = self.manufacturing_invalid_flags
        predictions = self.ensemble.predict(_frame(perturbed, self.alpha))
        lift = predictions["prediction_cl"].to_numpy(float).reshape(len(perturbed), len(self.alpha))
        trusted = self._support_and_disagreement(predictions, lift.shape)
        constraints, _ = self._nominal_constraints(candidate[None, :])
        violation = float(np.sum(np.maximum(constraints[0], 0)) * 1_000)
        failed = invalid | ~np.all(trusted, axis=1) | ~np.isfinite(lift).all(axis=1)
        if failed.any():
            # A sensitivity baseline has no physical lift outcome for a failed
            # scenario. Fail closed rather than replacing it with nominal lift.
            return 1_000_000.0 + violation + int(failed.sum())
        score = ava_lift_sensitivity(
            np.mean(lift, axis=0), np.std(lift, axis=0, ddof=1),
            np.full(len(self.alpha), 1 / len(self.alpha)), sensitivity_lambda,
        )
        return score + violation


def summarise_valid_aerodynamics(cl: np.ndarray, cm: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    cl, cm, valid = np.asarray(cl,float), np.asarray(cm,float), np.asarray(valid,bool)
    if cl.ndim != 2 or cl.shape != cm.shape or valid.shape != (len(cl),):
        raise ValueError("Aerodynamic diagnostics require aligned sample matrices")
    valid = valid & np.isfinite(cl).all(axis=1) & np.isfinite(cm).all(axis=1)
    count = int(valid.sum())
    unavailable = [None] * cl.shape[1]
    return {
        "conditional_aerodynamic_sample_count": count,
        "aerodynamic_moment_scope": "conditional on valid geometry/refit, finite required-lift roots and empirical trust; invalid placeholders excluded",
        "mean_lift_by_alpha": np.mean(cl[valid],axis=0).tolist() if count else unavailable,
        "lift_std_by_alpha": np.std(cl[valid],axis=0,ddof=1).tolist() if count > 1 else unavailable,
        "mean_cm_by_alpha": np.mean(cm[valid],axis=0).tolist() if count else unavailable,
    }
'''+t[end:]
replace('''    if len(x):
        _, constraints, diagnostics = evaluator.evaluate(x)
    else:
        constraints = np.empty((0, 6))
    return {
        "schema_version": 5,''','''    recheck_difference = 0.0
    if len(x):
        reevaluated, constraints, diagnostics = evaluator.evaluate(x)
        recheck_difference = float(np.max(abs(reevaluated-f)))
        f = reevaluated
    else:
        constraints = np.empty((0, 6))
    return {
        "schema_version": 6,
        "final_recheck_max_objective_difference": recheck_difference,''')
replace('def _agreement(runs: list[dict[str, Any]], bounds: DesignBounds) -> dict[str, Any]:',
        'def _agreement(runs: list[dict[str, Any]], bounds: DesignBounds, *, objective_scales: np.ndarray | None = None) -> dict[str, Any]:')
replace('''    objective_scale = (
        np.maximum(np.ptp(all_objectives, axis=0), 1e-12) if len(all_objectives) else np.ones(2)
    )''','''    objective_scale = (
        np.asarray(objective_scales,float) if objective_scales is not None
        else np.maximum(np.ptp(all_objectives, axis=0), 1e-12) if len(all_objectives) else np.ones(2)
    )''')
replace('    objective_metrics["passed"] = False\n','    if objective_scales is None:\n        objective_metrics["passed"] = False\n')
replace('''        "agreement_status": (
            "diagnostic_posthoc_scale_not_accepted"
            if feasible_runs
            else "insufficient_feasible_runs"
        ),''','''        "agreement_status": (
            ("pass" if objective_metrics["passed"] else "fail")
            if objective_scales is not None else
            "diagnostic_posthoc_scale_not_accepted" if feasible_runs else "insufficient_feasible_runs"
        ),''')
replace('        "objective_scale_context": "posthoc_pooled_range_not_a_predeclared_acceptance_scale",',
        '        "objective_scale_context": "predeclared_reference_objectives" if objective_scales is not None else "posthoc_pooled_range_not_a_predeclared_acceptance_scale",')
replace('''    ensemble_manifest_hash: str,
    uncertainty: UncertaintyConfig,
) -> dict[str, Any]:''','''    ensemble_manifest_hash: str,
    uncertainty: UncertaintyConfig,
    objective_scales: np.ndarray | None = None,
    checkpoint_each_generation: bool = False,
) -> dict[str, Any]:''')
replace('            "objective_schema_version": 5,','            "objective_schema_version": 6,\n            "objective_agreement_scales": None if objective_scales is None else objective_scales.tolist(),\n            "dense_geometry_verification_points": 2001,')
replace('                and cached.get("schema_version") == 5','                and cached.get("schema_version") == 6')
replace('''        result = minimize(
            problem,
            NSGA2(pop_size=profile.population, sampling=initial_population),
            ("n_gen", profile.generations),
            seed=seed,
            callback=callback,
            verbose=False,
        )''','''        if checkpoint_each_generation:
            checkpoint_root = output_root / "checkpoints" / f"seed_{seed}"
            checkpoint_context = {"seed":seed,"cache_context_hash":cache_context_hash,
                                  "population":profile.population,"generations":profile.generations,
                                  "sample_count":profile.uncertainty_samples}
            if (checkpoint_root / "algorithm_checkpoint.json").is_file():
                algorithm = load_algorithm_checkpoint(checkpoint_root,checkpoint_context,problem=problem,evaluator=evaluator)
            else:
                algorithm = NSGA2(pop_size=profile.population,sampling=initial_population)
                algorithm.setup(problem,termination=("n_gen",profile.generations),seed=seed,callback=callback,verbose=False)
            while algorithm.has_next():
                algorithm.next()
                save_algorithm_checkpoint(algorithm,checkpoint_root,checkpoint_context)
                history = algorithm.callback.history
                (checkpoint_root/"GENERATION_PROGRESS.json").write_text(json.dumps({"seed":seed,"completed_generations":len(history),"required_generations":profile.generations,"last":history[-1]},indent=2),encoding="utf-8")
                if len(history) % 5 == 0 or len(history) == 1:
                    print(f"PRODUCTION seed={seed} generation={len(history)}/{profile.generations} feasible={history[-1]['feasible_count']}/{profile.population}",flush=True)
            result = algorithm.result()
            callback = algorithm.callback
        else:
            result = minimize(
                problem,
                NSGA2(pop_size=profile.population, sampling=initial_population),
                ("n_gen", profile.generations), seed=seed, callback=callback, verbose=False,
            )''')
replace('        "agreement": _agreement(runs, bounds),','        "agreement": _agreement(runs, bounds, objective_scales=objective_scales),')
p.write_text(t,encoding='utf-8')
p=ROOT/'src/robust_airfoil/optimisation/candidate_uncertainty.py'
t=p.read_text(encoding='utf-8')
old='''                ups.append(self._resample_valid(upper[i]))
                lows.append(self._resample_valid(lower[i]))
                retained.append(i)'''
new='''                top_values = self._resample_valid(upper[i])
                bottom_values = self._resample_valid(lower[i])
                ups.append(top_values)
                lows.append(bottom_values)
                retained.append(i)'''
assert t.count(old)==1
p.write_text(t.replace(old,new),encoding='utf-8')
print('PRODUCTION_EVALUATOR_INTEGRATED')

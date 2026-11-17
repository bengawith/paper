"""Connect the regression-proven duplicate-free seeding to real robust runs."""
from pathlib import Path
import hashlib
import json
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
path=ROOT/'src/robust_airfoil/optimisation/study.py'
original=path.read_bytes()
text=original.decode('utf-8').replace('\r\n','\n')
anchor='from robust_airfoil.optimisation.objectives import ('
assert text.count(anchor)==1
text=text.replace(anchor,'from robust_airfoil.optimisation.nominal_control import initial_nominal_population\n'+anchor)
start=text.index('        rng = np.random.default_rng(seed)',text.index('def _run_nsga_profile('))
end=text.index('        if checkpoint_each_generation:',start)
text=text[:start]+'''        observed_designs = development_points[PARAMETER_COLUMNS].drop_duplicates().to_numpy(float)
        initial_population = initial_nominal_population(
            evaluator, bounds, reference_parameters, observed_designs, profile.population, seed,
        )
'''+text[end:]
anchor='''                history = algorithm.callback.history
                (checkpoint_root/"GENERATION_PROGRESS.json")'''
replacement='''                history = algorithm.callback.history
                if history[-1]["population_count"] != profile.population:
                    raise AssertionError("The robust generation does not contain the declared full population")
                (checkpoint_root/"GENERATION_PROGRESS.json")'''
assert text.count(anchor)==1
text=text.replace(anchor,replacement)
anchor='''        seed_path.write_text(json.dumps(serialised, indent=2, sort_keys=True), encoding="utf-8")'''
replacement='''        if len(serialised["history"]) != profile.generations or any(
            row["population_count"] != profile.population for row in serialised["history"]
        ):
            raise AssertionError("Robust seed does not satisfy its full generation/population contract")
        seed_path.write_text(json.dumps(serialised, indent=2, sort_keys=True), encoding="utf-8")'''
assert text.count(anchor)==1
text=text.replace(anchor,replacement)
(HERE/'study_before_population_repair.py.txt').write_bytes(original)
if path.read_bytes()!=original:
    raise RuntimeError('Concurrent source edit: reconcile before changing production initialisation')
path.write_text(text,encoding='utf-8')
(HERE/'POPULATION_REPAIR.json').write_text(json.dumps({'source':path.relative_to(ROOT).as_posix(),'before_sha256':hashlib.sha256(original).hexdigest(),'after_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'root_cause':'The reference also appeared among observed seeds, so duplicate elimination reduced generation one below the declared population.','fix':'Shared distinct reference/observed/random seeding in robust and nominal controls; actual full population checked in every recorded generation.','scientific_campaigns_run_under_changed_initialiser':False},indent=2),encoding='utf-8')
print('FULL_POPULATION_INITIALISATION_CONTRACT_CONNECTED')

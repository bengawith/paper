"""The controlled study's finer service grid reaches real production consumers."""
import numpy as np

from robust_airfoil import research_lineage
from robust_airfoil.constants import ROOT


def test_controlled_grid_is_loaded_without_changing_engineering_mission_or_budget():
    load=getattr(research_lineage,'load_optimisation_for_lineage',None)
    assert load is not None, 'Declared controlled service grid is not connected to consumers'
    historical=load(ROOT,'rebuilt')
    controlled=load(ROOT,'controlled')
    assert historical.service_alpha_deg is None
    np.testing.assert_array_equal(controlled.service_alpha_deg,np.arange(0.,8.0001,.25))
    assert controlled.service_targets==historical.service_targets
    assert controlled.relative_constraints==historical.relative_constraints
    assert controlled.full_nsga2==historical.full_nsga2
    assert controlled.full_nsga2.population==128
    assert controlled.full_nsga2.generations==200
    assert controlled.full_nsga2.uncertainty_samples==64
    assert controlled.full_nsga2.independent_seeds==10
    assert controlled.ava_baseline.alpha_deg==historical.ava_baseline.alpha_deg


def test_controlled_production_and_profile_use_the_explicit_configuration_loader():
    for filename in ['run_repaired_production.py','profile_repaired_evaluator.py']:
        source=(ROOT/'scripts/robust_v2'/filename).read_text(encoding='utf-8')
        assert 'load_optimisation_for_lineage(ROOT,args.lineage)' in source or 'load_optimisation_for_lineage(ROOT, args.lineage)' in source

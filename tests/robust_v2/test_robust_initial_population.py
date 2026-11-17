"""The declared robust population must not shrink from duplicate reference seeding."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from robust_airfoil.config import OptimisationConfig, UncertaintyConfig, load_yaml
from robust_airfoil.constants import FEATURE_COLUMNS, ROOT
from robust_airfoil.optimisation import study
from robust_airfoil.optimisation.problems import DesignBounds


def test_actual_robust_profile_passes_the_full_distinct_initial_population(monkeypatch,tmp_path):
    opt=OptimisationConfig.model_validate(load_yaml(ROOT/'configs/robust_v2/optimisation.yaml'))
    unc=UncertaintyConfig.model_validate(load_yaml(ROOT/'configs/robust_v2/uncertainty.yaml'))
    profile=opt.full_nsga2.model_copy(update={'population':8,'generations':1,'uncertainty_samples':2,'independent_seeds':1})
    reference=np.zeros(12)
    observed=np.zeros((3,13))
    observed[1,0]=.1
    observed[2,0]=.2
    points=pd.DataFrame(observed,columns=FEATURE_COLUMNS)
    captured={}
    class CompleteCapture(Exception):
        pass
    class Evaluator:
        def __init__(self,*args,**kwargs):
            self.counters={'candidate_evaluations':0}
        def evaluate(self,values):
            return np.ones((len(values),2)),np.full((len(values),6),-1.),[{} for _ in values]
        def _nominal_constraints(self,values):
            return np.full((len(values),6),-1.),[{} for _ in values]
    def stop(problem,algorithm,*args,**kwargs):
        captured['initial']=np.asarray(algorithm.initialization.sampling).copy()
        raise CompleteCapture
    monkeypatch.setattr(study,'RobustCandidateEvaluator',Evaluator)
    monkeypatch.setattr(study,'minimize',stop)
    ensemble=SimpleNamespace(artifact_hashes={'fixture':'not-a-scientific-model'},device='cpu')
    with pytest.raises(CompleteCapture):
        study._run_nsga_profile('population-contract',profile,opt,reference,np.zeros((2,16)),np.zeros(2,bool),
                                points,ensemble,{},DesignBounds(-np.ones(12),np.ones(12)),tmp_path,0,'fixture-manifest',unc)
    initial=captured['initial']
    assert initial.shape==(8,12)
    assert len(np.unique(initial,axis=0))==8
    assert int(np.all(initial==reference,axis=1).sum())==1

"""Strict selected-design solver accounting; synthetic arrays are not research data."""
import importlib
import importlib.util

import numpy as np
import pandas as pd
import pytest


def module():
    name='robust_airfoil.validation.controlled_selected'
    assert importlib.util.find_spec(name) is not None, 'Explicit selected-design solver protocol is absent'
    return importlib.import_module(name)


def points():
    alpha=np.arange(0.,11.0001,.25)
    return pd.DataFrame({'alpha_deg':alpha,'cl':.2+.12*alpha,'cd':.006+.0001*alpha**2,'cm':np.full(len(alpha),-.04)})


def test_warmup_occurs_before_polar_recording_and_requested_grid_is_complete():
    functions=module()
    command=functions.selected_xfoil_commands(panel_count=160)
    assert command.index('ASEQ -2 0 0.5')<command.index('PACC\npolar.txt')
    assert command.index('PACC\npolar.txt')<command.index('ASEQ 0 11 0.25')
    assert 'PPAR\nN 160' in command
    assert 'VISC 1000000' in command and 'MACH 0' in command and 'N 9' in command
    assert 'ITER 200' in command


def test_only_complete_clean_finite_positive_sweeps_with_all_service_roots_are_usable():
    functions=module()
    accepted=functions.assess_selected_polar(points(),return_code=0,timed_out=False)
    assert accepted['solver_complete'] and accepted['required_lift_feasible'] and accepted['usable']
    assert accepted['expected_points']==45 and accepted['observed_valid_points']==45
    assert len(accepted['required_lift_roots'])==3
    assert all(0<=root['alpha_deg']<=8 for root in accepted['required_lift_roots'])
    partial=functions.assess_selected_polar(points().drop(index=20),return_code=0,timed_out=False)
    assert not partial['usable'] and partial['weighted_cd'] is None
    assert partial['missing_alpha_deg']==[5.]
    bad=points()
    bad.loc[5,'cd']=-.1
    assert not functions.assess_selected_polar(bad,return_code=0,timed_out=False)['usable']
    assert not functions.assess_selected_polar(points(),return_code=1,timed_out=False)['usable']
    assert not functions.assess_selected_polar(points(),return_code=None,timed_out=True)['usable']


def test_lift_roots_outside_fixed_service_domain_or_after_a_stall_are_refused():
    functions=module()
    slow=points()
    slow['cl']=.1+.09*slow.alpha_deg
    result=functions.assess_selected_polar(slow,return_code=0,timed_out=False)
    assert result['solver_complete'] and not result['required_lift_feasible'] and not result['usable']
    assert result['weighted_cd'] is None
    stalled=points()
    stalled.loc[stalled.alpha_deg>=3.,'cl']=.3+.12*(stalled.loc[stalled.alpha_deg>=3.,'alpha_deg']-3.)
    assert not functions.assess_selected_polar(stalled,return_code=0,timed_out=False)['usable']


def test_conflicting_repeated_angles_are_not_averaged_into_a_success():
    functions=module()
    repeated=pd.concat([points(),points().iloc[[1]].assign(cl=99.)],ignore_index=True)
    result=functions.assess_selected_polar(repeated,return_code=0,timed_out=False)
    assert not result['usable']
    assert result['conflicting_alpha_deg']==[.25]


@pytest.mark.parametrize('panels',[0,10,10000])
def test_numerical_panel_protocol_cannot_be_silently_invalid(panels):
    with pytest.raises(ValueError):
        module().selected_xfoil_commands(panel_count=panels)


def test_an_invalid_duplicate_cannot_be_dropped_to_relabel_a_sweep_clean():
    functions=module()
    invalid_duplicate=points().iloc[[1]].copy()
    invalid_duplicate['cd']=-.1
    mixed=pd.concat([points(),invalid_duplicate],ignore_index=True)
    result=functions.assess_selected_polar(mixed,return_code=0,timed_out=False)
    assert not result['usable']
    assert result['invalid_printed_rows']==1

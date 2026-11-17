"""Software fixtures only: exact-coordinate labels must be actual clean solver rows."""
import numpy as np
import pandas as pd

from robust_airfoil.data import builder


def functions():
    admit = getattr(builder, 'admit_local_polar_rows', None)
    merge = getattr(builder, 'merge_local_polar_rows', None)
    assert admit is not None and merge is not None, 'Controlled solver-row admission is not implemented'
    return admit, merge


def frame():
    return pd.DataFrame({'alpha_deg':[0.,.25,.5,.75,1.,99.],
                         'cl':[0.,.1,.2,np.nan,.4,8.],
                         'cd':[.01,.011,-.1,.012,.014,.5],
                         'cm':[-.02]*6,'source_segment':['fixture']*6})


def test_only_real_finite_positive_requested_rows_are_admitted():
    admit, _ = functions()
    rows = admit(frame(),np.arange(0,1.01,.25),return_code=0,timed_out=False)
    assert rows.alpha_deg.tolist() == [0.,.25,1.]
    assert rows.cl.tolist() == [0.,.1,.4]
    np.testing.assert_allclose(rows.log_cd,np.log([.01,.011,.014]),rtol=0,atol=0)
    assert rows[['mask_cl','mask_log_cd','mask_cm']].all().all()
    assert rows.source_segment.tolist() == ['fixture']*3


def test_abnormal_exit_or_timeout_never_contributes_partial_labels():
    admit, _ = functions()
    assert admit(frame(),np.arange(5.),return_code=1,timed_out=False).empty
    assert admit(frame(),np.arange(5.),return_code=None,timed_out=True).empty


def test_duplicate_angles_count_once_and_conflicting_labels_are_not_averaged():
    admit, merge = functions()
    one = admit(frame(),np.arange(0,1.01,.25),return_code=0,timed_out=False)
    two = one.copy()
    two['source_segment'] = 'second'
    two.loc[two.alpha_deg.eq(.25),'cl'] += .1
    rows, audit = merge([one,two])
    assert rows.alpha_deg.tolist() == [0.,1.]
    assert audit['conflicting_alpha_count'] == 1
    assert audit['consistent_duplicate_rows_removed'] == 2
    assert audit['conflicting_alpha_values'] == [.25]
    assert rows.cl.tolist() == [0.,.4]


def test_empty_segment_is_recordable_without_fabricating_a_polar():
    admit, merge = functions()
    empty = admit(pd.DataFrame(),np.arange(3.),return_code=0,timed_out=False)
    rows, audit = merge([empty])
    assert rows.empty and audit['admitted_states'] == 0

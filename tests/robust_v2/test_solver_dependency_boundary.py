"""A solver-only process must not load the GPU training framework."""
import subprocess
import sys

from robust_airfoil import constants
from robust_airfoil.constants import ROOT


def test_real_xfoil_driver_import_does_not_load_torch():
    result = subprocess.run(
        [sys.executable,'-c',
         "import sys; from robust_airfoil.validation.xfoil_runner import XFoilCase, _run_xfoil_segment; assert 'torch' not in sys.modules, 'Solver-only path unexpectedly loads PyTorch'"],
        cwd=ROOT,capture_output=True,text=True,check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_feature_contract_is_available_without_importing_training_dataset():
    columns = getattr(constants,'FEATURE_COLUMNS',None)
    assert columns == [f'lower_weight_{i}' for i in range(5)] + [f'upper_weight_{i}' for i in range(5)] + ['leading_edge_weight','TE_thickness','alpha_deg']
    assert getattr(constants,'TARGET_COLUMNS',None) == ['cl','log_cd','cm']

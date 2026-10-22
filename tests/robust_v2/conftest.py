from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from robust_airfoil.modelling.dataset import FEATURE_COLUMNS


@pytest.fixture
def coordinate_text() -> str:
    return """demo
1.0 0.001
0.75 0.05
0.50 0.08
0.25 0.07
0.0 0.0
0.25 -0.03
0.50 -0.04
0.75 -0.02
1.0 -0.001
"""


@pytest.fixture
def polar_text() -> str:
    return """XFOIL polar
Mach = 0.0  Re = 1.000e6  Ncrit = 9.0
Alpha CL CD CDp CM Top_Xtr Bot_Xtr
----------------------------------
-2.0 -0.20 0.014 0.010 -0.02 0.8 0.7
 0.0  0.00 0.010 0.007 -0.03 0.7 0.6
 2.0  0.22 0.011 0.008 -0.04 0.6 0.5
 4.0  0.43 0.013 0.009 -0.05 0.5 0.4
"""


@pytest.fixture
def model_frame() -> pd.DataFrame:
    rows = []
    for airfoil_index, name in enumerate(("a", "b", "c", "d")):
        for alpha in np.linspace(-4, 8, 12):
            row: dict[str, object] = {column: 0.01 * (airfoil_index + offset) for offset, column in enumerate(FEATURE_COLUMNS[:-1])}
            row.update({
                "airfoil_id": name,
                "alpha_deg": float(alpha),
                "cl": float(0.105 * alpha + 0.02 * airfoil_index),
                "log_cd": float(np.log(0.01 + 0.0003 * alpha**2 + 0.0001 * airfoil_index)),
                "cm": float(-0.03 - 0.001 * alpha),
                "mask_cl": True,
                "mask_log_cd": True,
                "mask_cm": True,
                "airfoil_weight": 1 / 12,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-xfoil", action="store_true", default=False)

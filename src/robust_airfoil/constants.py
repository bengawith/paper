from pathlib import Path

PROJECT_SEED = 20260824
ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "robust_v2"
DATA_ROOT = ROOT / "data" / "robust_v2"
RESULTS_ROOT = ROOT / "results" / "robust_v2"
REPORTS_ROOT = ROOT / "reports" / "robust_v2"
RUN_STATE_PATH = REPORTS_ROOT / "run_state.json"
FROZEN_MANIFEST_PATH = RESULTS_ROOT / "FROZEN_MODEL_MANIFEST.json"


FEATURE_COLUMNS = [f"lower_weight_{i}" for i in range(5)] + [f"upper_weight_{i}" for i in range(5)] + ["leading_edge_weight", "TE_thickness", "alpha_deg"]
TARGET_COLUMNS = ["cl", "log_cd", "cm"]

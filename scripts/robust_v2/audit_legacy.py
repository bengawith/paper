from pathlib import Path

from robust_airfoil.data.legacy import write_legacy_audit

write_legacy_audit(Path("data/csv/dataset_12CST_params.csv"), Path("results/robust_v2/legacy/legacy_audit.json"))

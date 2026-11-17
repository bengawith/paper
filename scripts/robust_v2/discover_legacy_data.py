import argparse
from pathlib import Path

from robust_airfoil.recovery import discover_candidates, write_recovery_csv

parser = argparse.ArgumentParser()
parser.add_argument("--auto-roots", action="store_true")
args = parser.parse_args()
home = Path.home()
roots = [Path.cwd(), Path.cwd().parent, home / "Documents", home / "Downloads", home / "Desktop", home / "OneDrive"]
write_recovery_csv(Path("reports/robust_v2/data/local_recovery_candidates.csv"), discover_candidates(roots))

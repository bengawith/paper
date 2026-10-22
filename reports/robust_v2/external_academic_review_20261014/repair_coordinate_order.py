"""Repair demonstrated Selig/Lednicer surface membership; retain old evidence."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
path = ROOT/'src/robust_airfoil/geometry/normalise.py'
text = path.read_text(encoding='utf-8')
old = '''    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Expected Nx2 coordinate array")
    le_index = int(np.argmin(points[:, 0]))
    if le_index == 0 or le_index == len(points) - 1:
        midpoint = np.median(points[:, 1])
        upper_raw = points[(points[:, 1] >= midpoint)][np.argsort(points[(points[:, 1] >= midpoint), 0])]
        lower_raw = points[(points[:, 1] < midpoint)][np.argsort(points[(points[:, 1] < midpoint), 0])]
        if len(upper_raw) < 3 or len(lower_raw) < 3:
            raise ValueError("Unable to split upper and lower surfaces")
    else:
        first, second = points[: le_index + 1], points[le_index:]
        upper_raw, lower_raw = (first, second) if np.nanmean(first[:, 1]) >= np.nanmean(second[:, 1]) else (second, first)
'''
new = '''    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 5:
        raise ValueError("Expected at least five Nx2 coordinates")
    if not np.isfinite(points).all() or cosine_points < 3:
        raise ValueError("Finite coordinates and at least three cosine points are required")
    span = float(np.ptp(points[:, 0]))
    if span <= 0:
        raise ValueError("Invalid chord")
    le_index = int(np.argmin(points[:, 0]))
    jumps = np.flatnonzero(np.diff(points[:, 0]) < -0.5 * span)
    # Lednicer files contain two LE-to-TE blocks, separated by a TE-to-LE
    # jump. Preserve their membership even when both surfaces have positive y.
    if len(jumps) == 1 and 2 <= jumps[0] <= len(points) - 4:
        first, second = points[: jumps[0] + 1], points[jumps[0] + 1 :]
        for surface in (first, second):
            if (
                np.any(np.diff(surface[:, 0]) < -0.02 * span)
                or surface[0, 0] > points[le_index, 0] + 0.05 * span
                or surface[-1, 0] < points[:, 0].max() - 0.05 * span
            ):
                raise ValueError("Unsupported coordinate ordering")
    elif 1 < le_index < len(points) - 2:
        # Selig contours run TE-to-LE-to-TE (either surface may come first).
        first, second = points[: le_index + 1], points[le_index:]
        if np.any(np.diff(first[:, 0]) > 0.02 * span) or np.any(
            np.diff(second[:, 0]) < -0.02 * span
        ):
            raise ValueError("Unsupported coordinate ordering")
    else:
        raise ValueError("Unsupported coordinate ordering; no median-height reconstruction")
    upper_raw, lower_raw = (
        (first, second) if np.mean(first[:, 1]) >= np.mean(second[:, 1]) else (second, first)
    )
'''
if text.count(old)!=1: raise RuntimeError('Exact coordinate-order anchor is not unique')
before = hashlib.sha256(path.read_bytes()).hexdigest()
text = text.replace(old,new)
path.write_text(text,encoding='utf-8',newline='\n')
result = {'path':path.relative_to(ROOT).as_posix(),'before_sha256':before,'after_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scientific_effect':'Corrected normalisation changes affected model inputs and requires a new model/data lineage. No frozen data or model rewritten.'}
(Path(__file__).parent/'COORDINATE_ORDER_EDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))

from __future__ import annotations

import numpy as np


def geometry_metrics(upper: np.ndarray, lower: np.ndarray) -> dict[str, float]:
    x = upper[:, 0]
    thickness = upper[:, 1] - lower[:, 1]
    camber = (upper[:, 1] + lower[:, 1]) / 2
    near_le = max(2, min(8, len(x) // 10))
    radius_proxy = float(np.mean(thickness[1:near_le] ** 2 / np.maximum(x[1:near_le], 1e-9)))
    return {
        "maximum_thickness": float(np.max(thickness)),
        "maximum_thickness_x": float(x[int(np.argmax(thickness))]),
        "maximum_camber": float(np.max(np.abs(camber))),
        "section_area": float(np.trapezoid(thickness, x)),
        "leading_edge_radius_proxy": radius_proxy,
        "te_thickness": float(thickness[-1]),
    }

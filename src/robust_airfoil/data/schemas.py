from __future__ import annotations

AIRFOIL_COLUMNS = [
    "airfoil_id", "canonical_name", "geometry_source", "geometry_path", "geometry_sha256",
    "normalised_geometry_hash", "cluster_id", "cst_rms_error", "cst_max_error",
    "maximum_thickness", "section_area", "leading_edge_radius_proxy", "valid_geometry",
]
POLAR_POINT_COLUMNS = [
    "airfoil_id", "polar_id", "source_tier", "source_path", "source_sha256", "source_row",
    "reynolds_number", "mach", "ncrit", "alpha_deg", "cl", "cd", "cdp", "cm", "xtr_top", "xtr_bottom",
]

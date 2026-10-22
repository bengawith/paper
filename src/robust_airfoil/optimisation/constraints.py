from __future__ import annotations


def relative_geometry_constraints(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {
        "thickness_lower_margin": candidate["maximum_thickness"] / reference["maximum_thickness"] - 0.98,
        "thickness_upper_margin": 1.02 - candidate["maximum_thickness"] / reference["maximum_thickness"],
        "area_margin": candidate["section_area"] / reference["section_area"] - 0.98,
        "leading_edge_radius_margin": candidate["leading_edge_radius_proxy"] / reference["leading_edge_radius_proxy"] - 0.80,
    }


def constraints_feasible(margins: dict[str, float]) -> bool:
    return all(value >= 0 for value in margins.values())

"""Reviewer-directed adversarial surrogate-vs-XFOIL disagreement search.

Thin wrapper over the lineage-bound entry point so the search can be run either
through the public CLI (`robust-airfoil validate stress`) or directly.
"""

import json

from robust_airfoil.study import build_study_surrogate_stress

LINEAGE = "robust-v2-current-20260908"

if __name__ == "__main__":
    summary = build_study_surrogate_stress(LINEAGE, pool=48, refine_top=6, timeout_seconds=120)
    print(
        json.dumps(
            {
                "requested": summary["requested_candidates"],
                "evaluated": summary["evaluated_candidates"],
                "rejected_geometry": summary["rejected_geometry"],
                "xfoil_convergence": summary["xfoil_convergence"],
                "support_vs_gap_spearman": summary["support_distance_vs_cl_gap_spearman"],
                "mean_cl_gap_fully_trusted": summary["mean_cl_gap_fully_trusted"],
                "mean_cl_gap_outside_trust": summary["mean_cl_gap_outside_trust"],
                "worst_overall_cl_gap": (summary["worst_case_overall"] or {})
                .get("disagreement", {})
                .get("max_absolute_cl_gap"),
            },
            indent=2,
        )
    )

from __future__ import annotations

from collections.abc import Sequence


def ranking_consistent(surrogate_scores: Sequence[float], verification_scores: Sequence[float]) -> bool:
    if len(surrogate_scores) != len(verification_scores) or len(surrogate_scores) < 2:
        raise ValueError("Rankings require equal sequences with at least two values")
    surrogate_order = sorted(range(len(surrogate_scores)), key=lambda index: surrogate_scores[index])
    verification_order = sorted(range(len(verification_scores)), key=lambda index: verification_scores[index])
    return surrogate_order == verification_order

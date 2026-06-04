"""Research metrics for composition-invariance analysis."""

from __future__ import annotations


def compute_gap(reference: float, target: float) -> float:
    """Return reference - target."""

    return float(reference) - float(target)


def compute_cigg(mixed_map: float, complex_map: float) -> float:
    """Composition-Invariance Generalization Gap.

    CIGG = mixed_test_mAP50 - complex_holdout_mAP50.
    Lower is better when the mixed setting is the reference.
    """

    return compute_gap(mixed_map, complex_map)


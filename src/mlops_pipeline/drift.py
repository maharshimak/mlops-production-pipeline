"""Per-feature population stability (PSI) drift evaluation.

Bin edges are fitted on the *reference* sample only (equal-width between the
reference minimum and maximum); current values outside that range fall into the
outer bins. A constant reference feature uses three bins (below, equal, above)
so a shift away from the constant is still detected instead of collapsing into a
single bin.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from mlops_pipeline.core import psi


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    feature: str
    psi: float
    reference_counts: tuple[float, ...]
    current_counts: tuple[float, ...]


def _bin_counts(values: Sequence[float], minimum: float, maximum: float, bins: int) -> list[float]:
    if minimum == maximum:
        counts = [0.0, 0.0, 0.0]
        for value in values:
            counts[0 if value < minimum else 1 if value == minimum else 2] += 1.0
        return counts
    width = (maximum - minimum) / bins
    counts = [0.0] * bins
    for value in values:
        index = int((value - minimum) / width) if value > minimum else 0
        counts[max(0, min(bins - 1, index))] += 1.0
    return counts


def feature_drift(
    feature_names: Sequence[str],
    reference: Sequence[Sequence[float]],
    current: Sequence[Sequence[float]],
    *,
    bins: int = 5,
) -> tuple[FeatureDrift, ...]:
    if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= 100:
        raise ValueError("bins must be an integer between 2 and 100")
    if not reference or not current:
        raise ValueError("reference and current samples must be non-empty")
    width = len(feature_names)
    for row in (*reference, *current):
        if len(row) != width:
            raise ValueError("every row must have one value per feature")
        if not all(isfinite(value) for value in row):
            raise ValueError("drift inputs must be finite")
    results: list[FeatureDrift] = []
    for column, name in enumerate(feature_names):
        ref_values = [row[column] for row in reference]
        cur_values = [row[column] for row in current]
        minimum, maximum = min(ref_values), max(ref_values)
        ref_counts = _bin_counts(ref_values, minimum, maximum, bins)
        cur_counts = _bin_counts(cur_values, minimum, maximum, bins)
        results.append(
            FeatureDrift(
                feature=name,
                psi=psi(ref_counts, cur_counts),
                reference_counts=tuple(ref_counts),
                current_counts=tuple(cur_counts),
            )
        )
    return tuple(results)

"""Cluster permutation test and Holm-Bonferroni correction (item 9).

Implements metrics.md section 3 steps 4 and 6.

Paired cluster-permutation test (step 4):
- Null hypothesis: exchangeability of the two system labels within a
  query. Because queries are clustered within scenes, the labels are
  flipped PER SCENE (one epsilon_s in {-1, +1} per scene applied to
  every query in that scene), never per query, so within-scene
  dependence is respected.
- Implementation: a scene flip swaps the paired (a_i, b_i) values for
  every query i in that scene, then the statistic is recomputed on the
  reconstructed samples. For the mean-difference statistic this is
  exactly the classic sign flip of the paired differences; doing it as
  a swap keeps one uniform mechanism that is also correct for the
  median-difference statistic T = med(a) - med(b), which does not
  decompose into per-query differences.
- Two-sided p = (1 + #{|T*| >= |T_obs|}) / (P + 1), P = 10000 default.
  The comparison uses a 1e-12 absolute tolerance so exact ties (for
  example, all differences zero) are counted as extreme rather than
  lost to float noise.

Holm-Bonferroni (step 6): step-down over the preregistered family,
K = 5. Adjusted p-values with monotonicity enforced:
p_adj_(i) = max_{j<=i} min(1, (K - j + 1) * p_(j)). The family size can
be held at the preregistered K even when fewer p-values are supplied
(missing comparisons); that choice is conservative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Absolute tolerance for the |T*| >= |T_obs| comparison.
_TIE_TOL = 1e-12

DEFAULT_N_PERMUTATIONS = 10000


def _mean_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a) - np.mean(b))


def _median_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.median(a) - np.median(b))


_STATISTICS = {
    "mean_diff": _mean_diff,
    "median_diff": _median_diff,
}


@dataclass(frozen=True)
class PermutationResult:
    """Observed statistic and its cluster-permutation p-value."""

    t_obs: float
    p_value: float
    n_perm: int
    n_scenes: int
    n_queries: int
    statistic: str


def cluster_permutation_test(
    values_a: Sequence[float],
    values_b: Sequence[float],
    scenes: Sequence[str],
    statistic: str = "mean_diff",
    n_perm: int = DEFAULT_N_PERMUTATIONS,
    rng: Optional[np.random.Generator] = None,
) -> PermutationResult:
    """Paired cluster-permutation test for a difference between systems.

    values_a, values_b: per-query aggregated values (one entry per
    query, aligned), e.g. per-query success fractions s_i or per-query
    mean censored errors x_i. scenes: the scene of each query, same
    length; label flips happen at scene level. statistic: "mean_diff"
    (difference in means, used for SR) or "median_diff" (difference in
    medians, used for the median-error comparison). rng must be an
    explicitly seeded numpy Generator.
    """
    if rng is None:
        raise ValueError(
            "cluster_permutation_test requires an explicitly seeded numpy "
            "Generator; implicit seeding would break determinism"
        )
    if statistic not in _STATISTICS:
        known = ", ".join(sorted(_STATISTICS))
        raise ValueError(f"unknown statistic {statistic!r}; known: {known}")
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("values_a and values_b must be 1D and the same length")
    if len(scenes) != a.shape[0]:
        raise ValueError("scenes must align with the per-query values")
    if a.shape[0] == 0:
        raise ValueError("no queries to test")

    stat = _STATISTICS[statistic]
    scene_keys = sorted(set(scenes))
    key_to_idx = {k: i for i, k in enumerate(scene_keys)}
    scene_idx = np.asarray([key_to_idx[s] for s in scenes], dtype=int)
    n_scenes = len(scene_keys)

    t_obs = stat(a, b)
    threshold = abs(t_obs) - _TIE_TOL
    count = 0
    for _ in range(int(n_perm)):
        flips = rng.integers(0, 2, size=n_scenes).astype(bool)
        flip_q = flips[scene_idx]
        a_perm = np.where(flip_q, b, a)
        b_perm = np.where(flip_q, a, b)
        if abs(stat(a_perm, b_perm)) >= threshold:
            count += 1
    p = (1.0 + count) / (float(n_perm) + 1.0)
    return PermutationResult(
        t_obs=float(t_obs), p_value=float(p), n_perm=int(n_perm),
        n_scenes=n_scenes, n_queries=int(a.shape[0]), statistic=statistic,
    )


def holm_bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
    family_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Holm-Bonferroni step-down over a preregistered family.

    p_values: raw p-values in any order; results map back to the input
    order. family_size: the preregistered K (defaults to len(p_values);
    must be >= len(p_values)). Holding K at the preregistered 5 when
    fewer comparisons could be evaluated is conservative.

    Returns {"adjusted": [...], "reject": [...], "alpha": alpha,
    "family_size": K}. Adjusted p-values enforce monotonicity:
    p_adj_(i) = max_{j<=i} min(1, (K - j + 1) * p_(j)). Rejections
    follow the step-down rule: walking p ascending, reject while
    p_(j) <= alpha / (K - j + 1), stop at the first non-rejection.
    """
    p = [float(x) for x in p_values]
    for x in p:
        if not (0.0 <= x <= 1.0):
            raise ValueError(f"p-value out of [0, 1]: {x}")
    m = len(p)
    k = int(family_size) if family_size is not None else m
    if k < m:
        raise ValueError(f"family_size {k} smaller than the {m} p-values given")

    # Sort ascending; ties broken by input position for determinism.
    order = sorted(range(m), key=lambda i: (p[i], i))

    adjusted: List[float] = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        multiplier = k - rank  # (K - j + 1) with j = rank + 1
        running = max(running, min(1.0, multiplier * p[i]))
        adjusted[i] = running

    reject = [False] * m
    for rank, i in enumerate(order):
        if p[i] <= alpha / (k - rank):
            reject[i] = True
        else:
            break

    return {"adjusted": adjusted, "reject": reject, "alpha": float(alpha), "family_size": k}

"""Hierarchical BCa bootstrap over scene clusters (item 9).

Implements the preregistered CI procedure from research/metrics.md
section 3 step 3:

- B resamples (default 10000). Each resample draws the scenes with
  replacement, then draws queries with replacement WITHIN each drawn
  scene, and recomputes the statistic on the resampled set. This
  respects the clustering of queries within scenes (2 to 4 queries per
  scene over 41 scenes on bench_v1_98); a flat query-level bootstrap
  would understate the CI width.
- Interval: BCa (bias corrected and accelerated), 95 percent by
  default. The bias correction z0 comes from the position of the point
  estimate in the bootstrap distribution; the acceleration a comes
  from a jackknife over SCENES (delete one scene at a time), the
  correct jackknife unit for clustered data.
- Percentile fallback: if any BCa ingredient degenerates the interval
  falls back to the plain percentile interval and the result records
  the reason. Degenerate cases handled:
    * the bootstrap distribution sits entirely on one side of the
      point estimate (bias correction undefined),
    * fewer than 2 scenes (jackknife undefined),
    * zero-variance jackknife, e.g. constant data (acceleration
      undefined),
    * the BCa-adjusted quantile levels leave (0, 1).

The RNG is always an explicitly passed, seeded numpy Generator;
nothing in this module seeds itself. The normal cdf and inverse cdf
come from the stdlib statistics.NormalDist, so scipy stays optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_NORM = NormalDist()

# Absolute tolerance below which the jackknife spread is treated as
# zero (degenerate acceleration).
_JACK_VAR_TOL = 1e-12

Statistic = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate with its bootstrap interval and provenance."""

    point: float
    lo: float
    hi: float
    method: str  # "bca" or "percentile"
    fallback_reason: Optional[str]
    z0: Optional[float]
    accel: Optional[float]
    n_boot: int
    n_scenes: int
    n_queries: int


def _scene_arrays(scene_values: Dict[str, np.ndarray]) -> Tuple[List[str], List[np.ndarray]]:
    """Sorted scene keys and their per-query value arrays as 2D floats."""
    keys = sorted(scene_values)
    if not keys:
        raise ValueError("scene_values is empty: nothing to bootstrap")
    arrays = []
    for k in keys:
        arr = np.asarray(scene_values[k], dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2 or arr.shape[0] == 0:
            raise ValueError(f"scene {k!r} must contribute at least one query row")
        arrays.append(arr)
    return keys, arrays


def bootstrap_distribution(
    arrays: Sequence[np.ndarray],
    statistic: Statistic,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """B hierarchical resample statistics: scenes, then queries within.

    Each resample draws len(arrays) scenes with replacement; each drawn
    scene contributes a with-replacement resample of its own queries,
    of its own size. The statistic sees the concatenated rows.
    """
    n_scenes = len(arrays)
    out = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        scene_idx = rng.integers(0, n_scenes, size=n_scenes)
        parts = []
        for si in scene_idx:
            arr = arrays[si]
            n = arr.shape[0]
            take = rng.integers(0, n, size=n)
            parts.append(arr[take])
        out[b] = statistic(np.concatenate(parts, axis=0))
    return out


def jackknife_over_scenes(arrays: Sequence[np.ndarray], statistic: Statistic) -> np.ndarray:
    """Delete-one-SCENE jackknife estimates of the statistic.

    The jackknife unit is the scene, not the query, because scenes are
    the independent sampling unit in the hierarchical design. Used for
    the BCa acceleration constant.
    """
    n_scenes = len(arrays)
    if n_scenes < 2:
        raise ValueError("jackknife over scenes needs at least 2 scenes")
    est = np.empty(n_scenes, dtype=float)
    for i in range(n_scenes):
        rest = [arrays[j] for j in range(n_scenes) if j != i]
        est[i] = statistic(np.concatenate(rest, axis=0))
    return est


def bca_interval(
    scene_values: Dict[str, np.ndarray],
    statistic: Statistic,
    n_boot: int = 10000,
    rng: Optional[np.random.Generator] = None,
    alpha: float = 0.05,
) -> BootstrapResult:
    """Hierarchical BCa interval for statistic over clustered queries.

    scene_values: {scene: array of per-query values}, 1D per scene or
    2D (n_queries, k) for paired statistics. statistic maps the
    concatenated (n, k) array to a float. rng must be an explicitly
    seeded numpy Generator (metrics.md pins default_rng(20260815) for
    the published analysis).

    Returns a BootstrapResult whose method field says whether the real
    BCa interval was produced or the documented percentile fallback
    was used (fallback_reason says why).
    """
    if rng is None:
        raise ValueError(
            "bca_interval requires an explicitly seeded numpy Generator; "
            "implicit seeding would break the determinism guarantee"
        )
    _, arrays = _scene_arrays(scene_values)
    all_rows = np.concatenate(arrays, axis=0)
    point = float(statistic(all_rows))
    boots = bootstrap_distribution(arrays, statistic, n_boot, rng)

    def percentile(reason: str) -> BootstrapResult:
        lo = float(np.quantile(boots, alpha / 2.0))
        hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
        return BootstrapResult(
            point=point, lo=lo, hi=hi, method="percentile",
            fallback_reason=reason, z0=None, accel=None,
            n_boot=int(n_boot), n_scenes=len(arrays),
            n_queries=int(all_rows.shape[0]),
        )

    # Bias correction. Ties with the point estimate count half, which
    # stabilizes z0 for discrete statistics (SR takes few values).
    n_less = int(np.sum(boots < point))
    n_eq = int(np.sum(boots == point))
    p0 = (n_less + 0.5 * n_eq) / float(n_boot)
    if not (0.0 < p0 < 1.0):
        return percentile(
            "bias correction degenerate: bootstrap distribution entirely "
            "on one side of the point estimate"
        )
    z0 = _NORM.inv_cdf(p0)

    # Acceleration from the delete-one-scene jackknife.
    if len(arrays) < 2:
        return percentile("fewer than 2 scenes: jackknife acceleration undefined")
    jack = jackknife_over_scenes(arrays, statistic)
    d = jack.mean() - jack
    ss = float(np.sum(d * d))
    if ss <= _JACK_VAR_TOL:
        return percentile("zero-variance jackknife: acceleration undefined")
    accel = float(np.sum(d ** 3) / (6.0 * ss ** 1.5))

    # BCa-adjusted quantile levels for both interval ends.
    levels = []
    for q in (alpha / 2.0, 1.0 - alpha / 2.0):
        z = _NORM.inv_cdf(q)
        den = 1.0 - accel * (z0 + z)
        if den <= 0.0:
            return percentile("BCa quantile adjustment left (0, 1)")
        levels.append(_NORM.cdf(z0 + (z0 + z) / den))
    lo_level, hi_level = levels
    if not (0.0 < lo_level < hi_level < 1.0):
        return percentile("BCa quantile adjustment left (0, 1)")

    lo = float(np.quantile(boots, lo_level))
    hi = float(np.quantile(boots, hi_level))
    return BootstrapResult(
        point=point, lo=lo, hi=hi, method="bca", fallback_reason=None,
        z0=float(z0), accel=accel, n_boot=int(n_boot),
        n_scenes=len(arrays), n_queries=int(all_rows.shape[0]),
    )

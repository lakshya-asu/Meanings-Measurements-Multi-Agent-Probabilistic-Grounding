# Navmesh-masked, renormalized MSP density (MAPG-12).

"""Mask the composed MSP density to navigable free space and renormalize.

This module makes the paper's "normalized over Omega_free" sentence true:
the unnormalized composed log density from ``src.msp.pdf.combined_logpdf``
is restricted to the navigable cells of a navmesh and renormalized with a
discrete partition function, so integrated quantities (posterior mass,
categorical sampling, calibration confidences) become meaningful.

Discrete measure
----------------
The navigable free space Omega_free is discretized as a regular grid of
square cells of side ``cell_size_m`` in the horizontal x-z plane (Habitat
world frame, y up), each cell represented by one navigable point at floor
height. With l_i = combined_logpdf(cell_i) the partition function is the
Riemann sum over navigable cells

    Z = sum_i exp(l_i) * A_cell,        A_cell = cell_size_m ** 2
    log_Z = log(A_cell) + logsumexp_i(l_i)

and the probability mass of cell i is

    p_i = exp(l_i) * A_cell / Z = softmax(l)_i

(the cell area cancels in p_i but not in log_Z, which normalizes density
values). This is exact conditioning of the product-of-experts density on
the event x in Omega_free, up to the grid discretization: one indicator
expert joins the product, and Z is one logsumexp over roughly 10^3..10^4
cells, evaluated in a single numpy pass.

``mass_in_ball(center, radius)`` returns sum of p_i over cells whose
(snapped, floor-height) centers lie within ``radius`` meters (Euclidean)
of ``center``. Called with the argmax cell and tau it yields the
tau-ball posterior mass, the confidence scalar the calibration suite
(ECE / Brier / risk-coverage) consumes downstream; MAPG-12 records it in
pdf_params as ``mass_in_tau_ball`` with tau = 1.0 m.

Pathfinder duck-type (injected; no habitat import at module level)
------------------------------------------------------------------
Any object with

    get_bounds() -> (lower, upper)   two 3-sequences, Habitat frame
    snap_point(p) -> 3-sequence      nearest navmesh point, or NaNs /
                                     None when the snap fails

works; ``habitat_sim.PathFinder`` satisfies both (same access pattern as
the injected snap function in src/verification/checks.py). A grid cell is
navigable iff its center, probed at mid-bounds height, snaps to a navmesh
point whose horizontal displacement is within ``snap_tolerance_m``
(default half a cell). Host tests inject a fake with a rectangular
walkable region.
"""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.msp.pdf import combined_logpdf

DEFAULT_CELL_SIZE_M = 0.25
DEFAULT_TAU_M = 1.0
MASKING_MODES = ("navmesh", "off")


def resolve_masking_mode(value: Any) -> Tuple[str, Optional[str]]:
    """Normalize cfg ``density_masking`` to a valid mode.

    Returns (mode, warning). Unknown values fall back to the default
    "navmesh" with a warning string; None means "not configured" and
    resolves to the default silently.
    """
    if value is None:
        return "navmesh", None
    mode = str(value).lower().strip()
    if mode in MASKING_MODES:
        return mode, None
    return "navmesh", (
        "unknown density_masking=%r; using 'navmesh' (valid: %s)"
        % (value, ", ".join(MASKING_MODES))
    )


def _logsumexp(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    m = float(np.max(a))
    if not math.isfinite(m):
        return m
    return m + math.log(float(np.sum(np.exp(a - m))))


class NavmeshGrid:
    """Regular-grid discretization of the navigable free space.

    Geometry (which cells are navigable, and their snapped floor-height
    centers) is built once in the constructor from the injected
    pathfinder; it is independent of the kernel parameters, so one grid
    can be reused across steps of an episode. ``evaluate(params)`` then
    scores the composed density on the cells and computes log_Z; the
    query methods (``normalized_logpdf``, ``argmax_cell``, ``sample``,
    ``mass_in_ball``) require a prior ``evaluate``.
    """

    def __init__(
        self,
        pathfinder: Any,
        cell_size_m: float = DEFAULT_CELL_SIZE_M,
        snap_tolerance_m: Optional[float] = None,
    ) -> None:
        cell = float(cell_size_m)
        if not math.isfinite(cell) or cell <= 0.0:
            raise ValueError("cell_size_m must be a positive finite number, got %r" % cell_size_m)
        self.cell_size_m = cell
        self.cell_area_m2 = cell * cell
        self.snap_tolerance_m = (
            float(snap_tolerance_m) if snap_tolerance_m is not None else 0.5 * cell
        )

        lo_raw, hi_raw = pathfinder.get_bounds()
        lo = np.asarray([float(v) for v in list(lo_raw)[:3]], dtype=np.float64)
        hi = np.asarray([float(v) for v in list(hi_raw)[:3]], dtype=np.float64)
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            raise ValueError("pathfinder bounds are not finite: %r, %r" % (lo_raw, hi_raw))
        if hi[0] <= lo[0] or hi[2] <= lo[2]:
            raise ValueError("degenerate pathfinder bounds: %r, %r" % (lo_raw, hi_raw))

        self._origin_x = float(lo[0])
        self._origin_z = float(lo[2])
        self._nx = max(1, int(math.ceil((hi[0] - lo[0]) / cell)))
        self._nz = max(1, int(math.ceil((hi[2] - lo[2]) / cell)))
        y_probe = 0.5 * (float(lo[1]) + float(hi[1]))

        nav_mask = np.zeros((self._nx, self._nz), dtype=bool)
        centers = []
        for ix in range(self._nx):
            x = self._origin_x + (ix + 0.5) * cell
            for iz in range(self._nz):
                z = self._origin_z + (iz + 0.5) * cell
                snapped = self._try_snap(pathfinder, x, y_probe, z)
                if snapped is None:
                    continue
                sx, sy, sz = snapped
                if math.hypot(sx - x, sz - z) > self.snap_tolerance_m:
                    continue
                nav_mask[ix, iz] = True
                centers.append((sx, sy, sz))
        if not centers:
            raise ValueError(
                "no navigable cells found (bounds %r..%r, cell %.3f m)" % (lo_raw, hi_raw, cell)
            )
        self._nav_mask = nav_mask
        self.cell_centers = np.asarray(centers, dtype=np.float64)

        self._params: Optional[Dict[str, Any]] = None
        self._cell_logpdf: Optional[np.ndarray] = None
        self._log_prob: Optional[np.ndarray] = None
        self._log_Z: Optional[float] = None

    @staticmethod
    def _try_snap(pathfinder: Any, x: float, y: float, z: float):
        try:
            s = pathfinder.snap_point([x, y, z])
        except Exception:
            return None
        if s is None:
            return None
        try:
            xyz = [float(s[0]), float(s[1]), float(s[2])]
        except (TypeError, ValueError, IndexError):
            return None
        if not all(math.isfinite(v) for v in xyz):
            return None
        return xyz

    @property
    def num_cells(self) -> int:
        return int(self.cell_centers.shape[0])

    @property
    def log_Z(self) -> float:
        self._require_evaluated()
        return float(self._log_Z)

    def evaluate(self, params: Dict[str, Any]) -> "NavmeshGrid":
        """Score combined_logpdf on the navigable cells; compute log_Z.

        log_Z = log(A_cell) + logsumexp_i(l_i), the discrete measure
        documented in the module docstring. Returns self for chaining.
        """
        c = self.cell_centers
        logp = np.asarray(
            combined_logpdf(c[:, 0], c[:, 1], c[:, 2], params), dtype=np.float64
        )
        lse = _logsumexp(logp)
        self._params = dict(params)
        self._cell_logpdf = logp
        self._log_prob = logp - lse
        self._log_Z = float(lse + math.log(self.cell_area_m2))
        return self

    def _require_evaluated(self) -> None:
        if self._cell_logpdf is None:
            raise RuntimeError("NavmeshGrid.evaluate(params) must be called first")

    def normalized_logpdf(self, points) -> np.ndarray:
        """Masked normalized log density at arbitrary points.

        Returns combined_logpdf(p) - log_Z for points falling in a
        navigable cell (x-z membership; y is ignored for masking, since
        the grid is a floor-height discretization), and -inf elsewhere.
        Accepts a single (3,) point or an (N, 3) array.
        """
        self._require_evaluated()
        pts = np.asarray(points, dtype=np.float64)
        single = pts.ndim == 1
        if single:
            pts = pts[None, :]
        logp = np.asarray(
            combined_logpdf(pts[:, 0], pts[:, 1], pts[:, 2], self._params),
            dtype=np.float64,
        )
        out = logp - float(self._log_Z)
        ix = np.floor((pts[:, 0] - self._origin_x) / self.cell_size_m).astype(int)
        iz = np.floor((pts[:, 2] - self._origin_z) / self.cell_size_m).astype(int)
        inside = (ix >= 0) & (ix < self._nx) & (iz >= 0) & (iz < self._nz)
        navigable = np.zeros(pts.shape[0], dtype=bool)
        navigable[inside] = self._nav_mask[ix[inside], iz[inside]]
        out[~navigable] = -np.inf
        return out[0] if single else out

    def argmax_cell(self) -> Dict[str, Any]:
        """The maximum-density navigable cell (the masked MAP estimate)."""
        self._require_evaluated()
        i = int(np.argmax(self._cell_logpdf))
        return {
            "index": i,
            "xyz": [float(v) for v in self.cell_centers[i]],
            "logpdf": float(self._cell_logpdf[i]),
            "normalized_logpdf": float(self._cell_logpdf[i] - self._log_Z),
            "prob": float(np.exp(self._log_prob[i])),
        }

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n cell centers from the categorical distribution p_i."""
        self._require_evaluated()
        p = np.exp(self._log_prob)
        p = p / float(np.sum(p))
        idx = rng.choice(self.num_cells, size=int(n), p=p)
        return self.cell_centers[idx].copy()

    def mass_in_ball(self, center, radius_m: float) -> float:
        """Posterior mass within radius_m of center (Euclidean, 3D).

        Distances are measured to the snapped floor-height cell centers;
        with ``center`` itself a cell center (the argmax) this is the
        tau-ball confidence used by the calibration suite.
        """
        self._require_evaluated()
        c = np.asarray([float(v) for v in list(center)[:3]], dtype=np.float64)
        d = np.linalg.norm(self.cell_centers - c[None, :], axis=1)
        return float(np.sum(np.exp(self._log_prob)[d <= float(radius_m)]))


def apply_density_masking(
    *,
    mode: str,
    pathfinder: Any,
    pdf_params: Dict[str, Any],
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
    tau_m: float = DEFAULT_TAU_M,
    grid: Optional[NavmeshGrid] = None,
):
    """Planner-facing wrapper: masked argmax point estimate, or fallback.

    Returns (point_xyz, record, grid):

    - point_xyz: float32 (3,) array, the argmax navigable cell center,
      or None when masking did not run (mode off, no usable pathfinder,
      or a failure); the caller then keeps its unmasked estimate.
    - record: pdf_params-ready provenance dict. Always contains
      density_masking (mode), density_masked (bool), log_Z,
      mass_in_tau_ball, tau_m; density_masking_reason is set whenever
      density_masked is False (fallback semantics: off = old behavior,
      recorded, never an exception).
    - grid: the NavmeshGrid used (pass it back in to reuse geometry
      across steps), or None when none was built.

    mode "off" short-circuits before any pathfinder access, so the off
    arm of the ablation is byte-identical to the pre-MAPG-12 estimator.
    """
    record: Dict[str, Any] = {
        "density_masking": str(mode),
        "density_masked": False,
        "log_Z": None,
        "mass_in_tau_ball": None,
        "tau_m": float(tau_m),
        "density_masking_reason": None,
    }
    if mode == "off":
        record["density_masking_reason"] = "density_masking=off (cfg); unmasked estimator kept"
        return None, record, grid

    if grid is None:
        if (
            pathfinder is None
            or not hasattr(pathfinder, "get_bounds")
            or not hasattr(pathfinder, "snap_point")
        ):
            record["density_masking_reason"] = (
                "no usable pathfinder reachable (need get_bounds + snap_point); "
                "fell back to the unmasked estimator"
            )
            return None, record, None
        try:
            grid = NavmeshGrid(pathfinder, cell_size_m=cell_size_m)
        except Exception as e:
            record["density_masking_reason"] = (
                "navmesh grid build failed: %s; fell back to the unmasked estimator" % e
            )
            return None, record, None

    try:
        grid.evaluate(pdf_params)
        am = grid.argmax_cell()
        mass = grid.mass_in_ball(am["xyz"], tau_m)
    except Exception as e:
        record["density_masking_reason"] = (
            "masked evaluation failed: %s; fell back to the unmasked estimator" % e
        )
        return None, record, grid

    record.update(
        density_masked=True,
        log_Z=float(grid.log_Z),
        mass_in_tau_ball=float(mass),
        argmax_cell_index=int(am["index"]),
        argmax_logpdf=float(am["logpdf"]),
        num_navigable_cells=int(grid.num_cells),
        navmesh_cell_size_m=float(grid.cell_size_m),
    )
    return np.asarray(am["xyz"], dtype=np.float32), record, grid

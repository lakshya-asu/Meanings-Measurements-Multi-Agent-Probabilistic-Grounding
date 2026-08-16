# MAPG-12: navmesh masking + renormalization, host-runnable (numpy only).

"""Tests for src.msp.navmesh_density against a fake rectangular navmesh.

The fake pathfinder implements the injected duck-type (get_bounds +
snap_point) with a rectangular walkable region: snap clamps the probe
point into the rectangle at floor height, exactly like habitat's
snap_point returns the nearest navmesh point. Tests pass a tight
snap_tolerance_m so cell navigability is exactly "cell center inside
the rectangle" and the hand-checked counts below are unambiguous.
"""

import math

import numpy as np
import pytest

from src.msp.navmesh_density import (
    DEFAULT_TAU_M,
    NavmeshGrid,
    apply_density_masking,
    resolve_masking_mode,
)


class FakeRectNavmesh:
    """Rectangular walkable region [x0, x1] x [z0, z1] at floor_y."""

    def __init__(self, x0, x1, z0, z1, floor_y=0.0, bounds_pad=0.0):
        self.x0, self.x1 = float(x0), float(x1)
        self.z0, self.z1 = float(z0), float(z1)
        self.floor_y = float(floor_y)
        self.bounds_pad = float(bounds_pad)

    def get_bounds(self):
        pad = self.bounds_pad
        return (
            [self.x0 - pad, self.floor_y - 0.5, self.z0 - pad],
            [self.x1 + pad, self.floor_y + 2.5, self.z1 + pad],
        )

    def snap_point(self, p):
        x = min(max(float(p[0]), self.x0), self.x1)
        z = min(max(float(p[2]), self.z0), self.z1)
        return [x, self.floor_y, z]


class ExplodingPathfinder:
    """Any navmesh access is a test failure (for the off-mode bypass test)."""

    def get_bounds(self):
        raise AssertionError("pathfinder touched in off mode")

    def snap_point(self, p):
        raise AssertionError("pathfinder touched in off mode")


def make_params(**overrides):
    """combined_logpdf params, all kernels flat unless overridden."""
    base = {
        "mu_x": 0.0, "mu_y": 0.0, "mu_z": 0.0, "sigma_s": 1.0e6,
        "x0": 0.0, "y0": 0.0, "z0": 0.0, "d0": 0.0, "sigma_m": 1.0e6,
        "theta0": 0.0, "phi0": math.pi / 2.0, "kappa": 0.0,
    }
    base.update(overrides)
    return base


def peaked_params(mu, sigma=0.4):
    """Semantic Gaussian peaked at mu, other kernels flat."""
    return make_params(
        mu_x=float(mu[0]), mu_y=float(mu[1]), mu_z=float(mu[2]), sigma_s=float(sigma)
    )


def test_normalization_sums_to_one_over_cells():
    pf = FakeRectNavmesh(0.0, 2.0, 0.0, 2.0, bounds_pad=1.0)
    grid = NavmeshGrid(pf, cell_size_m=0.25, snap_tolerance_m=0.01)
    # 2 m / 0.25 m = 8 cells per axis inside the rectangle.
    assert grid.num_cells == 64
    grid.evaluate(peaked_params([1.0, 0.0, 1.0], sigma=0.5))

    # Discrete measure: sum over cells of exp(normalized logpdf) * A = 1.
    norm_lp = grid.normalized_logpdf(grid.cell_centers)
    total_mass = float(np.sum(np.exp(norm_lp)) * grid.cell_area_m2)
    assert total_mass == pytest.approx(1.0, abs=1e-9)

    # log_Z is log(cell area) plus the logsumexp of the raw cell values.
    from src.msp.pdf import combined_logpdf
    raw = combined_logpdf(
        grid.cell_centers[:, 0], grid.cell_centers[:, 1], grid.cell_centers[:, 2],
        grid._params,
    )
    m = float(np.max(raw))
    lse = m + math.log(float(np.sum(np.exp(raw - m))))
    assert grid.log_Z == pytest.approx(lse + math.log(0.25 * 0.25), abs=1e-9)


def test_masking_excludes_non_navigable_peak():
    # Density mean far outside the walkable rectangle: unmasked argmax
    # would sit at the mean; the masked argmax must be navigable.
    pf = FakeRectNavmesh(0.0, 2.0, 0.0, 2.0, bounds_pad=1.0)
    grid = NavmeshGrid(pf, cell_size_m=0.25, snap_tolerance_m=0.01)
    grid.evaluate(peaked_params([5.0, 0.0, 5.0], sigma=1.0))

    am = grid.argmax_cell()
    x, _, z = am["xyz"]
    assert 0.0 <= x <= 2.0 and 0.0 <= z <= 2.0
    # The peak pulls toward the near corner of the rectangle.
    assert x == pytest.approx(2.0 - 0.125, abs=1e-9)
    assert z == pytest.approx(2.0 - 0.125, abs=1e-9)

    # Points off the navmesh have -inf normalized log density, even at
    # the (non-navigable) mean where the raw density peaks.
    assert grid.normalized_logpdf(np.array([5.0, 0.0, 5.0])) == -np.inf
    assert np.isfinite(grid.normalized_logpdf(np.array(am["xyz"])))


def test_mass_in_ball_hand_checked_uniform():
    # 1 m x 1 m rectangle, 0.25 m cells: 16 cells, uniform density, so
    # each cell carries mass 1/16. Centers at 0.125, 0.375, 0.625, 0.875.
    pf = FakeRectNavmesh(0.0, 1.0, 0.0, 1.0, bounds_pad=0.5)
    grid = NavmeshGrid(pf, cell_size_m=0.25, snap_tolerance_m=0.01)
    assert grid.num_cells == 16
    grid.evaluate(make_params())

    probs = np.exp(grid._log_prob)
    assert np.allclose(probs, 1.0 / 16.0, atol=1e-12)

    # Ball at the grid crossing (0.5, 0, 0.5): the 4 nearest centers are
    # at distance sqrt(2) * 0.125 = 0.177, the next ring at 0.395.
    center = [0.5, 0.0, 0.5]
    assert grid.mass_in_ball(center, 0.3) == pytest.approx(4.0 / 16.0, abs=1e-12)
    # Radius 0.4 adds the 8 cells at distance 0.395: 12 cells total.
    assert grid.mass_in_ball(center, 0.4) == pytest.approx(12.0 / 16.0, abs=1e-12)
    # A ball covering everything has mass exactly 1.
    assert grid.mass_in_ball(center, 10.0) == pytest.approx(1.0, abs=1e-12)


def test_sample_is_deterministic_with_seeded_rng():
    pf = FakeRectNavmesh(0.0, 2.0, 0.0, 2.0, bounds_pad=1.0)
    grid = NavmeshGrid(pf, cell_size_m=0.25, snap_tolerance_m=0.01)
    grid.evaluate(peaked_params([1.0, 0.0, 1.0], sigma=0.3))

    s1 = grid.sample(128, np.random.default_rng(7))
    s2 = grid.sample(128, np.random.default_rng(7))
    assert np.array_equal(s1, s2)

    # Samples are cell centers (categorical over cells, no jitter).
    centers = {tuple(np.round(c, 9)) for c in grid.cell_centers}
    assert all(tuple(np.round(s, 9)) in centers for s in s1)

    s3 = grid.sample(128, np.random.default_rng(8))
    assert not np.array_equal(s1, s3)


def test_apply_density_masking_masked_path():
    pf = FakeRectNavmesh(0.0, 2.0, 0.0, 2.0, bounds_pad=1.0)
    xyz, record, grid = apply_density_masking(
        mode="navmesh",
        pathfinder=pf,
        pdf_params=peaked_params([5.0, 0.0, 5.0], sigma=1.0),
        cell_size_m=0.25,
        tau_m=DEFAULT_TAU_M,
    )
    assert record["density_masked"] is True
    assert record["density_masking_reason"] is None
    assert xyz is not None and xyz.shape == (3,)
    assert 0.0 <= float(xyz[0]) <= 2.0 and 0.0 <= float(xyz[2]) <= 2.0
    assert math.isfinite(record["log_Z"])
    assert 0.0 < record["mass_in_tau_ball"] <= 1.0
    assert record["tau_m"] == 1.0
    assert isinstance(grid, NavmeshGrid)

    # The returned grid is reusable: second call skips the rebuild.
    xyz2, record2, grid2 = apply_density_masking(
        mode="navmesh",
        pathfinder=None,  # would fail without the cached grid
        pdf_params=peaked_params([0.5, 0.0, 0.5], sigma=0.3),
        grid=grid,
    )
    assert grid2 is grid
    assert record2["density_masked"] is True
    assert xyz2 is not None


def test_off_mode_bypasses_everything():
    xyz, record, grid = apply_density_masking(
        mode="off",
        pathfinder=ExplodingPathfinder(),  # any touch raises
        pdf_params=make_params(),
    )
    assert xyz is None
    assert grid is None
    assert record["density_masked"] is False
    assert record["log_Z"] is None
    assert record["mass_in_tau_ball"] is None
    assert "off" in record["density_masking_reason"]


def test_no_pathfinder_falls_back_with_reason():
    xyz, record, grid = apply_density_masking(
        mode="navmesh",
        pathfinder=None,
        pdf_params=make_params(),
    )
    assert xyz is None and grid is None
    assert record["density_masked"] is False
    assert "no usable pathfinder" in record["density_masking_reason"]

    # An object without the duck-type methods is not usable either.
    xyz, record, _ = apply_density_masking(
        mode="navmesh",
        pathfinder=object(),
        pdf_params=make_params(),
    )
    assert xyz is None
    assert "no usable pathfinder" in record["density_masking_reason"]


def test_resolve_masking_mode():
    assert resolve_masking_mode(None) == ("navmesh", None)
    assert resolve_masking_mode("navmesh") == ("navmesh", None)
    assert resolve_masking_mode("off") == ("off", None)
    assert resolve_masking_mode(" OFF ") == ("off", None)
    mode, warn = resolve_masking_mode("banana")
    assert mode == "navmesh"
    assert warn and "banana" in warn

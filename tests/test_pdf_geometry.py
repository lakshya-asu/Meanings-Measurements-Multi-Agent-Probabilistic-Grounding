"""Geometry tests for the directional kernel in src/msp/pdf.py (P0 fix).

Verifies that the vMF directional exponent uses the Habitat y-up
convention (yaw in the x-z plane from +x toward +z, polar angle from
+y), consistent by construction with src/evals/angular.py, and that the
specific failures of the old z-polar spherical expansion are gone:
x-axis yaw bias, vanishing weight for goals along world z, and
above/below queries scored on a horizontal axis.

All expected values are hand computed. numpy is only needed because
pdf.py itself uses it; the checks are stdlib math.
"""

import math
import random

import pytest

np = pytest.importorskip("numpy")

from src.evals.angular import yaw_pitch_from_vector
from src.msp.pdf import combined_logpdf, combined_pdf, unit_direction_from_angles


def _params(theta0, phi0, kappa=4.0, d0=1.0, sigma_m=0.5, sigma_s=1.0):
    """Anchor and semantic mean at the origin; simple hand-checkable spreads."""
    return {
        'mu_x': 0.0, 'mu_y': 0.0, 'mu_z': 0.0, 'sigma_s': sigma_s,
        'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'd0': d0, 'sigma_m': sigma_m,
        'theta0': theta0, 'phi0': phi0, 'kappa': kappa,
    }


KAPPA = 4.0
# At any point with r = 1: semantic = -1 / 2 = -0.5 (sigma_s = 1, mu at
# anchor), metric = 0 (d0 = 1). So logpdf = -0.5 + KAPPA * dot(m, u).
BASE = -0.5


class TestStraightUp:
    """Mean direction straight up: phi0 = 0 (zenith in the y-up frame)."""

    def test_peak_directly_above_anchor(self):
        p = _params(theta0=0.0, phi0=0.0, kappa=KAPPA)
        # Directly above: u = (0, 1, 0), dot = 1. Hand: -0.5 + 4 = 3.5.
        above = combined_logpdf(0.0, 1.0, 0.0, p)
        assert above == pytest.approx(BASE + KAPPA, abs=1e-9)
        # Beside (level): dot = 0. Hand: -0.5.
        beside = combined_logpdf(1.0, 0.0, 0.0, p)
        assert beside == pytest.approx(BASE, abs=1e-9)
        # Below: dot = -1. Hand: -0.5 - 4 = -4.5.
        below = combined_logpdf(0.0, -1.0, 0.0, p)
        assert below == pytest.approx(BASE - KAPPA, abs=1e-9)
        assert above > beside > below

    def test_yaw_symmetric_around_up_axis(self):
        p = _params(theta0=0.0, phi0=0.0, kappa=KAPPA)
        # Ring of unit points at 45 degrees elevation, all yaws: the
        # up-pointing kernel must not care about yaw at all.
        # Hand: dot = cos(pi/4), logpdf = -0.5 + 4 * cos(pi/4).
        expected = BASE + KAPPA * math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        c = math.cos(math.pi / 4)
        for yaw in (0.0, 0.9, 1.7, math.pi / 2, 2.6, math.pi, -2.1, -math.pi / 2):
            v = combined_logpdf(math.cos(yaw) * s, c, math.sin(yaw) * s, p)
            assert v == pytest.approx(expected, abs=1e-9), "yaw %r broke symmetry" % yaw


class TestWorldZEqualsWorldX:
    """A +z goal with theta0 = +z must match the mirrored +x case.

    The old z-polar code collapsed the directional weight to
    kappa * cos(theta0) * dx / r for level queries, so theta0 = pi/2
    (goal along world z) scored ~0 while theta0 = 0 scored kappa.
    """

    def test_plus_z_scores_like_plus_x(self):
        p_z = _params(theta0=math.pi / 2, phi0=math.pi / 2, kappa=KAPPA)
        p_x = _params(theta0=0.0, phi0=math.pi / 2, kappa=KAPPA)
        at_z = combined_logpdf(0.0, 0.0, 1.0, p_z)  # goal along +z
        at_x = combined_logpdf(1.0, 0.0, 0.0, p_x)  # mirrored goal along +x
        # Hand: both are -0.5 + kappa * 1 = 3.5.
        assert at_z == pytest.approx(BASE + KAPPA, abs=1e-9)
        assert at_z == pytest.approx(at_x, abs=1e-12)

    def test_plus_z_kernel_prefers_plus_z_over_plus_x(self):
        p_z = _params(theta0=math.pi / 2, phi0=math.pi / 2, kappa=KAPPA)
        on_axis = combined_logpdf(0.0, 0.0, 1.0, p_z)
        off_axis = combined_logpdf(1.0, 0.0, 0.0, p_z)
        # Hand: on axis -0.5 + 4, off axis -0.5 + 0.
        assert on_axis - off_axis == pytest.approx(KAPPA, abs=1e-9)


class TestLevelNoVerticalPreference:
    """phi0 = pi/2 (level) must be indifferent to up versus down."""

    def test_mirror_in_horizontal_plane(self):
        p = _params(theta0=0.7, phi0=math.pi / 2, kappa=KAPPA)
        up = combined_logpdf(0.6, 0.5, 0.4, p)
        down = combined_logpdf(0.6, -0.5, 0.4, p)
        assert up == pytest.approx(down, abs=1e-9)

    def test_mean_direction_has_no_vertical_component(self):
        _, my, _ = unit_direction_from_angles(0.7, math.pi / 2)
        assert abs(my) < 1e-12


class TestAgreementWithAngularModule:
    """The two modules must agree by construction.

    yaw_pitch_from_vector applied to the constructed mean direction must
    recover (theta0, phi0) for random directions, within 1e-6 radians,
    using pitch = pi/2 - phi0.
    """

    def test_roundtrip_20_random_directions(self):
        rng = random.Random(20260815)
        for _ in range(20):
            theta0 = rng.uniform(-math.pi + 1e-3, math.pi - 1e-3)
            phi0 = rng.uniform(0.05, math.pi - 0.05)  # away from the poles
            m = unit_direction_from_angles(theta0, phi0)
            norm = math.sqrt(m[0]**2 + m[1]**2 + m[2]**2)
            assert norm == pytest.approx(1.0, abs=1e-12)
            yp, reason = yaw_pitch_from_vector(m)
            assert reason is None
            yaw_rad = math.radians(yp[0])
            phi_rad = math.pi / 2 - math.radians(yp[1])
            # Wrap the yaw difference to (-pi, pi].
            dyaw = (yaw_rad - theta0 + math.pi) % (2 * math.pi) - math.pi
            assert abs(dyaw) < 1e-6, (theta0, phi0, yp)
            assert abs(phi_rad - phi0) < 1e-6, (theta0, phi0, yp)


class TestInterfaceUnchanged:
    """Signatures and radial/semantic terms stay as callers expect."""

    def test_pdf_is_exp_of_logpdf(self):
        p = _params(theta0=1.1, phi0=0.8, kappa=3.0)
        lp = combined_logpdf(0.3, 0.4, 0.5, p)
        assert combined_pdf(0.3, 0.4, 0.5, p) == pytest.approx(math.exp(lp), rel=1e-12)

    def test_vectorized_arrays_accepted(self):
        p = _params(theta0=0.0, phi0=math.pi / 2, kappa=2.0)
        xs = np.array([1.0, 0.0, -1.0])
        ys = np.zeros(3)
        zs = np.zeros(3)
        out = combined_logpdf(xs, ys, zs, p)
        assert out.shape == (3,)
        # Hand: r = 1 for the outer two, dot = +1 and -1.
        assert out[0] == pytest.approx(BASE + 2.0, abs=1e-9)
        assert out[2] == pytest.approx(BASE - 2.0, abs=1e-9)

    def test_radial_gaussian_unchanged(self):
        # Pure metric check: kill semantic (huge sigma_s) and direction
        # (kappa = 0). Hand: at r = 2 with d0 = 1, sigma_m = 0.5:
        # -(2 - 1)^2 / (2 * 0.25) = -2.0.
        p = _params(theta0=0.0, phi0=math.pi / 2, kappa=0.0, d0=1.0,
                    sigma_m=0.5, sigma_s=1e6)
        v = combined_logpdf(2.0, 0.0, 0.0, p)
        assert v == pytest.approx(-2.0, abs=1e-6)

"""MAPG-01: kernel bandwidths from real Hydra AABB extents.

Host-runnable tests for src.msp.kernel_bandwidths.py, the pure
sizing helpers MultiAgentMSPPlanner feeds the MSP engine with. The
engine itself (MSPEngineSmart) is not importable on the host (its
module chain needs the provider SDKs), so the sigma_s assertions use
the documented mirror ``sigma_s_m``: the engine derives
sigma_s = sigma_s_factor * max(anchor_size), and the planner hands it
an isotropic cube whose max is exactly the bandwidth scale.

Worked example (the ticket's acceptance case): a 2 m sofa and a 0.3 m
lamp get different sigma_s under from_bbox and identical sigma_s under
fixed.
"""

import unittest

from src.msp.kernel_bandwidths import (
    DEFAULT_MODE,
    FIXED_SIZE_M,
    bandwidth_size_hab,
    extents_zup_to_size_hab,
    max_horizontal_extent_hab,
    resolve_mode,
    resolve_object_size_hab,
    sigma_s_m,
)

# z-up Hydra extents [dx, dy, dz_up], as stored by SceneGraphSim.
SOFA_EXTENTS_ZUP = [2.0, 0.9, 0.8]   # 2 m wide sofa
LAMP_EXTENTS_ZUP = [0.3, 0.25, 1.4]  # 0.3 m floor lamp, tall


class TestResolveMode(unittest.TestCase):
    def test_default_is_from_bbox(self):
        mode, warn = resolve_mode(None)
        self.assertEqual(mode, "from_bbox")
        self.assertEqual(DEFAULT_MODE, "from_bbox")
        self.assertIsNone(warn)

    def test_fixed_accepted(self):
        mode, warn = resolve_mode("fixed")
        self.assertEqual(mode, "fixed")
        self.assertIsNone(warn)

    def test_case_and_whitespace_normalized(self):
        mode, warn = resolve_mode("  From_Bbox ")
        self.assertEqual(mode, "from_bbox")
        self.assertIsNone(warn)

    def test_unknown_falls_back_with_warning(self):
        mode, warn = resolve_mode("banana")
        self.assertEqual(mode, "from_bbox")
        self.assertIn("banana", warn)


class TestFrameMapping(unittest.TestCase):
    def test_zup_to_habitat_order(self):
        # z-up [dx, dy, dz_up] -> habitat y-up [dx, dz_up, dy], matching
        # pos_normal_to_habitat (+90 deg about x, unsigned extents).
        self.assertEqual(extents_zup_to_size_hab([1.0, 2.0, 3.0]), [1.0, 3.0, 2.0])

    def test_max_horizontal_is_habitat_x_z(self):
        # habitat y (index 1) is vertical and must be excluded.
        self.assertEqual(max_horizontal_extent_hab([0.3, 9.0, 0.25]), 0.3)


class TestWorkedExample(unittest.TestCase):
    """A 2 m sofa and a 0.3 m lamp: different sigma_s under from_bbox,
    identical sigma_s under fixed."""

    def _scale(self, extents_zup, mode):
        size_hab, source, reason = resolve_object_size_hab(extents_zup, mode)
        cube, scale = bandwidth_size_hab(size_hab, source)
        return size_hab, source, reason, cube, scale

    def test_from_bbox_sofa(self):
        size_hab, source, reason, cube, scale = self._scale(SOFA_EXTENTS_ZUP, "from_bbox")
        self.assertEqual(source, "hydra_bbox")
        self.assertIsNone(reason)
        self.assertEqual(size_hab, [2.0, 0.8, 0.9])
        # s = max horizontal extent = max(2.0, 0.9) = 2.0
        self.assertEqual(scale, 2.0)
        self.assertEqual(cube, [2.0, 2.0, 2.0])
        # sigma_s = 0.5 * s = 1.0 m
        self.assertAlmostEqual(sigma_s_m(scale), 1.0)

    def test_from_bbox_lamp(self):
        size_hab, source, reason, cube, scale = self._scale(LAMP_EXTENTS_ZUP, "from_bbox")
        self.assertEqual(source, "hydra_bbox")
        self.assertIsNone(reason)
        self.assertEqual(size_hab, [0.3, 1.4, 0.25])
        # Vertical 1.4 m excluded: s = max(0.3, 0.25) = 0.3
        self.assertEqual(scale, 0.3)
        self.assertEqual(cube, [0.3, 0.3, 0.3])
        # sigma_s = 0.5 * 0.3 = 0.15 m
        self.assertAlmostEqual(sigma_s_m(scale), 0.15)

    def test_from_bbox_sofa_and_lamp_differ(self):
        _, _, _, _, s_sofa = self._scale(SOFA_EXTENTS_ZUP, "from_bbox")
        _, _, _, _, s_lamp = self._scale(LAMP_EXTENTS_ZUP, "from_bbox")
        self.assertNotEqual(sigma_s_m(s_sofa), sigma_s_m(s_lamp))

    def test_fixed_identical_for_both(self):
        for extents in (SOFA_EXTENTS_ZUP, LAMP_EXTENTS_ZUP):
            size_hab, source, reason, cube, scale = self._scale(extents, "fixed")
            self.assertEqual(source, "fixed_cfg")
            self.assertIsNone(reason)
            self.assertEqual(size_hab, list(FIXED_SIZE_M))
            self.assertEqual(cube, list(FIXED_SIZE_M))
            self.assertEqual(scale, 0.5)
            # Historical behavior: sigma_s = 0.5 * 0.5 = 0.25 m
            self.assertAlmostEqual(sigma_s_m(scale), 0.25)


class TestFallbacks(unittest.TestCase):
    """A missing or invalid box falls back to the fixed cube with a
    recorded reason and never raises."""

    BAD_EXTENTS = [
        None,
        [],
        [1.0, 2.0],
        [float("nan"), 1.0, 1.0],
        [float("inf"), 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
        ["a", "b", "c"],
        object(),
    ]

    def test_all_bad_extents_fall_back_with_reason(self):
        for bad in self.BAD_EXTENTS:
            size_hab, source, reason = resolve_object_size_hab(bad, "from_bbox")
            self.assertEqual(size_hab, list(FIXED_SIZE_M), msg=repr(bad))
            self.assertEqual(source, "fixed_fallback", msg=repr(bad))
            self.assertTrue(isinstance(reason, str) and reason, msg=repr(bad))

    def test_fallback_bandwidth_is_historical_cube(self):
        size_hab, source, _ = resolve_object_size_hab(None, "from_bbox")
        cube, scale = bandwidth_size_hab(size_hab, source)
        self.assertEqual(cube, list(FIXED_SIZE_M))
        self.assertEqual(scale, 0.5)
        self.assertAlmostEqual(sigma_s_m(scale), 0.25)

    def test_fixed_mode_ignores_extents_entirely(self):
        size_hab, source, reason = resolve_object_size_hab(object(), "fixed")
        self.assertEqual(size_hab, list(FIXED_SIZE_M))
        self.assertEqual(source, "fixed_cfg")
        self.assertIsNone(reason)


class TestSigmaSMirror(unittest.TestCase):
    def test_formula(self):
        # sigma_s = sigma_s_factor * scale (engine: factor * max(size)).
        self.assertAlmostEqual(sigma_s_m(2.0, 0.5), 1.0)
        self.assertAlmostEqual(sigma_s_m(0.3, 0.5), 0.15)
        self.assertAlmostEqual(sigma_s_m(1.0, 0.25), 0.25)


if __name__ == "__main__":
    unittest.main()

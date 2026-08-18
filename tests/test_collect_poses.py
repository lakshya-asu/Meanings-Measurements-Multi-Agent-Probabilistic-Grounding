"""Unit tests for the pure logic in src/scripts/collect_init_poses.py.

Habitat-free: floor-band computation, band picking, yaw choice, diverse
selection, composite scoring, review-sheet parsing and the v2 CSV append
format are all exercised with plain python data. The habitat-dependent
path is covered by the in-container smoke run (see the board log).
"""

import math

import pytest

from src.scripts.collect_init_poses import (
    angle_from_direction,
    anchor_heights_for_pair,
    build_v2_text,
    choose_yaw,
    cluster_heights,
    composite_score,
    format_pose_row,
    missing_pairs,
    pair_seed,
    parse_review_selections,
    pick_floor_band,
    select_diverse,
    yaw_bin_angle,
)


# ---------------------------------------------------------------------------
# floor bands
# ---------------------------------------------------------------------------

def test_cluster_heights_two_floors():
    ys = [0.02, 0.05, 0.11, 0.0, 3.02, 3.1, 2.98]
    bands = cluster_heights(ys, gap=0.5)
    assert len(bands) == 2
    (lo0, hi0, n0), (lo1, hi1, n1) = bands
    assert lo0 == 0.0 and hi0 == 0.11 and n0 == 4
    assert lo1 == 2.98 and hi1 == 3.1 and n1 == 3


def test_cluster_heights_single_and_empty():
    assert cluster_heights([]) == []
    assert cluster_heights([1.5]) == [(1.5, 1.5, 1)]


def test_cluster_heights_chain_within_gap():
    # Steps of 0.4 chain into one band even though the ends are far apart.
    bands = cluster_heights([0.0, 0.4, 0.8, 1.2], gap=0.5)
    assert bands == [(0.0, 1.2, 4)]


def test_pick_floor_band_votes():
    bands = [(-3.1, -2.9, 50), (0.0, 0.2, 80)]
    # Anchors sit 0.5 to 1.5 above the upper floor.
    idx, votes = pick_floor_band(bands, [0.6, 1.5, 0.9])
    assert idx == 1
    assert votes == [0, 3]


def test_pick_floor_band_lower_floor():
    bands = [(-3.1, -2.9, 50), (0.0, 0.2, 80)]
    idx, votes = pick_floor_band(bands, [-2.2, -2.4])
    assert idx == 0
    assert votes == [2, 0]


def test_pick_floor_band_tie_breaks_toward_median_anchor():
    # One anchor votes for each band; the median anchor height decides.
    bands = [(0.0, 0.1, 10), (3.0, 3.1, 10)]
    idx, votes = pick_floor_band(bands, [0.5, 0.6, 3.5])
    assert votes == [2, 1]
    assert idx == 0


def test_pick_floor_band_no_anchors_uses_biggest_band():
    bands = [(0.0, 0.1, 10), (3.0, 3.1, 90)]
    idx, _ = pick_floor_band(bands, [])
    assert idx == 1


def test_pick_floor_band_empty_raises():
    with pytest.raises(ValueError):
        pick_floor_band([], [1.0])


# ---------------------------------------------------------------------------
# yaw
# ---------------------------------------------------------------------------

def test_angle_from_direction_matches_habitat_forward():
    # Habitat forward is -z: angle 0 must look along (0, 0, -1).
    for a in [0.0, 0.7, -2.1, math.pi / 2]:
        fx, fz = -math.sin(a), -math.cos(a)
        assert angle_from_direction(fx, fz) == pytest.approx(a)


def test_choose_yaw_picks_open_sector():
    n = 72
    prof = [0.5] * n
    # Open corridor peaking at bin 18 (90 degrees to the left).
    for b in range(6, 31):
        prof[b] = 5.0 - 0.2 * abs(b - 18)
    yaw, mean = choose_yaw(prof)
    assert yaw == pytest.approx(yaw_bin_angle(18, n), abs=2 * math.pi / n + 1e-9)
    assert mean > 2.0


def test_choose_yaw_wraparound():
    n = 72
    prof = [0.25] * n
    # Open sector straddling bin 0.
    for b in list(range(0, 5)) + list(range(67, 72)):
        prof[b] = 4.0
    yaw, _ = choose_yaw(prof)
    assert abs(yaw) < math.radians(30)


def test_choose_yaw_deterministic_on_uniform_profile():
    yaw, mean = choose_yaw([1.0] * 72)
    assert yaw == pytest.approx(0.0)
    assert mean == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# candidate selection and scoring
# ---------------------------------------------------------------------------

def test_select_diverse_spreads_and_is_deterministic():
    pts = [(0, 0, 0), (0.1, 0, 0), (5, 0, 0), (0, 0, 5), (5, 0, 5), (2.5, 0, 2.5)]
    got = select_diverse(pts, 4)
    assert got == select_diverse(pts, 4)
    assert len(set(got)) == 4
    # The two near-duplicates at the origin must not both be picked.
    assert not ({0, 1} <= set(got))


def test_select_diverse_small_pool_returns_all():
    assert select_diverse([(0, 0, 0), (1, 1, 1)], 5) == [0, 1]


def test_composite_score_ordering_and_bounds():
    good = composite_score(1.0, 3.0, 1.0, 1.0)
    bad = composite_score(0.1, 0.5, 0.2, 0.1)
    assert 0.0 <= bad < good <= 1.0
    # Clearance dominates openness per the weights.
    a = composite_score(1.0, 0.0, 0.0, 0.0)
    b = composite_score(0.0, 3.0, 0.0, 0.0)
    assert a > b


def test_bbox_check_penetration_tolerance():
    from src.scripts.collect_init_poses import _bbox_check
    # A 2x2 box at origin, tall enough to overlap the torso slab.
    boxes = [("table", [-1.0, 0.0, -1.0], [1.0, 2.0, 1.0])]
    # Grazing 5 cm inside: not a hard fail.
    deep, dist, pen = _bbox_check(boxes, [0.95, 0.0, 0.0])
    assert not deep and dist == 0.0 and pen == pytest.approx(0.05)
    # Deep inside: hard fail.
    deep, _, pen = _bbox_check(boxes, [0.0, 0.0, 0.0])
    assert deep and pen == pytest.approx(1.0)
    # Outside: distance reported.
    deep, dist, pen = _bbox_check(boxes, [1.5, 0.0, 0.0])
    assert not deep and dist == pytest.approx(0.5) and pen == 0.0
    # Box entirely above the head does not count.
    high = [("lamp", [-1.0, 2.0, -1.0], [1.0, 3.0, 1.0])]
    deep, dist, _ = _bbox_check(high, [0.0, 0.0, 0.0])
    assert not deep and dist == 1.0


def test_pair_seed_stable_and_distinct():
    s1 = pair_seed(20260815, "00217-qz3829g1Lzf_0")
    assert s1 == pair_seed(20260815, "00217-qz3829g1Lzf_0")
    assert s1 != pair_seed(20260815, "00217-qz3829g1Lzf_3")
    assert 0 <= s1 < 2 ** 31


# ---------------------------------------------------------------------------
# CSV append format
# ---------------------------------------------------------------------------

ORIGINAL = (
    "scene_floor,init_x,init_y,init_z,init_angle\n"
    "00506-QVAA6zecMHu_0,0.84,0.04,0.76,-0.785\n"
    "00299-bdp1XNEdvmW_0,3.36,0,-2.74,1.196\n"
)


def test_format_pose_row_matches_existing_style():
    assert format_pose_row("s_0", 10.081, 0.0, -3.955, 1.2204) == \
        "s_0,10.08,0,-3.96,1.22"
    assert format_pose_row("s_1", 2.0, -3.09, 0.5, -0.7854) == \
        "s_1,2,-3.09,0.5,-0.785"


def test_build_v2_text_appends_verbatim():
    new = [("00217-qz3829g1Lzf_0", 1.2345, 0.061, -2.5, 0.5236)]
    out = build_v2_text(ORIGINAL, new)
    assert out.startswith(ORIGINAL)
    assert out.endswith("00217-qz3829g1Lzf_0,1.23,0.06,-2.5,0.524\n")
    assert out.count("\n") == ORIGINAL.count("\n") + 1


def test_build_v2_text_handles_missing_trailing_newline():
    out = build_v2_text(ORIGINAL.rstrip("\n"), [("a_0", 1, 2, 3, 0.1)])
    assert "1.196\na_0,1,2,3,0.1\n" in out


def test_build_v2_text_rejects_duplicate_pair():
    with pytest.raises(ValueError):
        build_v2_text(ORIGINAL, [("00299-bdp1XNEdvmW_0", 1, 2, 3, 4)])


# ---------------------------------------------------------------------------
# split bookkeeping
# ---------------------------------------------------------------------------

BENCH = [
    {"scene": "00023-zepmXAdrpjR", "floor": "0", "ann_pos_y": "1.62",
     "anchor_center_y": "1.60"},
    {"scene": "00023-zepmXAdrpjR", "floor": "1", "ann_pos_y": "4.1",
     "anchor_center_y": "4.0"},
    {"scene": "00057-1UnKg1rAb8A", "floor": "2.0", "ann_pos_y": "-0.71",
     "anchor_center_y": ""},
]


def test_missing_pairs_and_floor_coercion():
    got = missing_pairs(BENCH, ["00023-zepmXAdrpjR_0"])
    assert got == ["00023-zepmXAdrpjR_1", "00057-1UnKg1rAb8A_2"]


def test_anchor_heights_for_pair():
    hs = anchor_heights_for_pair(BENCH, "00023-zepmXAdrpjR_1")
    assert hs == [4.1, 4.0]
    assert anchor_heights_for_pair(BENCH, "00057-1UnKg1rAb8A_2") == [-0.71]
    assert anchor_heights_for_pair(BENCH, "nope_0") == []


# ---------------------------------------------------------------------------
# review sheet parsing
# ---------------------------------------------------------------------------

REVIEW = """# Init pose review sheet

## 00217-qz3829g1Lzf_0

| cand | recommended | x | score | image | ACCEPT | REJECT |
|---|---|---|---|---|---|---|
| cand_0 | RECOMMENDED | 1.0 | 0.9 | a.png | [ ] | [ ] |
| cand_1 |  | 2.0 | 0.8 | b.png | [x] | [ ] |

## 00245-741Fdj7NLF9_0

| cand | recommended | x | score | image | ACCEPT | REJECT |
|---|---|---|---|---|---|---|
| cand_0 | RECOMMENDED | 3.0 | 0.7 | c.png | [X] | [ ] |
| cand_1 |  | 4.0 | 0.6 | d.png | [ ] | [x] |

## 00720-8B43pG641ff_0

SCENE FAILED: boom
"""


def test_parse_review_selections():
    got = parse_review_selections(REVIEW)
    assert got == {
        "00217-qz3829g1Lzf_0": "cand_1",
        "00245-741Fdj7NLF9_0": "cand_0",
    }


def test_parse_review_rejects_double_accept():
    bad = REVIEW.replace("| 1.0 | 0.9 | a.png | [ ]", "| 1.0 | 0.9 | a.png | [x]")
    with pytest.raises(ValueError):
        parse_review_selections(bad)

"""Tests for src/parsing/metric_literal.py (P0 fix 1: d0 single source).

Stdlib only. The parser is deterministic and has NO default: absence of
a metric literal returns value_m None, never 1.0.
"""

import math

from src.parsing.metric_literal import (
    MetricParse,
    parse_metric_literal,
    infer_relation,
    resolve_categorical_distance,
    RELATION_DEFAULT_DISTANCES_M,
    PLAUSIBLE_RANGE_M,
)


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# (utterance, expected value_m or None) covering plain forms, unit
# variants, word numbers, fractions, and adversarial near-misses.
UTTERANCE_TABLE = [
    # Plain numeric + unit
    ("Where is the spot 2.5m in front of the tv?", 2.5),
    ("Go to the point 2.5 m from the couch", 2.5),
    ("stand 3 meters behind the sofa", 3.0),
    ("a location 3.0 meters in front of the large TV", 3.0),
    ("the point 3 metres left of the bed", 3.0),
    ("move 50 cm to the right of the fridge", 0.5),
    ("about 50 centimeters from the lamp", 0.5),
    ("stand 5 feet from the mirror", 5 * 0.3048),
    ("one foot from the door", 0.3048),
    ("12 inches from the wall", 12 * 0.0254),
    ("2,5 m from the plant", 2.5),
    ("2m ahead of the chair", 2.0),
    # Word numbers
    ("two meters in front of the desk", 2.0),
    ("ten meters from the entrance", 10.0),
    # Fractions and article forms
    ("half a meter from the nightstand", 0.5),
    ("a meter and a half in front of the wardrobe", 1.5),
    ("two and a half meters from the oven", 2.5),
    ("one and a half metres behind the armchair", 1.5),
    ("about a meter away from the sofa", 1.0),
    # No distance at all
    ("where is the chair near the table?", None),
    ("find the mug next to the sink", None),
    ("what is between the bed and the dresser?", None),
    # Adversarial near-misses: numbers that are NOT distances
    ("go to room 2", None),
    ("sit on the 2 seater sofa", None),
    ("the apartment on floor 3", None),
    ("check the 3 drawers in the cabinet", None),
    # Unsupported units must not parse as meters
    ("a gap of 2 mm", None),
    ("the trail is 2 miles long", None),
    ("it is 5 in on the ruler", None),
    # A literal near an object name still parses
    ("find the spot 2 meters from the 2 seater sofa", 2.0),
]


def test_utterance_table():
    for text, expected in UTTERANCE_TABLE:
        got = parse_metric_literal(text).value_m
        if expected is None:
            assert got is None, f"{text!r}: expected None, got {got}"
        else:
            assert got is not None, f"{text!r}: expected {expected}, got None"
            assert _close(got, expected), f"{text!r}: expected {expected}, got {got}"


def test_table_is_large_enough():
    assert len(UTTERANCE_TABLE) >= 25


def test_no_default_value():
    # The old _parse_q_dist returned 1.0 here. Absence means None now.
    p = parse_metric_literal("where is the plant near the window?")
    assert p.value_m is None
    assert p.unit is None
    assert p.raw is None
    assert p.warnings == []


def test_result_fields_populated():
    p = parse_metric_literal("stand 50 cm from the lamp")
    assert isinstance(p, MetricParse)
    assert _close(p.value_m, 0.5)
    assert p.unit == "cm"
    assert "50 cm" in p.raw
    assert p.warnings == []


def test_empty_and_none_input():
    assert parse_metric_literal("").value_m is None
    assert parse_metric_literal("   ").value_m is None
    assert parse_metric_literal(None).value_m is None
    assert "empty_text" in parse_metric_literal(None).warnings


def test_multiple_literals_first_wins_with_warning():
    p = parse_metric_literal("2 meters from the tv and 3 meters from the couch")
    assert _close(p.value_m, 2.0)
    assert any("multiple" in w for w in p.warnings)


def test_out_of_range_warns_but_parses():
    lo, hi = PLAUSIBLE_RANGE_M
    p_small = parse_metric_literal("a shift of 0.05 m")
    assert _close(p_small.value_m, 0.05)
    assert any("out_of_range" in w for w in p_small.warnings)
    p_big = parse_metric_literal("a walk of 20 meters")
    assert _close(p_big.value_m, 20.0)
    assert any("out_of_range" in w for w in p_big.warnings)
    p_ok = parse_metric_literal("a walk of 5 meters")
    assert not any("out_of_range" in w for w in p_ok.warnings)
    assert lo < 5.0 < hi


def test_determinism():
    text = "a meter and a half in front of the wardrobe"
    first = parse_metric_literal(text)
    second = parse_metric_literal(text)
    assert first == second


def test_categorical_defaults():
    assert _close(resolve_categorical_distance("near"), 0.5)
    assert _close(resolve_categorical_distance("next_to"), 0.75)
    assert _close(resolve_categorical_distance("intrinsic_front"), 1.0)
    assert _close(resolve_categorical_distance("in_front_of"), 1.0)
    assert _close(resolve_categorical_distance("intrinsic_back"), 1.0)
    assert _close(resolve_categorical_distance("intrinsic_left"), 0.75)
    # No radial default for these: metric kernel stays omitted.
    assert resolve_categorical_distance("between") is None
    assert resolve_categorical_distance("none") is None
    assert resolve_categorical_distance("unknown_relation") is None
    assert resolve_categorical_distance(None) is None
    # Table values are finite positive when present.
    for rel, val in RELATION_DEFAULT_DISTANCES_M.items():
        if val is not None:
            assert math.isfinite(val) and val > 0, rel


def test_infer_relation():
    assert infer_relation("the point in front of the tv") == "in_front_of"
    assert infer_relation("the lamp behind the couch") == "behind"
    assert infer_relation("the mug next to the sink") == "next_to"
    assert infer_relation("the chair near the table") == "near"
    assert infer_relation("what is between the bed and the desk") == "between"
    assert infer_relation("where is the plant?") is None
    assert infer_relation(None) is None

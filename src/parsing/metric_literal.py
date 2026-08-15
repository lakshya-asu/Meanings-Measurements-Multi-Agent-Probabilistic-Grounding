"""Single source of truth for metric literals (d0) in MAPG (P0 fix 1).

Every planner obtains the commanded distance d0 ONLY from
``parse_metric_literal``, applied once per episode to the raw question.
There is deliberately NO default value: when the utterance contains no
metric literal, ``value_m`` is None and the caller must OMIT the metric
kernel from the product-of-experts composition instead of fabricating a
1.0 m constraint (the old ``_parse_q_dist`` silent default).

Composition change for no-distance queries
------------------------------------------
Old behavior: a query with no distance literal was scored with a hard
radial Gaussian peaked at a fabricated d0 = 1.0 m. New behavior: the
metric expert is dropped from the composition entirely (the density is
directional + semantic only, or directional only in O-W mode) and the
episode record carries ``metric_kernel_active: false`` so ablations can
condition on it.

Ablation modes (cfg flag ``numeric_conditioning``)
--------------------------------------------------
- ``on`` (default): d0 = parsed literal, or None (metric kernel omitted).
- ``off``: d0 = None always; the metric kernel is dropped even when a
  literal exists.
- ``categorical_only``: d0 = a per-relation nominal distance from
  ``RELATION_DEFAULT_DISTANCES_M`` (the LINGO-Space-style comparison:
  categorical predicate conditioning with no numeric-literal pathway).

Deterministic grammar
---------------------
Numbers: integers, decimals (dot or comma), word numbers one..ten,
articles a/an (= 1), and half forms ("half a meter", "a meter and a
half", "two and a half meters"). Units: meters/metres/m,
centimeters/centimetres/cm, feet/foot/ft, inches/inch. All values are
converted to meters. The bare token "in" is NOT accepted for inches
(too ambiguous: "in front of"). Unknown units (mm, km, miles) do not
parse. The first literal wins; extra literals add a warning. Values
outside [0.1, 10.0] m parse but add an out-of-range warning.

This module is stdlib only on purpose: importable and testable outside
the container.
"""

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

# Meters per unit token.
UNIT_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "metre": 1.0,
    "metres": 1.0,
    "cm": 0.01,
    "centimeter": 0.01,
    "centimeters": 0.01,
    "centimetre": 0.01,
    "centimetres": 0.01,
    "ft": 0.3048,
    "foot": 0.3048,
    "feet": 0.3048,
    "inch": 0.0254,
    "inches": 0.0254,
}

WORD_NUMBERS = {
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
}

# Plausible commanded-distance range for an indoor scene, in meters.
PLAUSIBLE_RANGE_M = (0.1, 10.0)

# Per-relation nominal distances (meters) for numeric_conditioning =
# "categorical_only". These are the documented stand-ins a categorical
# system would use when the utterance's numeric literal has no pathway
# into the distribution. "between" and "none" have no radial default:
# the metric kernel stays omitted for them.
RELATION_DEFAULT_DISTANCES_M = {
    "near": 0.5,
    "next_to": 0.75,
    "beside": 0.75,
    "in_front_of": 1.0,
    "intrinsic_front": 1.0,
    "behind": 1.0,
    "intrinsic_back": 1.0,
    "left_of": 0.75,
    "intrinsic_left": 0.75,
    "right_of": 0.75,
    "intrinsic_right": 0.75,
    "above": 0.5,
    "below": 0.5,
    "between": None,
    "none": None,
}


@dataclass
class MetricParse:
    """Result of a deterministic metric-literal parse.

    value_m: parsed distance in meters, or None when the text contains
        no metric literal. None means "omit the metric kernel", never
        "use a default".
    unit: the raw unit token that matched (e.g. "meters", "cm"), or None.
    raw: the full matched span (e.g. "2.5 m"), or None.
    warnings: deterministic parse warnings (multiple literals, value out
        of the plausible range, empty input).
    """

    value_m: Optional[float]
    unit: Optional[str]
    raw: Optional[str]
    warnings: List[str] = field(default_factory=list)


# Unit alternation sorted longest-first so "meters" wins over "m".
_ALL_UNITS = "|".join(sorted(UNIT_TO_METERS, key=len, reverse=True))
# Full-word singular units that may follow a bare article ("a meter").
_ARTICLE_UNITS = "meter|metre|centimeter|centimetre|foot|inch"
_WORD_NUMS = "|".join(WORD_NUMBERS)

_MASTER_RE = re.compile(
    r"(?P<half>\bhalf\s+(?:of\s+)?an?\s+(?P<unit_h>" + _ARTICLE_UNITS + r")\b)"
    r"|(?P<numhalf>\b(?P<q_nh>\d+(?:[.,]\d+)?|" + _WORD_NUMS + r")\s+and\s+a\s+half\s+(?P<unit_nh>" + _ALL_UNITS + r")\b)"
    r"|(?P<arthalf>\ban?\s+(?P<unit_ah>" + _ARTICLE_UNITS + r")\s+and\s+a\s+half\b)"
    r"|(?P<numunit>\b(?P<q_nu>\d+(?:[.,]\d+)?)\s*(?P<unit_nu>" + _ALL_UNITS + r")\b)"
    r"|(?P<wordunit>\b(?P<q_wu>" + _WORD_NUMS + r")\s+(?P<unit_wu>" + _ALL_UNITS + r")\b)"
    r"|(?P<artunit>\ban?\s+(?P<unit_au>" + _ARTICLE_UNITS + r")\b)"
)


def _quantity(token: str) -> float:
    token = token.strip().lower()
    if token in WORD_NUMBERS:
        return WORD_NUMBERS[token]
    return float(token.replace(",", "."))


def parse_metric_literal(text: Optional[str]) -> MetricParse:
    """Parse the first metric literal in text. Deterministic, no default.

    Returns a MetricParse; value_m is None when nothing parses. See the
    module docstring for the grammar and the composition contract.
    """
    if text is None or not str(text).strip():
        return MetricParse(None, None, None, ["empty_text"])

    s = str(text).lower()
    matches = list(_MASTER_RE.finditer(s))
    if not matches:
        return MetricParse(None, None, None, [])

    warnings: List[str] = []
    if len(matches) > 1:
        warnings.append(
            "multiple_metric_literals: %d literals found, using the first" % len(matches)
        )

    m = matches[0]
    if m.group("half") is not None:
        qty = 0.5
        unit = m.group("unit_h")
    elif m.group("numhalf") is not None:
        qty = _quantity(m.group("q_nh")) + 0.5
        unit = m.group("unit_nh")
    elif m.group("arthalf") is not None:
        qty = 1.5
        unit = m.group("unit_ah")
    elif m.group("numunit") is not None:
        qty = _quantity(m.group("q_nu"))
        unit = m.group("unit_nu")
    elif m.group("wordunit") is not None:
        qty = _quantity(m.group("q_wu"))
        unit = m.group("unit_wu")
    else:
        qty = 1.0
        unit = m.group("unit_au")

    value_m = qty * UNIT_TO_METERS[unit]
    if not math.isfinite(value_m):
        return MetricParse(None, unit, m.group(0), warnings + ["non_finite_value"])

    lo, hi = PLAUSIBLE_RANGE_M
    if not (lo <= value_m <= hi):
        warnings.append(
            "value_out_of_range: %.4f m outside [%.1f, %.1f]" % (value_m, lo, hi)
        )

    return MetricParse(value_m=float(value_m), unit=unit, raw=m.group(0), warnings=warnings)


# Ordered phrase table for deterministic relation inference from raw
# text (used when no structured relation label is available, e.g. the
# single-agent planner in categorical_only mode). First hit wins.
_RELATION_PHRASES = (
    ("in front of", "in_front_of"),
    ("front of", "in_front_of"),
    ("behind", "behind"),
    ("next to", "next_to"),
    ("beside", "beside"),
    ("close to", "near"),
    ("near", "near"),
    ("left of", "left_of"),
    ("right of", "right_of"),
    ("above", "above"),
    ("below", "below"),
    ("under", "below"),
    ("between", "between"),
)


def infer_relation(text: Optional[str]) -> Optional[str]:
    """Deterministic keyword scan for the spatial relation in text."""
    if not text:
        return None
    s = str(text).lower()
    for phrase, relation in _RELATION_PHRASES:
        if phrase in s:
            return relation
    return None


def resolve_categorical_distance(relation: Optional[str]) -> Optional[float]:
    """Nominal distance for a relation label, or None (no radial default).

    Accepts the orchestrator's composition_logic labels (near, between,
    intrinsic_front, ...) and the phrase-table labels from
    infer_relation. Unknown labels resolve to None: the metric kernel
    stays omitted rather than guessing.
    """
    if relation is None:
        return None
    key = str(relation).strip().lower().replace(" ", "_")
    return RELATION_DEFAULT_DISTANCES_M.get(key)

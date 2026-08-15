"""Tests for object-selection scoring (item 6): strict node-ID rule and
the lenient free-text cascade of metrics.md 2.3.

Stdlib + pytest only. The embedding step is exercised through fake
embedders returning fixed vectors, so no sentence-transformers install
is needed and every result is deterministic.
"""

import hashlib
import json
import os

import pytest

from src.evals.object_selection import (
    EMBED_THRESHOLD,
    MatchResult,
    SYNONYMS_PATH,
    canonicalize,
    cosine_similarity,
    lenient_match,
    load_synonyms,
    strict_accuracy,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A tiny synonym table so canonicalization tests do not depend on the
# exact contents of the frozen file.
SYN = {
    "couch": "sofa",
    "tv": "television",
    "garbage can": "trash can",
    "bin": "trash can",
}


def nodes(*specs):
    """Build scene-node dicts from (node_id, label[, centroid]) tuples."""
    out = []
    for spec in specs:
        node = {"node_id": spec[0], "label": spec[1]}
        if len(spec) > 2:
            node["centroid"] = list(spec[2])
        out.append(node)
    return out


# ---------------------------------------------------------------- canonicalize

@pytest.mark.parametrize("raw, expected", [
    ("sofa", "sofa"),
    ("The Couch", "sofa"),                 # article + case + synonym
    ("couches", "sofa"),                   # -es plural then synonym
    ("TVs", "television"),                 # short word plural + synonym
    ("a garbage-can!", "trash can"),       # punctuation to space, synonym
    ("shelves", "shelf"),                  # irregular plural
    ("accessories", "accessory"),          # -ies plural
    ("glasses", "glass"),                  # -sses keeps the stem s
    ("cactus", "cactus"),                  # -us is not a plural marker
    ("  the   lamps  ", "lamp"),           # whitespace collapse + -s
    ("ｓｏｆａ", "sofa"),  # NFKC folds fullwidth letters
    ("Bin", "trash can"),
])
def test_canonicalize_table(raw, expected):
    assert canonicalize(raw, SYN) == expected


def test_canonicalize_is_deterministic():
    for _ in range(3):
        assert canonicalize("The Couches", SYN) == "sofa"


# --------------------------------------------------------------------- strict

def test_strict_accuracy_exact_match():
    assert strict_accuracy(7, 7) is True
    assert strict_accuracy(7, 8) is False


def test_strict_accuracy_rejects_non_int_predictions():
    assert strict_accuracy(None, 7) is False
    assert strict_accuracy("7", 7) is False
    assert strict_accuracy(7.0, 7) is False
    assert strict_accuracy(True, 1) is False


def test_strict_accuracy_requires_int_gt():
    with pytest.raises(ValueError):
        strict_accuracy(7, "7")


# ------------------------------------------------------------ lenient cascade

def test_lenient_exact_string_hit_same_scene():
    scene = nodes((3, "sofa"), (5, "table"))
    res = lenient_match("the couch", scene, gt_node_id=3, synonyms=SYN)
    assert isinstance(res, MatchResult)
    assert res.matched and res.correct
    assert res.node_id == 3
    assert res.step == "exact_string"
    assert res.cosine is None
    assert res.tie_break is None
    assert res.pred_canonical == "sofa"


def test_lenient_synonym_hit():
    scene = nodes((11, "television"), (12, "sofa"))
    res = lenient_match("TV", scene, gt_node_id=11, synonyms=SYN)
    assert res.matched and res.correct and res.node_id == 11
    assert res.step == "exact_string"


def test_lenient_exact_hit_wrong_node_is_incorrect():
    # Matched a real node, but not the GT node: matched yet incorrect.
    scene = nodes((3, "sofa"), (5, "table"))
    res = lenient_match("table", scene, gt_node_id=3, synonyms=SYN)
    assert res.matched and not res.correct
    assert res.node_id == 5


def test_lenient_unmatched_below_threshold_counted_incorrect():
    # Orthogonal vectors: every cosine is 0, nothing reaches 0.75.
    def embedder(texts):
        return [[1.0 if i == j else 0.0 for j in range(len(texts))]
                for i in range(len(texts))]

    scene = nodes((1, "sofa"), (2, "table"))
    res = lenient_match("zebra", scene, gt_node_id=1,
                        embedder=embedder, synonyms=SYN)
    assert not res.matched
    assert not res.correct          # counted incorrect, never dropped
    assert res.node_id is None
    assert res.step == "unmatched"
    assert res.cosine == pytest.approx(0.0)


def test_lenient_embedding_hit_above_threshold():
    vectors = {
        "armchair": [1.0, 0.0],
        "sofa": [0.9, 0.1],       # cosine ~0.994, above 0.75
        "loveseat": [0.8, 0.2],   # cosine ~0.970, above 0.75 but lower
        "table": [0.0, 1.0],      # cosine 0
    }

    def embedder(texts):
        return [vectors[t] for t in texts]

    scene = nodes((4, "sofa"), (9, "table"), (6, "loveseat"))
    res = lenient_match("armchair", scene, gt_node_id=4,
                        embedder=embedder, synonyms=SYN)
    assert res.matched and res.correct and res.node_id == 4
    assert res.step == "embedding"
    assert res.cosine is not None and res.cosine >= EMBED_THRESHOLD
    assert res.tie_break == "cosine"


def test_lenient_tiebreak_cosine_then_centroid_then_id():
    # Two candidates tie exactly on cosine; the one whose centroid is
    # closer to the GT node must win.
    vec = {"pred": [1.0, 0.0], "sofa": [1.0, 0.0], "table": [0.0, 1.0]}

    def embedder(texts):
        return [vec["pred"] if i == 0 else vec[t]
                for i, t in enumerate(texts)]

    scene = nodes(
        (2, "table", (0.0, 0.0, 0.0)),   # GT node
        (7, "sofa", (5.0, 0.0, 0.0)),    # same cosine, farther
        (5, "sofa", (1.0, 0.0, 0.0)),    # same cosine, closer: wins
    )
    res = lenient_match("pred", scene, gt_node_id=2,
                        embedder=embedder, synonyms=SYN)
    assert res.node_id == 5
    assert res.step == "embedding"
    assert res.tie_break == "cosine>centroid"

    # Identical centroids as well: lower node ID wins.
    scene2 = nodes(
        (2, "table", (0.0, 0.0, 0.0)),
        (7, "sofa", (1.0, 0.0, 0.0)),
        (5, "sofa", (1.0, 0.0, 0.0)),
    )
    res2 = lenient_match("pred", scene2, gt_node_id=2,
                         embedder=embedder, synonyms=SYN)
    assert res2.node_id == 5
    assert res2.tie_break == "cosine>centroid>node_id"


def test_lenient_exact_string_tiebreak_by_centroid():
    # Two same-label instances (distractor case): closer to GT wins.
    scene = nodes(
        (1, "lamp", (0.0, 0.0, 0.0)),    # GT node
        (8, "sofa", (4.0, 0.0, 0.0)),
        (6, "sofa", (1.0, 0.0, 0.0)),
    )
    res = lenient_match("couch", scene, gt_node_id=1, synonyms=SYN)
    assert res.node_id == 6
    assert res.step == "exact_string"
    assert res.tie_break == "centroid"


def test_lenient_embedder_absent_records_unavailable():
    scene = nodes((1, "sofa"), (2, "table"))
    res = lenient_match("ottoman", scene, gt_node_id=1,
                        embedder=None, synonyms=SYN)
    assert not res.matched and not res.correct
    assert res.node_id is None
    assert res.step == "embedding_unavailable"
    assert res.cosine is None


def test_lenient_is_deterministic():
    def embedder(texts):
        return [[1.0, 0.0] for _ in texts]

    scene = nodes(
        (9, "sofa", (2.0, 0.0, 0.0)),
        (3, "chair", (2.0, 0.0, 0.0)),
        (5, "table", (0.0, 0.0, 0.0)),
    )
    results = [
        lenient_match("bench", scene, gt_node_id=5,
                      embedder=embedder, synonyms=SYN)
        for _ in range(3)
    ]
    assert all(r == results[0] for r in results)
    # All cosines tie at 1.0; the GT node itself sits closest, so the
    # centroid leg decides.
    assert results[0].node_id == 5
    assert results[0].tie_break == "cosine>centroid"


# -------------------------------------------------------------- cosine helper

def test_cosine_similarity_basics():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
    with pytest.raises(ValueError):
        cosine_similarity([1.0], [1.0, 0.0])


# ---------------------------------------------------- frozen file + manifest

def test_frozen_synonyms_load_and_are_canonical_form():
    table = load_synonyms()
    assert "_meta" not in table
    assert 20 <= len(table) <= 40
    assert table["couch"] == "sofa"
    assert table["tv"] == "television"
    for key, val in table.items():
        # Keys and values must already be in canonical (identity) form
        # under the table itself, so canonicalization is idempotent.
        assert val == canonicalize(val)


def test_manifest_aux_files_pin_matches_synonyms_file():
    manifest_path = os.path.join(REPO_ROOT, "splits", "MANIFEST.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    entry = manifest["aux_files"]["synonyms_v1"]
    with open(SYNONYMS_PATH, "rb") as f:
        actual = hashlib.sha256(f.read()).hexdigest()
    assert entry["sha256"] == actual
    # The pre-existing split pins are untouched.
    assert "bench_v1_98" in manifest["splits"]

"""Tests for the canonical final-prediction schema (gate 1).

Covers: every legacy key, dict x/y/z form, numpy-like arrays via duck
typing, missing points, preservation of original fields, idempotency, and
realistic samples shaped like each planner's actual output.
"""

from src.schema.prediction import (
    CANONICAL_POINT_KEY,
    FinalPrediction,
    normalize_prediction,
)


class FakeArray:
    """Duck-typed stand-in for a numpy array: has tolist, is iterable."""

    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


# ---------------------------------------------------------------------------
# Legacy key mapping
# ---------------------------------------------------------------------------

def test_each_legacy_key_maps_to_canonical():
    legacy_keys = [
        "target_location",
        "target_xyz_hab",
        "selected_object_xyz",
        "selected_object_center_xyz",
        "selected_center_xyz",
    ]
    for key in legacy_keys:
        out = normalize_prediction({key: [1.0, 2.0, 3.0]})
        assert out[CANONICAL_POINT_KEY] == [1.0, 2.0, 3.0], key
        # legacy key survives untouched
        assert out[key] == [1.0, 2.0, 3.0], key


def test_canonical_key_wins_over_legacy():
    out = normalize_prediction({
        "target_point_xyz": [9.0, 9.0, 9.0],
        "target_location": [1.0, 2.0, 3.0],
    })
    assert out[CANONICAL_POINT_KEY] == [9.0, 9.0, 9.0]


def test_tuple_and_int_values_coerce_to_floats():
    out = normalize_prediction({"target_location": (1, 2, 3)})
    assert out[CANONICAL_POINT_KEY] == [1.0, 2.0, 3.0]
    assert all(isinstance(v, float) for v in out[CANONICAL_POINT_KEY])


# ---------------------------------------------------------------------------
# Value forms
# ---------------------------------------------------------------------------

def test_dict_xyz_form():
    out = normalize_prediction({"target_xyz_hab": {"x": 1.5, "y": -2.0, "z": 0.25}})
    assert out[CANONICAL_POINT_KEY] == [1.5, -2.0, 0.25]


def test_dict_xyz_form_uppercase_keys():
    out = normalize_prediction({"target_location": {"X": 1.0, "Y": 2.0, "Z": 3.0}})
    assert out[CANONICAL_POINT_KEY] == [1.0, 2.0, 3.0]


def test_numpy_like_array_via_duck_typing():
    out = normalize_prediction({"target_location": FakeArray([0.5, 1.5, 2.5])})
    assert out[CANONICAL_POINT_KEY] == [0.5, 1.5, 2.5]


def test_empty_list_falls_through_to_next_key():
    # vlm_planner_msp uses target_xyz_hab: [] as a deliberate no-point
    # sentinel; a later key should still be honored.
    out = normalize_prediction({
        "target_xyz_hab": [],
        "selected_object_xyz": [4.0, 5.0, 6.0],
    })
    assert out[CANONICAL_POINT_KEY] == [4.0, 5.0, 6.0]


# ---------------------------------------------------------------------------
# Missing and malformed input
# ---------------------------------------------------------------------------

def test_missing_point_yields_none_and_warning():
    out = normalize_prediction({"action_type": "answer", "chosen_id": "obj_1"})
    assert out[CANONICAL_POINT_KEY] is None
    assert any("no target point" in w for w in out["schema_warnings"])


def test_none_value_yields_none_and_warning():
    out = normalize_prediction({"target_location": None})
    assert out[CANONICAL_POINT_KEY] is None
    assert len(out["schema_warnings"]) >= 1


def test_malformed_values_never_raise():
    bad_inputs = [
        {"target_location": [1.0, 2.0]},
        {"target_location": "not a point"},
        {"target_location": {"x": 1.0}},
        {"target_location": ["a", "b", "c"]},
        {"target_location": 42},
        "not even a dict",
        None,
        [1.0, 2.0, 3.0],
    ]
    for bad in bad_inputs:
        out = normalize_prediction(bad)
        assert out[CANONICAL_POINT_KEY] is None, bad
        assert len(out["schema_warnings"]) >= 1, bad


# ---------------------------------------------------------------------------
# Field preservation and idempotency
# ---------------------------------------------------------------------------

def test_original_fields_preserved():
    pred = {
        "action_type": "goto_object",
        "chosen_id": "POINT_GUESS",
        "confidence": 0.95,
        "thought": "some reasoning",
        "pdf_params": {"theta": 0.1, "kappa": 4.0},
        "target_location": [1.0, 2.0, 3.0],
        "top_k_objects": [{"id": "o1", "name": "chair", "position": [0, 0, 0], "confidence": 0.5}],
    }
    out = normalize_prediction(pred)
    for key, value in pred.items():
        assert out[key] == value, key
    # input dict is not mutated
    assert CANONICAL_POINT_KEY not in pred


def test_idempotent_with_point():
    once = normalize_prediction({"target_xyz_hab": [1.0, 2.0, 3.0], "confidence": 0.8})
    twice = normalize_prediction(once)
    assert once == twice


def test_idempotent_without_point():
    once = normalize_prediction({"action_type": "lookaround", "chosen_id": ""})
    twice = normalize_prediction(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Real planner output shapes (copied from the planner code)
# ---------------------------------------------------------------------------

def test_multi_agent_msp_planner_shape():
    # Shape from multi_agent_msp_planner.py build_answer
    pred = {
        "action_type": "goto_object",
        "chosen_id": "POINT_GUESS",
        "confidence": 0.95,
        "thought": "Anchor is locked and spatial geometry is calculated.",
        "pdf_params": {"theta": 1.57, "phi": 1.57, "kappa": 8.0},
        "target_location": [2.31, 0.12, -4.55],
        "top_k_objects": [
            {"id": "object_12", "name": "table", "position": [2.0, 0.0, -4.0], "confidence": 0.61},
        ],
    }
    out = normalize_prediction(pred)
    assert out[CANONICAL_POINT_KEY] == [2.31, 0.12, -4.55]
    assert out["target_location"] == [2.31, 0.12, -4.55]
    assert out["top_k_objects"] == pred["top_k_objects"]


def test_vlm_planner_msp_shape_point_guess():
    # Shape from vlm_planner_msp.py selector plan when chosen_id == POINT_GUESS
    pred = {
        "thought": "The point guess matches the described location.",
        "action_type": "answer",
        "chosen_id": "POINT_GUESS",
        "target_xyz_hab": [1.02, 0.4, -3.3],
        "answer_text": "About one meter in front of the sofa.",
        "confidence": 0.92,
        "selector": {
            "mode": "where",
            "chosen_id": "POINT_GUESS",
            "answer_type": "point",
            "confidence": 0.92,
            "point_guess": {"id": "POINT_GUESS", "target_xyz_hab": [1.02, 0.4, -3.3], "msp_score": -0.7},
        },
    }
    out = normalize_prediction(pred)
    assert out[CANONICAL_POINT_KEY] == [1.02, 0.4, -3.3]
    assert out["target_xyz_hab"] == [1.02, 0.4, -3.3]
    assert out["selector"] == pred["selector"]


def test_vlm_planner_msp_shape_object_answer_no_point():
    # WHICH-mode answer: target_xyz_hab is the empty-list sentinel
    pred = {
        "thought": "Best object selected.",
        "action_type": "answer",
        "chosen_id": "object_7",
        "target_xyz_hab": [],
        "answer_text": "the lamp (id=object_7)",
        "confidence": 0.9,
    }
    out = normalize_prediction(pred)
    assert out[CANONICAL_POINT_KEY] is None
    assert out["target_xyz_hab"] == []
    assert any("no target point" in w for w in out["schema_warnings"])


def test_gemini_benchmark_shape_already_canonical():
    # Shape from vlm_planner_benchmark_gemini.py structured output
    pred = {
        "explanation_ans": "The point is left of the fridge.",
        "anchor_object_id": "object_3",
        "target_point_xyz": [0.5, 1.1, -2.2],
        "explanation_conf": "Clear view of the anchor.",
        "confidence_level": 0.85,
        "is_confident": True,
    }
    out = normalize_prediction(pred)
    assert out[CANONICAL_POINT_KEY] == [0.5, 1.1, -2.2]
    # already-canonical input gains no provenance warning
    assert "schema_warnings" not in out or out["schema_warnings"] == []
    # everything else untouched
    assert out["anchor_object_id"] == "object_3"
    assert out["is_confident"] is True


# ---------------------------------------------------------------------------
# Dataclass spec
# ---------------------------------------------------------------------------

def test_final_prediction_dataclass_round_trip():
    fp = FinalPrediction(
        action_type="answer",
        chosen_id="object_1",
        confidence=0.9,
        target_point_xyz=[1.0, 2.0, 3.0],
        thought="done",
    )
    d = fp.to_dict()
    out = normalize_prediction(d)
    assert out[CANONICAL_POINT_KEY] == [1.0, 2.0, 3.0]
    assert out["action_type"] == "answer"

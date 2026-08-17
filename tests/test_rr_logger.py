import json

from src.logging.rr_logger import simplify_step_transcript


def test_simplified_transcript_keeps_decisions_and_drops_reasoning():
    ledger = [
        {
            "agent": "Orchestrator",
            "type": "ParseQuery",
            "status": "PASS",
            "details": {
                "reasoning": "SECRET_INTERNAL_REASONING",
                "target_entity": "location",
                "composition_logic": "intrinsic_front",
                "anchors": [
                    {"label": "lamp", "modifiers": "floor", "metric": "1.5 meters"}
                ],
            },
        },
        {
            "agent": "Grounding",
            "type": "MatchObjects",
            "status": "PASS",
            "details": {
                "reasoning": "SECRET_INTERNAL_REASONING",
                "grounded_anchors": [
                    {
                        "anchor_label": "lamp",
                        "matched_object_id": "object_16",
                        "confidence": 0.95,
                    }
                ],
                "needs_exploration": False,
            },
        },
    ]
    decision = {
        "action_type": "goto_object",
        "chosen_id": "POINT_GUESS",
        "confidence": 0.95,
        "target_location": [1.0, 2.0, 3.0],
        "pdf_params": {"density_masked": True, "mass_in_tau_ball": 0.7},
        "thought": "SECRET_INTERNAL_REASONING",
    }

    summary = simplify_step_transcript(ledger, decision)
    rendered = json.dumps(summary)
    assert "SECRET_INTERNAL_REASONING" not in rendered
    assert "floor lamp 1.5 meters" in summary["roles"]["orchestrator"]
    assert "lamp -> object_16 (0.95)" in summary["roles"]["grounding"]
    assert "Navmesh masked: True" in summary["decision"]


def test_verifier_summary_marks_skipped_checks():
    summary = simplify_step_transcript(
        [
            {
                "agent": "Verifier",
                "type": "Critique",
                "status": "PASS",
                "details": {
                    "status": "PASS",
                    "reasoning": "not shown",
                    "checks": {
                        "schema_valid": {"ok": True, "skipped": False},
                        "on_navmesh": {"ok": True, "skipped": True},
                        "all_ok": True,
                    },
                },
            }
        ],
        {},
    )
    text = summary["roles"]["verifier"]
    assert "schema_valid: PASS" in text
    assert "on_navmesh: SKIP" in text

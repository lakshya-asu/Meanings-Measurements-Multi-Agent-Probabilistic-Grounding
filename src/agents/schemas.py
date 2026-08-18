"""Typed output schemas for every agent role (MAPG-09 / P1).

One JSON-schema dict per role is the single source of truth. The same
dict is (a) rendered verbatim into the role prompt, so every backend
sees a byte-identical schema block, and (b) used to validate every
parsed reply, so a malformed payload is a logged, counted failure
(``schema_invalid``), never a silent fallback.

Why a stdlib dict validator instead of pydantic:

- The host that runs the fast test suite has no pydantic; the
  container does (openai SDK dependency). Validation must behave
  identically in both places, so it cannot depend on an optional
  package.
- The schema text is embedded byte-for-byte in prompts and covered by
  golden snapshots (tests/test_golden_prompts.py). pydantic's
  ``model_json_schema()`` output changes across pydantic versions,
  which would silently change prompts between environments and break
  the byte-identical guarantee the backend factorial rests on.

Strictness is the superset of what any legacy family enforced:

- every declared field is required (openai ``.parse`` strictness; the
  legacy claude family validated nothing beyond ``json.loads``),
- categorical fields are enums (gemini ``protos.Schema`` strictness),
- numeric fields accept int/float and numeric strings, coerced to
  float (legacy claude ``float(...)`` coercion),
- extra keys are preserved, matching every legacy family.

The ``logical`` role is not defined here on purpose: it was
constructed by the planner and never called (EXPERIMENT_PLAN section 1
"no orphan components"); it is deleted with the legacy files.
"""

from typing import Any, Dict, List, Tuple

ROLES = ("orchestrator", "grounding", "spatial", "verifier", "qa")

COMPOSITION_LOGIC_VALUES = [
    "none",
    "near",
    "between",
    "intrinsic_front",
    "intrinsic_back",
    "intrinsic_left",
    "intrinsic_right",
]

ORCHESTRATOR_SCHEMA: Dict[str, Any] = {
    "title": "OrchestratorOutput",
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Think step-by-step about the grammar and spatial layout implied by the query.",
        },
        "target_entity": {
            "type": "string",
            "description": "The main object or area the user wants to find or answer a question about.",
        },
        "anchors": {
            "type": "array",
            "description": "List of reference objects used to locate the target.",
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "CRITICAL: The base noun ONLY (e.g., 'sofa', 'chair', 'table'). NO ADJECTIVES.",
                    },
                    "modifiers": {
                        "type": "string",
                        "description": "All adjectives, sizes, colors, and spatial hints (e.g., '2 seater', 'next to the wall', 'wooden'). DO NOT put explicit distances here.",
                    },
                    "metric": {
                        "type": "string",
                        "description": "Explicit distances or metrics (e.g., '3.0 meters'). Leave empty if none.",
                    },
                },
                "required": ["label", "modifiers", "metric"],
            },
        },
        "composition_logic": {
            "type": "string",
            "enum": COMPOSITION_LOGIC_VALUES,
            "description": "The spatial relationship between the target and the anchors.",
        },
        "requires_logical_reasoning": {
            "type": "boolean",
            "description": "Set to true if the question asks for a specific factual answer/complex deduction beyond just a target navigation location.",
        },
    },
    "required": [
        "reasoning",
        "target_entity",
        "anchors",
        "composition_logic",
        "requires_logical_reasoning",
    ],
}

GROUNDING_SCHEMA: Dict[str, Any] = {
    "title": "GroundingOutput",
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Explain your visual verification of the modifiers.",
        },
        "grounded_anchors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "anchor_label": {"type": "string"},
                    "matched_object_id": {
                        "type": "string",
                        "description": "The exact ID from the scene graph. Use 'NONE' if not found.",
                    },
                    "confidence": {"type": "number"},
                },
                "required": ["anchor_label", "matched_object_id", "confidence"],
            },
        },
        "needs_exploration": {
            "type": "boolean",
            "description": "True if a required anchor is missing from the scene graph AND the image.",
        },
    },
    "required": ["reasoning", "grounded_anchors", "needs_exploration"],
}

SPATIAL_SCHEMA: Dict[str, Any] = {
    "title": "SpatialOutput",
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "theta_radians": {
            "type": "number",
            "description": "Egocentric Azimuth of the intrinsic front vector",
        },
        "phi_radians": {
            "type": "number",
            "description": "Elevation of the intrinsic front vector. 0.0 rad = Up. 1.57 rad = Level. 3.14 rad = Down.",
        },
        "target_frontier_id": {
            "type": "string",
            "description": "If the object is visible but not grounded, output a frontier id towards the object. Otherwise 'NONE'.",
        },
    },
    "required": ["reasoning", "theta_radians", "phi_radians", "target_frontier_id"],
}

VERIFIER_SCHEMA: Dict[str, Any] = {
    "title": "VerifierOutput",
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "status": {"type": "string", "enum": ["PASS", "FAIL"]},
        "feedback": {
            "type": "string",
            "description": "If FAIL, explain what went wrong so the system can recover.",
        },
    },
    "required": ["reasoning", "status", "feedback"],
}

QA_SCHEMA: Dict[str, Any] = {
    "title": "QaOutput",
    "type": "object",
    "properties": {
        "prior_hypothesis": {
            "type": "string",
            "description": "Based on the question, formulate a prior hypothesis of what the expected answer or target location should be.",
        },
        "hypothesis_likelihood": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Given the current evidence in the scene graph and visual input, how likely is your prior hypothesis to be correct?",
        },
        "reasoning": {
            "type": "string",
            "description": "Break down the query to identify the required target object, map the answer choices to symbols, and deduce the logical next step.",
        },
        "action_type": {
            "type": "string",
            "enum": ["goto_object", "goto_frontier", "lookaround", "answer"],
            "description": "The action to take. 'goto_object' to go to a known object, 'goto_frontier' to explore for missing context, 'lookaround' to spin, 'answer' if the final answer is definitively known now.",
        },
        "chosen_id": {
            "type": "string",
            "description": "The Node ID of the object or frontier to navigate to. Use 'NONE' if action_type is lookaround or answer.",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence score between 0.0 and 1.0 of the chosen action or answer.",
        },
        "answer": {
            "type": "string",
            "enum": ["A", "B", "C", "D", "NONE"],
            "description": "If action_type is 'answer', provide EXACTLY the option symbol (A, B, C, or D) from the choices provided. Otherwise use 'NONE'.",
        },
    },
    "required": [
        "prior_hypothesis",
        "hypothesis_likelihood",
        "reasoning",
        "action_type",
        "chosen_id",
        "confidence",
        "answer",
    ],
}

SCHEMAS: Dict[str, Dict[str, Any]] = {
    "orchestrator": ORCHESTRATOR_SCHEMA,
    "grounding": GROUNDING_SCHEMA,
    "spatial": SPATIAL_SCHEMA,
    "verifier": VERIFIER_SCHEMA,
    "qa": QA_SCHEMA,
}


class SchemaError(ValueError):
    """A parsed payload does not satisfy its role schema.

    Roles catch this and return their error shape with a
    ``schema_invalid`` marker so the failure is logged and counted,
    never silently defaulted.
    """

    def __init__(self, errors: List[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _check(value: Any, spec: Dict[str, Any], path: str, errors: List[str]) -> Any:
    """Validate ``value`` against ``spec``; return the coerced value.

    Appends human-readable reasons to ``errors``. Never raises.
    """
    kind = spec.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            errors.append(f"{path}: expected object, got {type(value).__name__}")
            return value
        out = dict(value)  # extra keys preserved
        props = spec.get("properties", {})
        for key in spec.get("required", []):
            if key not in value:
                errors.append(f"{path}.{key}: required field missing")
        for key, sub in props.items():
            if key in value:
                out[key] = _check(value[key], sub, f"{path}.{key}", errors)
        return out
    if kind == "array":
        if not isinstance(value, list):
            errors.append(f"{path}: expected array, got {type(value).__name__}")
            return value
        item_spec = spec.get("items")
        if item_spec is None:
            return list(value)
        return [
            _check(item, item_spec, f"{path}[{i}]", errors)
            for i, item in enumerate(value)
        ]
    if kind == "string":
        if not isinstance(value, str):
            errors.append(f"{path}: expected string, got {type(value).__name__}")
            return value
        allowed = spec.get("enum")
        if allowed is not None and value not in allowed:
            errors.append(f"{path}: {value!r} not one of {allowed}")
        return value
    if kind == "number":
        if isinstance(value, bool):
            errors.append(f"{path}: expected number, got bool")
            return value
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
        errors.append(f"{path}: expected number, got {type(value).__name__}")
        return value
    if kind == "boolean":
        if not isinstance(value, bool):
            errors.append(f"{path}: expected boolean, got {type(value).__name__}")
        return value
    errors.append(f"{path}: unknown schema type {kind!r}")
    return value


def validate(role: str, payload: Any) -> Dict[str, Any]:
    """Validate a parsed reply against its role schema.

    Returns a coerced copy (numbers as float, extra keys preserved).
    Raises SchemaError with every reason when the payload does not
    conform; the caller records it as a counted schema_invalid failure.
    """
    role = str(role).lower()
    if role not in SCHEMAS:
        raise SchemaError([f"unknown role {role!r}"])
    errors: List[str] = []
    coerced = _check(payload, SCHEMAS[role], role, errors)
    if errors:
        raise SchemaError(errors)
    return coerced


def try_validate(role: str, payload: Any) -> Tuple[bool, Any, List[str]]:
    """Non-raising validate: (ok, coerced_or_original, errors)."""
    try:
        return True, validate(role, payload), []
    except SchemaError as e:
        return False, payload, e.errors

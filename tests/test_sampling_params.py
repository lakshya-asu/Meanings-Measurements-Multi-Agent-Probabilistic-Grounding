"""Model-aware sampling parameters on the Anthropic adapter (MAPG-25).

Anthropic models from Opus 4.7 onward reject a non-default temperature
(and top_p / top_k) with a 400. The pinned claude-opus-4-6 and the
claude-haiku-4-5 cheap tier accept it.

The load-bearing requirement is the negative one: what the PINNED arm
sends must not change. MAPG-09 unified the prompts so the backends are
byte-identical and the factorial is not confounded; quietly altering
the pinned request shape would undo that and invalidate anything
already collected. So these tests assert both directions, not just
that new models work.
"""

from src.agents.backends.claude import (
    ClaudeBackend,
    accepts_sampling_params,
)
from src.agents.base import DEFAULT_TEMPERATURE, text_part

SYSTEM = "system text"
USER = "user text"


def _request_for(model):
    return ClaudeBackend(model).build_request(SYSTEM, [text_part(USER)])


# ----------------------------------------------------------------------
# accepts_sampling_params
# ----------------------------------------------------------------------

def test_pinned_opus_accepts_temperature():
    assert accepts_sampling_params("claude-opus-4-6") is True


def test_haiku_cheap_tier_accepts_temperature():
    assert accepts_sampling_params("claude-haiku-4-5") is True


def test_dated_snapshot_resolves_to_its_family():
    # The cheap tier is pinned to a dated id, so family matching has to
    # see through the date suffix or the tiered arm would lose its
    # temperature the moment it is pinned.
    assert accepts_sampling_params("claude-haiku-4-5-20251001") is True


def test_current_generation_models_reject_temperature():
    for model in (
        "claude-opus-4-7",
        "claude-opus-4-8",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5",
    ):
        assert accepts_sampling_params(model) is False, model


def test_unknown_model_omits_rather_than_guesses():
    # Allow-list, not deny-list: an unreleased model we have never
    # heard of must fall on the safe side. Omitting temperature runs at
    # the provider default; sending it to a model that rejects it is a
    # hard 400 in the middle of a paid run.
    assert accepts_sampling_params("claude-something-unreleased") is False


def test_empty_and_none_are_safe():
    assert accepts_sampling_params(None) is False
    assert accepts_sampling_params("") is False
    assert accepts_sampling_params("   ") is False


def test_matching_is_case_insensitive():
    assert accepts_sampling_params("Claude-Opus-4-6") is True


# ----------------------------------------------------------------------
# build_request
# ----------------------------------------------------------------------

def test_pinned_arm_request_is_unchanged():
    # The regression that matters: the pinned arm must still send
    # exactly 0.1, not a default and not nothing.
    req = _request_for("claude-opus-4-6")
    assert req["temperature"] == DEFAULT_TEMPERATURE


def test_rejecting_model_omits_the_key_entirely():
    # Absent, not None and not some "default" value: there is no
    # temperature these models accept, so the key cannot be present.
    req = _request_for("claude-opus-5")
    assert "temperature" not in req
    assert "top_p" not in req
    assert "top_k" not in req


def test_omitting_temperature_changes_nothing_else():
    # Only the sampling key differs between the two arms. If this ever
    # fails, the adapter is varying more than MAPG-25 intended and the
    # backends are no longer byte-identical.
    pinned = _request_for("claude-opus-4-6")
    current = _request_for("claude-opus-5")
    assert set(pinned) - set(current) == {"temperature"}
    assert set(current) - set(pinned) == set()
    for key in current:
        if key == "model":
            continue
        assert current[key] == pinned[key], key


def test_system_and_cache_structure_survive_the_omission():
    # MAPG-10's cache breakpoint must not be collateral damage.
    req = _request_for("claude-opus-5")
    assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert req["system"][0]["text"] == SYSTEM
    assert req["messages"][0]["content"][0]["text"] == USER

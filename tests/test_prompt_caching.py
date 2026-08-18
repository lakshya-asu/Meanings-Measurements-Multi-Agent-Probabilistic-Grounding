"""Prompt-caching request structure and cache-usage extraction (MAPG-10).

The contract: caching annotations are REQUEST STRUCTURE, never text.
The claude adapter marks the system block and any cache-marked user
text part with ``cache_control: ephemeral``; openai/gemini requests
carry no cache keys (their prefix caching is provider-automatic); the
prompt bytes are identical either way (the golden suite asserts the
byte equality, this file asserts the structure).
"""

from types import SimpleNamespace

from src.agents.backends.claude import ClaudeBackend
from src.agents.backends.gemini import GeminiBackend
from src.agents.backends.openai_compat import OpenAICompatBackend
from src.agents.base import text_part
from src.agents.prompts import qa as qa_prompt
from src.agents.prompts import spatial as spatial_prompt
from src.results.calls import CallLog, extract_cache_usage
from tests.golden.context import (
    ANCHOR_OBJ,
    make_mcq_blackboard,
    make_where_blackboard,
)

SYSTEM = "system text"
STABLE = "stable scene graph prefix"
VOLATILE = "volatile pose and history"


# ----------------------------------------------------------------------
# render_parts: chunking is exactly a split of the render() text
# ----------------------------------------------------------------------

def test_spatial_render_parts_concatenates_to_render():
    bb = make_where_blackboard()
    system_joined, user_joined = spatial_prompt.render(bb, ANCHOR_OBJ)
    system, chunks = spatial_prompt.render_parts(bb, ANCHOR_OBJ)
    assert system == system_joined
    assert "".join(text for text, _ in chunks) == user_joined
    # First chunk (the scene-graph block) is the cacheable one.
    assert [cacheable for _, cacheable in chunks] == [True, False]
    assert bb.scene_graph_str in chunks[0][0]


def test_qa_render_parts_concatenates_to_render():
    bb = make_mcq_blackboard()
    system_joined, user_joined = qa_prompt.render(bb)
    system, chunks = qa_prompt.render_parts(bb)
    assert system == system_joined
    assert "".join(text for text, _ in chunks) == user_joined
    assert [cacheable for _, cacheable in chunks] == [True, False]
    # The stable chunk carries the episode-constant question and the
    # scene graph; the volatile chunk carries the per-step state.
    assert bb.question in chunks[0][0]
    assert bb.scene_graph_str in chunks[0][0]
    assert "GLOBAL FAILURE HISTORY" in chunks[1][0]


# ----------------------------------------------------------------------
# claude request structure
# ----------------------------------------------------------------------

def test_claude_system_block_carries_cache_control():
    request = ClaudeBackend().build_request(SYSTEM, [text_part(VOLATILE)])
    assert isinstance(request["system"], list)
    [block] = request["system"]
    assert block["text"] == SYSTEM
    assert block["cache_control"] == {"type": "ephemeral"}


def test_claude_marks_only_cache_marked_user_parts():
    request = ClaudeBackend().build_request(
        SYSTEM, [text_part(STABLE, cache=True), text_part(VOLATILE)]
    )
    stable_block, volatile_block = request["messages"][0]["content"]
    assert stable_block["text"] == STABLE
    assert stable_block["cache_control"] == {"type": "ephemeral"}
    assert volatile_block["text"] == VOLATILE
    assert "cache_control" not in volatile_block


def test_openai_and_gemini_requests_carry_no_cache_keys():
    parts = [text_part(STABLE, cache=True), text_part(VOLATILE)]
    for backend in (
        OpenAICompatBackend(provider="openai"),
        OpenAICompatBackend(provider="alibaba"),
        GeminiBackend(),
    ):
        request = backend.build_request(SYSTEM, parts)
        assert "cache" not in str(request)
        assert "cache_control" not in str(request)


# ----------------------------------------------------------------------
# cache-usage extraction (provider shapes; None never estimated)
# ----------------------------------------------------------------------

def test_extract_cache_usage_anthropic_shape():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=5000,
            output_tokens=250,
            cache_read_input_tokens=4100,
            cache_creation_input_tokens=900,
        )
    )
    assert extract_cache_usage(response) == (4100, 900)


def test_extract_cache_usage_openai_nested_shape():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=5000,
            completion_tokens=250,
            prompt_tokens_details=SimpleNamespace(cached_tokens=3800),
        )
    )
    assert extract_cache_usage(response) == (3800, None)


def test_extract_cache_usage_gemini_shape():
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=5000,
            candidates_token_count=250,
            cached_content_token_count=2700,
        )
    )
    assert extract_cache_usage(response) == (2700, None)


def test_extract_cache_usage_normalized_dict_shape():
    # The shape src.agents.base.usage_dict emits and roles pass along.
    result = {
        "ok": True,
        "usage": {
            "prompt_tokens": 5000,
            "completion_tokens": 250,
            "cache_read_tokens": 4100,
            "cache_write_tokens": 900,
        },
    }
    assert extract_cache_usage(result) == (4100, 900)


def test_extract_cache_usage_absent_is_none():
    assert extract_cache_usage(None) == (None, None)
    assert extract_cache_usage({"usage": {"prompt_tokens": 10}}) == (None, None)
    assert extract_cache_usage(object()) == (None, None)


def test_call_log_records_cache_fields():
    log = CallLog()
    log.call(
        "spatial",
        lambda: {
            "ok": True,
            "usage": {
                "prompt_tokens": 5000,
                "completion_tokens": 250,
                "cache_read_tokens": 4100,
                "cache_write_tokens": 900,
            },
        },
        model_name="claude-opus-4-6",
        step_idx=3,
    )
    [row] = log.rows()
    assert row["prompt_tokens"] == 5000
    assert row["cache_read_tokens"] == 4100
    assert row["cache_write_tokens"] == 900


def test_call_log_cache_fields_default_none():
    log = CallLog()
    log.call("qa", lambda: {"ok": True, "usage": {"prompt_tokens": 10,
                                                  "completion_tokens": 2}})
    [row] = log.rows()
    assert row["cache_read_tokens"] is None
    assert row["cache_write_tokens"] is None

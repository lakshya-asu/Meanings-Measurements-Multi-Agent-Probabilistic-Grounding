"""Tests for MAPG-02 per-call accounting (src/results/calls.py) and the
calls table in the results store.

Stdlib + pytest only. Fake agents stand in for the multi_agent classes
(which return parsed dicts without provider usage) and fake response
objects stand in for the Anthropic / OpenAI / Gemini SDK responses.
"""

from types import SimpleNamespace

import pytest

from src.results.calls import CallLog, extract_usage, model_name_of
from src.results.manifest import build_manifest
from src.results.store import ResultsStore


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------

class FakeDictAgent:
    """Like the multi_agent classes today: parsed dict out, provider
    response (and its usage) dropped on the floor."""

    def __init__(self, model_name="claude-opus-4-6", payload=None):
        self.model_name = model_name
        self.payload = payload if payload is not None else {"ok": True}
        self.invocations = 0

    def process(self, *args, **kwargs):
        self.invocations += 1
        return dict(self.payload)


class FakeGeminiStyleAgent:
    """Gemini family shape: no model_name attr, GenerativeModel on
    .model which exposes model_name."""

    def __init__(self):
        self.model = SimpleNamespace(model_name="models/gemini-2.5-pro")

    def process(self, *args, **kwargs):
        return {"ok": True}


class ExplodingAgent:
    def process(self, *args, **kwargs):
        raise RuntimeError("api down")


def anthropic_response(inp=1234, out=250):
    return SimpleNamespace(usage=SimpleNamespace(input_tokens=inp, output_tokens=out))


def openai_response(prompt=900, completion=120):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion))


def gemini_response(prompt=5000, candidates=300):
    return SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt,
            candidates_token_count=candidates,
            total_token_count=prompt + candidates,
        )
    )


# ----------------------------------------------------------------------
# extract_usage: what the providers actually expose
# ----------------------------------------------------------------------

def test_extract_usage_anthropic_shape():
    assert extract_usage(anthropic_response(1234, 250)) == (1234, 250)


def test_extract_usage_openai_shape():
    assert extract_usage(openai_response(900, 120)) == (900, 120)


def test_extract_usage_gemini_shape():
    assert extract_usage(gemini_response(5000, 300)) == (5000, 300)


def test_extract_usage_dict_with_usage_key():
    assert extract_usage({"usage": {"input_tokens": 10, "output_tokens": 2}}) == (10, 2)
    assert extract_usage({"usage": {"prompt_token_count": 7, "candidates_token_count": 3}}) == (7, 3)


def test_extract_usage_absent_is_none_never_estimated():
    assert extract_usage(None) == (None, None)
    assert extract_usage({"ok": True, "action_type": "answer"}) == (None, None)
    assert extract_usage(SimpleNamespace(text="{}")) == (None, None)
    # Malformed usage values coerce to None, not garbage.
    assert extract_usage({"usage": {"input_tokens": "abc"}}) == (None, None)


def test_model_name_of():
    assert model_name_of(FakeDictAgent("claude-opus-4-6")) == "claude-opus-4-6"
    assert model_name_of(FakeGeminiStyleAgent()) == "models/gemini-2.5-pro"
    assert model_name_of(object()) is None


# ----------------------------------------------------------------------
# CallLog against fake agent invocations
# ----------------------------------------------------------------------

def full_step(log, step_idx, is_retry=False, verifier_llm=False):
    """Reproduce the planner's 4-call 'where' path with anchor locked."""
    orch = FakeDictAgent("claude-opus-4-6")
    ground = FakeDictAgent("claude-opus-4-6")
    spatial = FakeDictAgent("claude-opus-4-6")
    verifier = FakeDictAgent("claude-opus-4-6", payload={"status": "PASS", "llm_used": verifier_llm})
    for role, agent in (("orchestrator", orch), ("grounding", ground), ("spatial", spatial)):
        log.call(role, agent.process, model_name=model_name_of(agent),
                 is_retry=is_retry, step_idx=step_idx)
    log.call(
        "verifier", verifier.process, model_name=model_name_of(verifier),
        is_retry=is_retry, step_idx=step_idx,
        record_if=lambda out: bool(isinstance(out, dict) and out.get("llm_used", False)),
    )


def test_mcq_step_is_one_call():
    log = CallLog()
    qa = FakeDictAgent("claude-opus-4-6", payload={"ok": True, "action_type": "answer"})
    out = log.call("qa", qa.process, model_name=model_name_of(qa), step_idx=1)
    assert out["ok"] is True
    assert qa.invocations == 1
    assert log.total() == 1
    rec = log.records()[0]
    assert rec.role == "qa"
    assert rec.model_name == "claude-opus-4-6"
    # Agents drop provider usage today (MAPG-09): None, never estimated.
    assert rec.prompt_tokens is None
    assert rec.completion_tokens is None
    assert rec.is_retry is False
    assert rec.step_idx == 1
    assert rec.latency_ms is not None and rec.latency_ms >= 0.0


def test_warmup_step_is_two_calls():
    log = CallLog()
    orch = FakeDictAgent()
    ground = FakeDictAgent()
    log.call("orchestrator", orch.process, step_idx=1)
    log.call("grounding", ground.process, step_idx=1)
    assert log.total() == 2
    assert [r.role for r in log.records()] == ["orchestrator", "grounding"]


def test_full_step_llm_verifier_off_is_three_calls():
    # Programmatic-only verification (llm_used False) is not an LLM call.
    log = CallLog()
    full_step(log, step_idx=1, verifier_llm=False)
    assert log.total() == 3
    assert [r.role for r in log.records()] == ["orchestrator", "grounding", "spatial"]


def test_full_step_llm_verifier_on_is_four_calls():
    log = CallLog()
    full_step(log, step_idx=1, verifier_llm=True)
    assert log.total() == 4
    assert log.records()[-1].role == "verifier"


def test_retry_steps_are_flagged_and_counted():
    # Step 1 rejected by the verifier; steps 2 and 3 are the bounded
    # retries (max_verifier_retries = 2 in the planner).
    log = CallLog()
    full_step(log, step_idx=1, is_retry=False, verifier_llm=True)
    full_step(log, step_idx=2, is_retry=True, verifier_llm=True)
    full_step(log, step_idx=3, is_retry=True, verifier_llm=True)
    assert log.total() == 12  # retries included in the rollup
    assert log.retries() == 8
    assert all(r.is_retry for r in log.records() if r.step_idx in (2, 3))
    assert not any(r.is_retry for r in log.records() if r.step_idx == 1)


def test_exception_is_recorded_and_reraised():
    log = CallLog()
    agent = ExplodingAgent()
    with pytest.raises(RuntimeError, match="api down"):
        log.call("spatial", agent.process, step_idx=4)
    assert log.total() == 1
    rec = log.records()[0]
    assert rec.role == "spatial"
    assert rec.prompt_tokens is None and rec.completion_tokens is None
    assert rec.step_idx == 4


def test_usage_captured_when_result_exposes_it():
    # The vlm_planner_msp choke points hand the raw Gemini response to
    # extract_usage; simulate that path through CallLog.call.
    log = CallLog()
    log.call("kernel", lambda: gemini_response(5000, 300),
             model_name="models/gemini-2.5-pro", step_idx=0)
    rec = log.records()[0]
    assert rec.prompt_tokens == 5000
    assert rec.completion_tokens == 300


def test_rows_are_store_ready_dicts():
    log = CallLog()
    full_step(log, step_idx=1, verifier_llm=True)
    rows = log.rows()
    assert len(rows) == 4
    assert rows[0]["role"] == "orchestrator"
    assert set(rows[0]) == {
        "role", "model_name", "prompt_tokens", "completion_tokens",
        "is_retry", "latency_ms", "step_idx",
    }


# ----------------------------------------------------------------------
# Runner rollup arithmetic
# ----------------------------------------------------------------------

def test_runner_rollup_matches_mixed_step_paths():
    """The runner does ep_vlm_calls = call_log.total() after every
    get_next_action; the final value must equal the sum of the true
    per-step counts (2 + 2 + 4 + 4-retry + 1), not steps * 4."""
    log = CallLog()
    per_step_expected = []

    # two warmup steps, 2 calls each (orchestrator + grounding, no anchor)
    for step in (1, 2):
        log.call("orchestrator", FakeDictAgent().process, step_idx=step)
        log.call("grounding", FakeDictAgent().process, step_idx=step)
        per_step_expected.append(2)
    # one full step, then one verifier-retry step
    full_step(log, step_idx=3, verifier_llm=True)
    per_step_expected.append(4)
    full_step(log, step_idx=4, is_retry=True, verifier_llm=True)
    per_step_expected.append(4)
    # MCQ fast path step
    log.call("qa", FakeDictAgent().process, step_idx=5)
    per_step_expected.append(1)

    ep_vlm_calls = log.total()
    assert ep_vlm_calls == sum(per_step_expected) == 13
    assert ep_vlm_calls != 5 * 4  # the old fiction
    assert log.retries() == 4

    # Cumulative per-step reads are monotone and end at the total.
    cumulative = []
    running = CallLog()
    for step, n in enumerate(per_step_expected, start=1):
        for _ in range(n):
            running.record("x", step_idx=step)
        cumulative.append(running.total())
    assert cumulative == [2, 4, 8, 12, 13]
    assert cumulative[-1] == ep_vlm_calls


# ----------------------------------------------------------------------
# Store round-trip
# ----------------------------------------------------------------------

CFG = {
    "seed": 7,
    "split": "bench_v1_98",
    "vlm": {"name": "gemini", "use_image": True},
}


def make_store(tmp_path):
    store = ResultsStore(tmp_path / "results.sqlite")
    run_id = store.start_run(build_manifest(CFG, seed=7, split_name="bench_v1_98"))
    return store, run_id


def test_store_calls_roundtrip(tmp_path):
    store, run_id = make_store(tmp_path)
    log = CallLog()
    full_step(log, step_idx=1, verifier_llm=True)
    log.call("kernel", lambda: gemini_response(5000, 300),
             model_name="models/gemini-2.5-pro", step_idx=2)

    store.record_calls(run_id, "0_scene_0", log.rows())
    rows = store.calls(run_id, "0_scene_0")
    assert len(rows) == 5
    assert [r["call_idx"] for r in rows] == [0, 1, 2, 3, 4]
    assert rows[0]["role"] == "orchestrator"
    assert rows[0]["model_name"] == "claude-opus-4-6"
    assert rows[0]["prompt_tokens"] is None
    assert rows[0]["is_retry"] == 0
    assert rows[4]["role"] == "kernel"
    assert rows[4]["prompt_tokens"] == 5000
    assert rows[4]["completion_tokens"] == 300
    assert rows[4]["step_idx"] == 2
    store.close()


def test_store_calls_replace_idempotent(tmp_path):
    store, run_id = make_store(tmp_path)
    log = CallLog()
    full_step(log, step_idx=1, verifier_llm=True)
    store.record_calls(run_id, "0_scene_0", log.rows())
    # Re-record with fewer rows: old rows must not linger.
    short = CallLog()
    short.call("qa", FakeDictAgent().process, step_idx=1)
    store.record_calls(run_id, "0_scene_0", short.rows())
    rows = store.calls(run_id, "0_scene_0")
    assert len(rows) == 1
    assert rows[0]["role"] == "qa"
    store.close()


def test_store_calls_requires_registered_run(tmp_path):
    store = ResultsStore(tmp_path / "results.sqlite")
    with pytest.raises(RuntimeError, match="never registered"):
        store.record_calls("no_such_run", "0_scene_0", [])
    store.close()


def test_episode_vlm_calls_matches_calls_table(tmp_path):
    """End-to-end rollup arithmetic: the vlm_calls episode column is the
    CallLog total, which equals the number of calls-table rows."""
    store, run_id = make_store(tmp_path)
    log = CallLog()
    full_step(log, step_idx=1, verifier_llm=True)
    full_step(log, step_idx=2, is_retry=True, verifier_llm=True)

    ep_vlm_calls = log.total()
    store.record_episode(run_id, {
        "qid": "0_scene_0",
        "method": "multi_agent",
        "backend": "claude",
        "split": "bench_v1_98",
        "seed": 7,
        "success": True,
        "num_steps": 2,
        "vlm_calls": ep_vlm_calls,
        "vlm_retry_calls": log.retries(),
    })
    store.record_calls(run_id, "0_scene_0", log.rows())

    ep = store.episodes(run_id)[0]
    rows = store.calls(run_id, "0_scene_0")
    assert ep["vlm_calls"] == len(rows) == 8
    assert ep["final"]["vlm_retry_calls"] == sum(r["is_retry"] for r in rows) == 4
    # Per-run reader covers every episode's rows in order.
    assert store.calls(run_id) == rows
    store.close()

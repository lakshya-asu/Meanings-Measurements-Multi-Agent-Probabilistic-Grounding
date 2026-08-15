"""Tests for the gate 4 results store and run manifest.

Stdlib + pytest only. Numpy is stubbed with duck-typed stand-ins when
it is not installed, so the store's tolist / default=str fallbacks are
exercised either way.
"""

import json
import os

import pytest

from src.results.manifest import (
    ManifestError,
    build_manifest,
    config_hash,
    write_manifest,
)
from src.results.store import ResultsStore
from src.splits import split_sha

try:
    import numpy as np

    def np_array(values):
        return np.asarray(values, dtype=np.float32)

    def np_float(value):
        return np.float32(value)

    def np_int(value):
        return np.int64(value)

except ImportError:
    # Duck-typed stand-ins with the same surface the store relies on.
    class _FakeArray:
        def __init__(self, values):
            self._values = [float(v) for v in values]

        def tolist(self):
            return list(self._values)

    class _FakeScalar:
        """Not JSON-serializable, but float()-able, like numpy scalars."""

        def __init__(self, value):
            self._value = value

        def __float__(self):
            return float(self._value)

        def __int__(self):
            return int(self._value)

        def item(self):
            return self._value

        def __repr__(self):
            return f"fake({self._value})"

    def np_array(values):
        return _FakeArray(values)

    def np_float(value):
        return _FakeScalar(float(value))

    def np_int(value):
        return _FakeScalar(int(value))


CFG = {
    "seed": 7,
    "split": "bench_v1_98",
    "vlm": {"name": "gemini", "use_image": True},
    "model_pins": {"gemini": "gemini-2.5-pro-001"},
}


def make_manifest():
    return build_manifest(CFG, seed=7, split_name="bench_v1_98")


def episode_row(qid="0_scene_0"):
    return {
        "qid": qid,
        "experiment_id": qid,
        "scene": "00001-scene",
        "method": "msp_smart",
        "backend": "gemini",
        "split": "bench_v1_98",
        "seed": 7,
        "success": True,
        "num_steps": np_int(12),
        "vlm_calls": 12,
        "traj_length": np_float(3.25),
        "episode_time_sec": 41.5,
        "final_confidence": np_float(0.9375),
        "target_point_xyz": np_array([1.5, 0.25, -2.0]),
        "final_pred": {"action_type": "answer", "confidence": np_float(0.9375)},
    }


def test_manifest_contents():
    m = make_manifest()
    assert len(m["git_sha"]) == 40
    assert all(c in "0123456789abcdef" for c in m["git_sha"])
    assert m["git_sha"].startswith(m["git_sha_short"])
    assert isinstance(m["git_dirty"], bool)
    assert m["git_branch"]
    assert m["split"] == "bench_v1_98"
    assert m["split_sha256"] == split_sha("bench_v1_98")
    assert len(m["config_sha256"]) == 64
    assert m["seed"] == 7
    assert m["model_pins"] == {"gemini": "gemini-2.5-pro-001"}
    assert m["run_id"].endswith("_s7")
    assert m["git_sha_short"] in m["run_id"]


def test_config_hash_is_canonical():
    a = {"x": 1, "y": {"b": 2, "a": 3}}
    b = {"y": {"a": 3, "b": 2}, "x": 1}
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash({"x": 1, "y": {"b": 2, "a": 4}})


def test_unpinned_split_gets_none_sha_and_warning():
    m = build_manifest(CFG, seed=0, split_name="unknown")
    assert m["split_sha256"] is None
    assert any("unknown" in w for w in m["warnings"])


def test_write_manifest_atomic(tmp_path):
    m = make_manifest()
    out = write_manifest(m, tmp_path)
    assert out == tmp_path / "run.json"
    with open(out) as f:
        loaded = json.load(f)
    assert loaded["run_id"] == m["run_id"]
    # No temp or partial files left behind.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]


def test_record_episode_roundtrip_with_numpy_values(tmp_path):
    store = ResultsStore(tmp_path / "results.sqlite")
    m = make_manifest()
    run_id = store.start_run(m)

    store.record_episode(run_id, episode_row())
    eps = store.episodes(run_id)
    assert len(eps) == 1
    ep = eps[0]

    assert ep["qid"] == "0_scene_0"
    assert ep["method"] == "msp_smart"
    assert ep["backend"] == "gemini"
    assert ep["split"] == "bench_v1_98"
    assert ep["seed"] == 7
    assert ep["success"] == 1
    assert ep["num_steps"] == 12
    assert ep["traj_length"] == pytest.approx(3.25)
    assert ep["final_confidence"] == pytest.approx(0.9375)
    assert ep["target_x"] == pytest.approx(1.5)
    assert ep["target_y"] == pytest.approx(0.25)
    assert ep["target_z"] == pytest.approx(-2.0)
    # Full dict survives in the JSON blob, numpy values included.
    assert ep["final"]["scene"] == "00001-scene"
    assert ep["final"]["final_pred"]["action_type"] == "answer"
    store.close()


def test_insert_or_replace_idempotent(tmp_path):
    store = ResultsStore(tmp_path / "results.sqlite")
    run_id = store.start_run(make_manifest())

    row = episode_row()
    store.record_episode(run_id, row)
    row2 = dict(row)
    row2["success"] = False
    row2["final_confidence"] = 0.1
    store.record_episode(run_id, row2)

    eps = store.episodes(run_id)
    assert len(eps) == 1
    assert eps[0]["success"] == 0
    assert eps[0]["final_confidence"] == pytest.approx(0.1)
    store.close()


def test_record_without_start_run_raises(tmp_path):
    store = ResultsStore(tmp_path / "results.sqlite")
    with pytest.raises(RuntimeError, match="never registered"):
        store.record_episode("no_such_run", episode_row())
    store.close()


def test_runs_and_status_lifecycle(tmp_path):
    store = ResultsStore(tmp_path / "results.sqlite")
    m = make_manifest()
    run_id = store.start_run(m)
    assert store.run_status(run_id) == "running"

    store.record_episode(run_id, episode_row())
    store.finish_run(run_id, "complete")
    assert store.run_status(run_id) == "complete"

    runs = store.runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["manifest"]["config_sha256"] == m["config_sha256"]
    store.close()


def test_git_hard_fail_when_git_missing(tmp_path, monkeypatch):
    # An empty PATH means subprocess cannot find git: must raise, not
    # silently continue like the old wandb path did.
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ManifestError, match="git"):
        build_manifest(CFG, seed=0, split_name="bench_v1_98")

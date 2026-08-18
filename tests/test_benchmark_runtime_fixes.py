from types import SimpleNamespace

import numpy as np

from src.scripts import run_multi_agent_benchmark as runner
from src.envs.habitat_interface import _sanitize_instance_labels


class _Agent:
    def get_state(self):
        return SimpleNamespace(position=np.array([1.0, 2.0, 3.0]))


class _Sim:
    def get_agent(self, _index):
        return _Agent()


class _Habitat:
    _sim = _Sim()

    def get_heading_angle(self):
        return 0.25

    def get_init_poses_eqa(self, position, yaw_deg, pitch):
        return [(position.tolist(), yaw_deg, pitch)]


def test_warmup_passes_habitat_run_arguments_by_keyword(monkeypatch, tmp_path):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(runner, "run", fake_run)
    objects = [object() for _ in range(5)]
    runner._warmup_like_eqa(
        objects[0],
        _Habitat(),
        objects[1],
        objects[2],
        objects[3],
        tmp_path,
        segmenter=objects[4],
        save_image=True,
        num_yaws=1,
    )

    args, kwargs = calls[0]
    assert len(args) == 3
    assert kwargs["segmenter"] is objects[4]
    assert kwargs["output_path"] == tmp_path
    assert kwargs["rr_logger"] is objects[1]
    assert kwargs["tsdf_planner"] is objects[2]
    assert kwargs["sg_sim"] is objects[3]
    assert kwargs["save_image"] is True


def test_step_budget_is_shared_across_warmup_main_and_confirmation():
    budget = runner._StepBudget(10)

    assert all(budget.take() for _ in range(4))  # warmup
    assert all(budget.take() for _ in range(5))  # main loop
    assert budget.take() is True                 # confirmation
    assert budget.used == 10
    assert budget.take() is False
    assert budget.used == 10


def test_sparse_hm3d_instance_ids_fall_back_to_unknown():
    labels = np.array([[0, 356, 357, 399]], dtype=np.int32)

    safe = _sanitize_instance_labels(labels, label_count=357)

    assert safe.tolist() == [[0, 356, 0, 0]]
    assert labels.tolist() == [[0, 356, 357, 399]]


def test_valid_hm3d_instance_ids_are_unchanged():
    labels = np.array([[0, 1, 356]], dtype=np.int32)

    safe = _sanitize_instance_labels(labels, label_count=357)

    assert safe is labels


def test_resume_manifest_keeps_run_id_and_records_code_segment(
    monkeypatch, tmp_path
):
    existing = {
        "run_id": "original-run",
        "seed": 42,
        "split": "bench_v1_98",
        "split_sha256": "split-sha",
        "config_sha256": "config-sha",
        "git_sha": "old-sha",
    }
    (tmp_path / "run.json").write_text(__import__("json").dumps(existing))
    fresh = {
        **existing,
        "run_id": "new-run-that-must-not-replace-original",
        "start_time_utc": "2026-08-18T00:00:00+00:00",
        "git_sha": "new-sha",
        "git_sha_short": "newshort",
        "git_branch": "main",
        "git_dirty": False,
        "renderer": {"mode": "software"},
    }
    monkeypatch.setattr(runner, "build_manifest", lambda *a, **k: fresh)

    resumed = runner._build_or_resume_manifest(
        {}, tmp_path, run_seed=42, run_split="bench_v1_98", resume=True
    )

    assert resumed["run_id"] == "original-run"
    assert resumed["git_sha"] == "old-sha"
    assert resumed["resume_segments"] == [{
        "resume_index": 1,
        "reason": "resume_after_abort",
        "start_time_utc": "2026-08-18T00:00:00+00:00",
        "git_sha": "new-sha",
        "git_sha_short": "newshort",
        "git_branch": "main",
        "git_dirty": False,
        "config_sha256": "config-sha",
        "renderer": {"mode": "software"},
    }]


def test_resume_archives_partial_episode_directory(tmp_path):
    partial = tmp_path / "95_scene_0"
    partial.mkdir()
    (partial / "partial.rrd").write_bytes(b"partial")

    archived = runner._archive_partial_episode_dir(tmp_path, partial.name)

    assert not partial.exists()
    assert archived.parent == tmp_path / "aborted_attempts"
    assert archived.name.startswith("95_scene_0_")
    assert (archived / "partial.rrd").read_bytes() == b"partial"


def test_resume_archive_is_noop_without_partial_directory(tmp_path):
    assert runner._archive_partial_episode_dir(tmp_path, "95_scene_0") is None
    assert not (tmp_path / "aborted_attempts").exists()


def test_resume_restores_prior_spend_before_new_calls():
    prior_calls = [
        {
            "model_name": "claude-opus-4-6",
            "prompt_tokens": 1_000_000,
            "completion_tokens": 100_000,
        }
    ]

    class _Store:
        def calls(self, run_id):
            assert run_id == "original-run"
            return prior_calls

    class _Governor:
        def __init__(self):
            self.rows = []

        def charge_rows(self, rows):
            self.rows.extend(rows)

        def summary(self):
            return {"total_spend_usd": 7.5, "calls_charged": len(self.rows)}

    manifest = {"resume_segments": [{}]}
    governor = _Governor()

    summary = runner._restore_resume_spend(
        governor, _Store(), "original-run", manifest, resume=True
    )

    assert governor.rows == prior_calls
    assert summary == {"total_spend_usd": 7.5, "calls_charged": 1}
    assert manifest["resume_segments"][-1] == {
        "prior_calls_charged": 1,
        "prior_spend_usd": 7.5,
    }

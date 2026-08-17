from types import SimpleNamespace

import numpy as np

from src.scripts import run_multi_agent_benchmark as runner


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

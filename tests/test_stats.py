"""Tests for the item 9 statistics package (src/stats).

Everything is seeded; every assertion is deterministic. The synthetic
data is built with known structure so each statistical property is
checked in a case where the right answer is obvious.
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.results.store import ResultsStore
from src.stats import endpoints as ep
from src.stats import report
from src.stats.bootstrap import bca_interval
from src.stats.permutation import cluster_permutation_test, holm_bonferroni


# ----------------------------------------------------------------------
# Censoring at exactly C = 10 m
# ----------------------------------------------------------------------
def test_censoring_applied_at_exactly_10m():
    assert ep.censor_error(12.3) == 10.0
    assert ep.censor_error(10.0 + 1e-9) == 10.0
    assert ep.censor_error(10.0) == 10.0
    assert ep.censor_error(9.99) == 9.99
    assert ep.censor_error(0.0) == 0.0
    # Missing or invalid errors are censored failures, never dropped.
    assert ep.censor_error(None) == 10.0
    assert ep.censor_error("not a number") == 10.0
    assert ep.censor_error(float("nan")) == 10.0
    assert ep.censor_error(-1.0) == 10.0
    # Custom cap flows through.
    assert ep.censor_error(7.0, cap=5.0) == 5.0


def test_censoring_does_not_touch_success():
    # A success at 0.5 m stays a success; a row over the cap is a
    # failure by threshold, not by censoring.
    near = {"success_gt_1m": True, "d_h": 0.5}
    far = {"success_gt_1m": False, "d_h": 12.0}
    missing = {"success_gt_1m": None, "d_h": None}
    assert ep.extract_value(near, "success_1m") == 1.0
    assert ep.extract_value(near, "censored_error") == 0.5
    assert ep.extract_value(far, "success_1m") == 0.0
    assert ep.extract_value(far, "censored_error") == 10.0
    # Missing prediction: failure with censored error.
    assert ep.extract_value(missing, "success_1m") == 0.0
    assert ep.extract_value(missing, "censored_error") == 10.0
    # Missing row entirely (None final): same rule.
    assert ep.extract_value(None, "success_1m") == 0.0
    assert ep.extract_value(None, "censored_error") == 10.0


# ----------------------------------------------------------------------
# Hierarchical BCa bootstrap
# ----------------------------------------------------------------------
def _synthetic_scene_values(rng, n_scenes=20, n_queries=5, mean=2.0):
    scene_values = {}
    for s in range(n_scenes):
        scene_effect = rng.normal(0.0, 0.2)
        vals = mean + scene_effect + rng.normal(0.0, 0.5, size=n_queries)
        scene_values[f"scene{s:02d}"] = vals
    return scene_values


def test_bootstrap_ci_covers_true_mean():
    rng_data = np.random.default_rng(7)
    scene_values = _synthetic_scene_values(rng_data, mean=2.0)
    rng = np.random.default_rng(ep.DEFAULT_ANALYSIS_SEED)
    res = bca_interval(
        scene_values, lambda m: float(np.mean(m[:, 0])), n_boot=2000, rng=rng
    )
    assert res.method == "bca"
    assert res.lo < 2.0 < res.hi
    assert res.lo < res.point < res.hi
    assert res.n_scenes == 20
    assert res.n_queries == 100


def test_bootstrap_ci_covers_true_success_rate():
    # Per-query success fractions in {0, 1/3, 2/3, 1} with a known
    # overall rate of 0.5, balanced within every scene so the true
    # cell value is unambiguous.
    scene_values = {
        f"s{k:02d}": np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]) for k in range(15)
    }
    rng = np.random.default_rng(123)
    res = bca_interval(
        scene_values, lambda m: float(np.mean(m[:, 0])), n_boot=2000, rng=rng
    )
    assert res.lo < 0.5 < res.hi


def test_bootstrap_degenerate_jackknife_falls_back_to_percentile():
    # Constant data: the jackknife over scenes has zero variance, so
    # the acceleration is undefined and the documented percentile
    # fallback must engage.
    scene_values = {f"s{k}": np.full(3, 4.2) for k in range(6)}
    rng = np.random.default_rng(0)
    res = bca_interval(
        scene_values, lambda m: float(np.mean(m[:, 0])), n_boot=500, rng=rng
    )
    assert res.method == "percentile"
    assert res.fallback_reason is not None
    assert "jackknife" in res.fallback_reason
    assert res.lo == res.hi == res.point == 4.2


def test_bootstrap_requires_explicit_rng():
    with pytest.raises(ValueError):
        bca_interval({"s": np.array([1.0])}, lambda m: 0.0, rng=None)


def test_bootstrap_deterministic_given_seed():
    scene_values = _synthetic_scene_values(np.random.default_rng(1))
    stat = lambda m: float(np.mean(m[:, 0]))
    r1 = bca_interval(scene_values, stat, n_boot=500, rng=np.random.default_rng(42))
    r2 = bca_interval(scene_values, stat, n_boot=500, rng=np.random.default_rng(42))
    assert (r1.lo, r1.hi, r1.point) == (r2.lo, r2.hi, r2.point)


# ----------------------------------------------------------------------
# Cluster permutation test
# ----------------------------------------------------------------------
def _paired_layout(n_scenes=20, n_queries=3):
    scenes = []
    for s in range(n_scenes):
        scenes.extend([f"scene{s:02d}"] * n_queries)
    return scenes


def test_permutation_p_small_when_effect_huge():
    scenes = _paired_layout()
    n = len(scenes)
    a = np.ones(n)
    b = np.zeros(n)
    rng = np.random.default_rng(5)
    res = cluster_permutation_test(a, b, scenes, "mean_diff", n_perm=2000, rng=rng)
    assert res.t_obs == 1.0
    assert res.p_value < 0.01


def test_permutation_p_is_one_when_systems_identical():
    scenes = _paired_layout()
    rng_data = np.random.default_rng(9)
    a = rng_data.normal(2.0, 1.0, size=len(scenes))
    b = a.copy()
    rng = np.random.default_rng(6)
    res = cluster_permutation_test(a, b, scenes, "mean_diff", n_perm=1000, rng=rng)
    # Every permuted statistic equals the observed 0, so p is exactly 1.
    assert res.t_obs == 0.0
    assert res.p_value == 1.0


def test_permutation_p_large_under_null_noise():
    scenes = _paired_layout()
    rng_data = np.random.default_rng(11)
    base = rng_data.normal(2.0, 1.0, size=len(scenes))
    a = base + rng_data.normal(0.0, 0.3, size=len(scenes))
    b = base + rng_data.normal(0.0, 0.3, size=len(scenes))
    rng = np.random.default_rng(12)
    res = cluster_permutation_test(a, b, scenes, "mean_diff", n_perm=2000, rng=rng)
    assert res.p_value > 0.05


def test_permutation_median_statistic_huge_effect():
    # Distinct continuous values: partial swaps strictly shrink the
    # median gap, so only the identity and full flips reproduce
    # |T_obs| and the p-value is tiny.
    scenes = _paired_layout()
    n = len(scenes)
    a = 0.3 + 0.01 * np.arange(n)  # small errors
    b = 7.0 + 0.01 * np.arange(n)  # large errors
    rng = np.random.default_rng(13)
    res = cluster_permutation_test(a, b, scenes, "median_diff", n_perm=2000, rng=rng)
    assert res.t_obs == pytest.approx(-6.7)
    assert res.p_value < 0.01


def test_permutation_requires_explicit_rng():
    with pytest.raises(ValueError):
        cluster_permutation_test([1.0], [0.0], ["s"], rng=None)


# ----------------------------------------------------------------------
# Holm-Bonferroni on a hand-worked 5-p-value example
# ----------------------------------------------------------------------
def test_holm_hand_worked_example():
    # Raw ps in input order; sorted: 0.005, 0.01, 0.03, 0.04, 0.2 with
    # multipliers 5, 4, 3, 2, 1 giving 0.025, 0.04, 0.09, 0.08, 0.2 and
    # monotone cummax 0.025, 0.04, 0.09, 0.09, 0.2.
    p = [0.01, 0.04, 0.03, 0.005, 0.2]
    out = holm_bonferroni(p, alpha=0.05)
    assert out["adjusted"] == pytest.approx([0.04, 0.09, 0.09, 0.025, 0.2])
    # Step-down: 0.005 <= 0.05/5, 0.01 <= 0.05/4, then 0.03 > 0.05/3.
    assert out["reject"] == [True, False, False, True, False]
    assert out["family_size"] == 5


def test_holm_family_size_held_at_preregistered_k():
    # Only 3 of the K=5 comparisons evaluable: multipliers still start
    # at 5 (conservative).
    out = holm_bonferroni([0.01, 0.002, 0.5], alpha=0.05, family_size=5)
    # sorted: 0.002*5=0.01, 0.01*4=0.04, 0.5*3=1.0 (capped)
    assert out["adjusted"] == pytest.approx([0.04, 0.01, 1.0])
    assert out["reject"] == [True, True, False]
    with pytest.raises(ValueError):
        holm_bonferroni([0.1, 0.2], family_size=1)


# ----------------------------------------------------------------------
# No dropped queries: the assertion fires on a missing qid
# ----------------------------------------------------------------------
def test_missing_qid_assertion_fires():
    qids = ["0_sA_1", "1_sA_1", "2_sB_1", "3_sB_1"]
    seeds = [0, 1, 2]
    present = [(q, s) for q in qids for s in seeds]
    present.remove(("2_sB_1", 1))
    with pytest.raises(ep.MissingQueriesError) as exc:
        ep.assert_no_dropped_queries("mapg", present, qids, seeds)
    msg = str(exc.value)
    assert "2_sB_1" in msg
    assert "seed 1" in msg
    assert "mapg" in msg
    # Complete design passes and reports the full count.
    ok = ep.assert_no_dropped_queries("mapg", [(q, s) for q in qids for s in seeds], qids, seeds)
    assert ok["ok"] and ok["expected_rows"] == 12


def test_value_matrix_refuses_missing_rows_by_default():
    qids = ["0_sA_1", "1_sA_1"]
    seeds = [0, 1, 2]
    rows = {(q, s): {"success_gt_1m": True, "d_h": 0.2} for q in qids for s in seeds}
    del rows[("1_sA_1", 2)]
    with pytest.raises(ep.MissingQueriesError):
        ep.value_matrix(rows, qids, seeds, "success_1m")
    # Explicit downgrade: the missing row scores as a censored failure.
    m = ep.value_matrix(rows, qids, seeds, "censored_error", allow_missing_rows=True)
    assert m[1, 2] == 10.0
    s = ep.value_matrix(rows, qids, seeds, "success_1m", allow_missing_rows=True)
    assert s[1, 2] == 0.0


# ----------------------------------------------------------------------
# End-to-end report on a synthetic database
# ----------------------------------------------------------------------
# 12 scenes gives 4096 distinct scene sign-flip patterns, so the
# cluster-permutation p can get small enough for Holm (K=5) to reject
# at alpha 0.05. With too few scenes the minimum attainable p is
# 2 / 2^n_scenes and nothing can ever be significant.
_N_SCENES = 12
_QUERIES_PER_SCENE = 2
_SEEDS = (0, 1, 2)
_SPLIT_NAME = "toy_v1"


def _write_toy_split(tmp: Path):
    """Synthetic frozen split: 6 scenes x 3 queries, SHA-pinned."""
    splits_dir = tmp / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    csv_path = splits_dir / "toy.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene", "floor"])
        for s in range(_N_SCENES):
            for _ in range(_QUERIES_PER_SCENE):
                w.writerow([f"scene{s:02d}", "1"])
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest_path = splits_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "splits": {
                    _SPLIT_NAME: {
                        "name": _SPLIT_NAME,
                        "path": "splits/toy.csv",
                        "sha256": sha,
                        "rows": _N_SCENES * _QUERIES_PER_SCENE,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _populate_db(db_path: Path, skip_pair=None):
    """3 methods x 3 seeds x 24 queries with known structure.

    mapg is much better than grapheqa_adapted; anchor_centroid is worst.
    Values are deterministic functions of (method, seed, query index).
    """
    profiles = {
        "mapg": 0.3,  # d_h base: mostly under 1 m
        "grapheqa_adapted": 2.5,  # mostly over 1 m
        "anchor_centroid": 6.0,
    }
    with ResultsStore(db_path) as store:
        for method, base in profiles.items():
            for seed in _SEEDS:
                run_id = f"run_{method}_s{seed}"
                store.start_run(
                    {"run_id": run_id, "seed": seed, "split": _SPLIT_NAME,
                     "method": method}
                )
                i = 0
                for s in range(_N_SCENES):
                    for _ in range(_QUERIES_PER_SCENE):
                        qid = f"{i}_scene{s:02d}_1"
                        if skip_pair == (qid, seed, method):
                            i += 1
                            continue
                        # Deterministic wobble, different per seed/query.
                        d_h = base + 0.1 * ((i * 7 + seed * 3) % 5) / 5.0
                        row = {
                            "qid": qid,
                            "method": method,
                            "backend": "test",
                            "split": _SPLIT_NAME,
                            "seed": seed,
                            "scene": f"scene{s:02d}",
                            "d_h": d_h,
                            "success_gt_1m": d_h <= 1.0,
                            "success_gt_node": None,
                        }
                        store.record_episode(run_id, row)
                        i += 1
                store.finish_run(run_id)


def _report_args(tmp: Path, json_name: str):
    return [
        "--db", str(tmp / "results.sqlite"),
        "--split", _SPLIT_NAME,
        "--split-manifest", str(tmp / "splits" / "MANIFEST.json"),
        "--n-boot", "300",
        "--n-perm", "500",
        "--json-out", str(tmp / json_name),
        "--text-out", str(tmp / (json_name + ".txt")),
    ]


def test_report_end_to_end_and_byte_identical_json(tmp_path, capsys):
    _write_toy_split(tmp_path)
    _populate_db(tmp_path / "results.sqlite")

    rc1 = report.main(_report_args(tmp_path, "out1.json"))
    rc2 = report.main(_report_args(tmp_path, "out2.json"))
    capsys.readouterr()
    assert rc1 == 0 and rc2 == 0

    b1 = (tmp_path / "out1.json").read_bytes()
    b2 = (tmp_path / "out2.json").read_bytes()
    # Determinism contract: same db + same analysis seed gives a
    # byte-identical JSON summary.
    assert b1 == b2

    summary = json.loads(b1)
    # All 5 preregistered comparisons are present, in rank order.
    assert [c["id"] for c in summary["comparisons"]] == ["C1", "C2", "C3", "C4", "C5"]
    by_id = {c["id"]: c for c in summary["comparisons"]}
    # C1/C2/C4 evaluable on this db; C3 (O-O) and C5 (SPL) have no
    # scored rows, so they are reported as not evaluable, never dropped
    # silently and never faked as p-values.
    assert by_id["C1"]["status"] == "evaluated"
    assert by_id["C2"]["status"] == "evaluated"
    assert by_id["C4"]["status"] == "evaluated"
    assert by_id["C3"]["status"] == "not_evaluable"
    assert by_id["C5"]["status"] == "not_evaluable"
    # Huge preregistered effects: raw and Holm-adjusted ps are small.
    assert by_id["C1"]["p_raw"] < 0.05
    assert by_id["C1"]["p_holm"] < 0.05
    assert by_id["C1"]["reject_at_alpha"] is True
    # mapg beats grapheqa on SR (positive delta) and on error (negative).
    assert by_id["C1"]["t_obs"] > 0
    assert by_id["C2"]["t_obs"] < 0
    # Holm multipliers started at the preregistered family size 5.
    assert summary["analysis"]["holm_family_size"] == 5

    # Cell sanity: mapg SR is 1.0, anchor_centroid SR is 0.0, and the
    # censored median for anchor_centroid reflects its 6 m errors.
    cells = summary["cells"]
    assert cells["mapg"]["sr_1m"]["estimate"]["point"] == 1.0
    assert cells["anchor_centroid"]["sr_1m"]["estimate"]["point"] == 0.0
    assert cells["anchor_centroid"]["med_dh"]["estimate"]["point"] > 5.0
    # Integrity: every method complete, 24 x 3 = 72 rows.
    for method in ("mapg", "grapheqa_adapted", "anchor_centroid"):
        assert summary["integrity"][method]["ok"] is True
        assert summary["integrity"][method]["expected_rows"] == 72

    # The text summary exists and names the report.
    text = (tmp_path / "out1.json.txt").read_text(encoding="utf-8")
    assert "MAPG preregistered statistical report" in text
    assert "C1" in text


def test_report_fails_loudly_on_missing_row(tmp_path, capsys):
    _write_toy_split(tmp_path)
    # Drop one (qid, seed) row for mapg: the report must refuse.
    _populate_db(tmp_path / "results.sqlite", skip_pair=("4_scene02_1", 1, "mapg"))
    rc = report.main(_report_args(tmp_path, "bad.json"))
    captured = capsys.readouterr()
    assert rc == 2
    assert "4_scene02_1" in captured.err
    assert not (tmp_path / "bad.json").exists()

    # With the explicit downgrade flag it runs, scores the missing row
    # as a censored failure, and says so.
    rc = report.main(_report_args(tmp_path, "ok.json") + ["--allow-incomplete"])
    capsys.readouterr()
    assert rc == 0
    summary = json.loads((tmp_path / "ok.json").read_bytes())
    assert summary["integrity"]["mapg"]["ok"] is False
    assert summary["integrity"]["mapg"]["missing_qids"] == ["4_scene02_1"]
    assert any("censored failures" in w for w in summary["warnings"])
    # mapg loses exactly the one seed vote on that query: SR drops from
    # 1.0 by (1/3)/24.
    sr = summary["cells"]["mapg"]["sr_1m"]["estimate"]["point"]
    assert sr == pytest.approx(1.0 - (1.0 / 3.0) / 24.0)

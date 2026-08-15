"""Unit tests for gate 3: path resolution, env parsing, pin enforcement.

Pure logic only. No habitat, no real dataset, no network.
"""

import os
from pathlib import Path

import pytest

from src.paths import REPO_ROOT, data_root, resolve_data_path
from src.scripts.preflight import (
    backends_for,
    collect_aliases,
    missing_env_backends,
    parse_env_file,
    pinned_aliases_needed,
    unpinned_aliases,
)


# ---------------------------------------------------------------------------
# resolve_data_path
# ---------------------------------------------------------------------------

class TestResolveDataPath:
    def test_datasets_path_remaps_to_repo_when_container_path_absent(self, monkeypatch):
        monkeypatch.delenv("MAPG_DATA_ROOT", raising=False)
        # Use a path that exists nowhere, host or container, so the
        # remap branch is exercised in both environments.
        missing = "/datasets/__no_such_dir__/foo.csv"
        assert not os.path.exists("/datasets/__no_such_dir__")
        out = resolve_data_path(missing)
        assert out == str(REPO_ROOT / "datasets" / "__no_such_dir__" / "foo.csv")

    def test_env_override_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MAPG_DATA_ROOT", str(tmp_path))
        out = resolve_data_path("/datasets/hm3d/train")
        assert out == str(tmp_path / "hm3d" / "train")
        assert data_root() == tmp_path

    def test_bare_datasets_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MAPG_DATA_ROOT", str(tmp_path))
        assert resolve_data_path("/datasets") == str(tmp_path)

    def test_non_datasets_absolute_path_untouched(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MAPG_DATA_ROOT", raising=False)
        p = str(tmp_path / "somewhere" / "else.csv")
        assert resolve_data_path(p) == p

    def test_datasets_prefix_lookalike_untouched(self, monkeypatch):
        # /datasets-extra is not under the /datasets mount.
        monkeypatch.delenv("MAPG_DATA_ROOT", raising=False)
        assert resolve_data_path("/datasets-extra/x.csv") == "/datasets-extra/x.csv"

    def test_relative_path_resolves_against_repo_root(self, monkeypatch):
        monkeypatch.delenv("MAPG_DATA_ROOT", raising=False)
        out = resolve_data_path("splits/bench_v1_98.csv")
        assert out == str(REPO_ROOT / "splits" / "bench_v1_98.csv")
        # The frozen split really is there.
        assert Path(out).is_file()

    def test_accepts_pathlib_input(self, monkeypatch):
        monkeypatch.delenv("MAPG_DATA_ROOT", raising=False)
        out = resolve_data_path(Path("splits") / "bench_v1_98.csv")
        assert out.endswith("bench_v1_98.csv")


# ---------------------------------------------------------------------------
# parse_env_file
# ---------------------------------------------------------------------------

class TestParseEnvFile:
    def test_basic_and_quotes_and_comments(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "# comment\n"
            "\n"
            "PLAIN=abc\n"
            "SINGLE='sq value'\n"
            'DOUBLE="dq value"\n'
            "export EXPORTED=yes\n"
            "EMPTY=\n"
            "not a kv line\n"
            "WITH_EQ=a=b=c\n",
            encoding="utf-8",
        )
        out = parse_env_file(env)
        assert out["PLAIN"] == "abc"
        assert out["SINGLE"] == "sq value"
        assert out["DOUBLE"] == "dq value"
        assert out["EXPORTED"] == "yes"
        assert out["EMPTY"] == ""
        assert out["WITH_EQ"] == "a=b=c"
        assert "not a kv line" not in out

    def test_missing_file_returns_empty(self, tmp_path):
        assert parse_env_file(tmp_path / "nope.env") == {}


# ---------------------------------------------------------------------------
# alias collection and pin enforcement
# ---------------------------------------------------------------------------

CFG = {
    "vlm": {"name": "gemini", "answer_mode": "msp_point"},
    "habitat": {"scene_type": "hm3d", "dataset_type": "train"},
    "model_pins": {
        "claude-opus-4-6": "",
        "gpt-5.2-chat-latest": "",
        "qwen3-vl-plus": "",
    },
    "notes": ["neutral", "where"],
}


class TestAliases:
    def test_collects_pin_keys_and_backend_names_only(self):
        aliases = collect_aliases(CFG)
        assert aliases == {
            "gemini",
            "claude-opus-4-6",
            "gpt-5.2-chat-latest",
            "qwen3-vl-plus",
        }
        # Ordinary config words must not be mistaken for aliases.
        assert "hm3d" not in aliases
        assert "train" not in aliases
        assert "neutral" not in aliases

    def test_bare_backend_names_do_not_need_pins(self):
        need = pinned_aliases_needed({"gemini", "claude-opus-4-6"})
        assert need == {"claude-opus-4-6"}

    def test_empty_pins_are_flagged(self):
        aliases = collect_aliases(CFG)
        bad = unpinned_aliases(aliases, CFG["model_pins"])
        assert bad == ["claude-opus-4-6", "gpt-5.2-chat-latest", "qwen3-vl-plus"]

    def test_filled_pins_pass(self):
        pins = {
            "claude-opus-4-6": "claude-opus-4-6-20260115",
            "gpt-5.2-chat-latest": "gpt-5.2-2026-01-01",
            "qwen3-vl-plus": "qwen3-vl-plus-2025-12-30",
        }
        assert unpinned_aliases(collect_aliases(CFG), pins) == []

    def test_missing_pin_entry_is_flagged(self):
        assert unpinned_aliases({"claude-opus-4-6"}, {}) == ["claude-opus-4-6"]
        assert unpinned_aliases({"claude-opus-4-6"}, None) == ["claude-opus-4-6"]

    def test_whitespace_pin_is_flagged(self):
        assert unpinned_aliases({"claude-opus-4-6"}, {"claude-opus-4-6": "  "}) == [
            "claude-opus-4-6"
        ]


class TestEnvBackends:
    def test_backend_mapping(self):
        assert backends_for({"claude-opus-4-6"}) == {"claude"}
        assert backends_for({"gpt-5.2-chat-latest"}) == {"gpt"}
        assert backends_for({"qwen3-vl-plus"}) == {"qwen"}
        assert backends_for({"gemini"}) == {"gemini"}
        assert backends_for({"hm3d", "train"}) == set()

    def test_missing_keys_reported_per_backend(self):
        aliases = {"claude-opus-4-6", "gpt-5.2-chat-latest", "qwen3-vl-plus", "gemini"}
        env = {"OPENAI_API_KEY": "sk-x", "GOOGLE_API_KEY": "g-x"}
        missing = missing_env_backends(aliases, env)
        assert [b for b, _ in missing] == ["claude", "qwen"]
        claude_keys = dict(missing)["claude"]
        assert "CLAUDE_API_KEY" in claude_keys
        assert "ANTHROPIC_API_KEY" in claude_keys

    def test_any_key_in_group_satisfies(self):
        env = {"ANTHROPIC_API_KEY": "sk-ant"}
        assert missing_env_backends({"claude-opus-4-6"}, env) == []
        env = {"CLAUDE_API_KEY": "sk-c"}
        assert missing_env_backends({"claude-opus-4-6"}, env) == []

    def test_empty_or_whitespace_value_does_not_satisfy(self):
        env = {"DASHSCOPE_API_KEY": "   "}
        missing = missing_env_backends({"qwen3-vl-plus"}, env)
        assert [b for b, _ in missing] == ["qwen"]

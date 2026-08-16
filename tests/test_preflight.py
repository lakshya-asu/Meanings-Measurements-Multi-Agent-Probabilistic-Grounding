"""Unit tests for gate 3: path resolution, env parsing, pin enforcement.

Pure logic only. No habitat, no real dataset, no network.
"""

import os
from pathlib import Path

import pytest

from src.paths import REPO_ROOT, data_root, resolve_data_path
from src.scripts.preflight import (
    backends_for,
    bare_backend_price_gap,
    collect_aliases,
    missing_env_backends,
    parse_env_file,
    pinned_aliases_needed,
    selected_aliases,
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
        # These pin values are synthetic fixtures exercising the
        # mechanism, not real snapshot ids. In particular
        # claude-opus-4-6 has NO dated variant: the alias IS the
        # complete pinned id, and the real cfg pins it to itself. Do
        # not copy the shape below into cfg.
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

    def test_pin_value_counts_as_pinned(self):
        # collect_aliases picks up pin VALUES as well as keys, so a
        # correctly pinned snapshot id used to appear as a fresh
        # unpinned alias and pinning could never converge: filling in
        # one pin manufactured the next failure out of its own value.
        pins = {"claude-haiku-4-5": "claude-haiku-4-5-20251001"}
        aliases = {"claude-haiku-4-5", "claude-haiku-4-5-20251001"}
        assert unpinned_aliases(aliases, pins) == []

    def test_pin_value_rule_does_not_excuse_a_genuinely_unpinned_alias(self):
        # The self-pin allowance must not leak: an alias nobody pinned
        # is still flagged even when other pins resolve fine.
        pins = {"claude-haiku-4-5": "claude-haiku-4-5-20251001"}
        aliases = {"claude-haiku-4-5-20251001", "gpt-5.2-chat-latest"}
        assert unpinned_aliases(aliases, pins) == ["gpt-5.2-chat-latest"]


# ---------------------------------------------------------------------------
# selected_aliases: gate on the arm actually being run
# ---------------------------------------------------------------------------

class TestSelectedAliases:
    def test_reads_vlm_name(self):
        assert selected_aliases({"vlm": {"name": "claude-opus-4-6"}}) == {
            "claude-opus-4-6"
        }

    def test_includes_non_null_model_tiers(self):
        cfg = {
            "vlm": {
                "name": "claude-opus-4-6",
                "model_tiers": {
                    "orchestrator": "claude-haiku-4-5-20251001",
                    "grounding": None,
                    "spatial": None,
                },
            }
        }
        assert selected_aliases(cfg) == {
            "claude-opus-4-6",
            "claude-haiku-4-5-20251001",
        }

    def test_null_like_tier_values_are_ignored(self):
        cfg = {
            "vlm": {
                "name": "claude-opus-4-6",
                "model_tiers": {"a": None, "b": "", "c": "null", "d": "none"},
            }
        }
        assert selected_aliases(cfg) == {"claude-opus-4-6"}

    def test_unselected_backends_are_excluded(self):
        # The whole point: the cfg documents every arm of the
        # factorial permanently, so a claude run must not be gated on
        # OpenAI, Google and DashScope keys just because their pins and
        # prices live in the same file.
        cfg = dict(CFG)
        cfg["vlm"] = {"name": "claude-opus-4-6"}
        wide = collect_aliases(cfg)
        assert "gpt-5.2-chat-latest" in wide
        assert "qwen3-vl-plus" in wide
        assert selected_aliases(cfg) == {"claude-opus-4-6"}

    def test_returns_empty_when_selection_unreadable(self):
        # Empty means "cannot tell", and the caller falls back to the
        # conservative whole-cfg walk. It must never mean "nothing is
        # selected, so check nothing".
        assert selected_aliases({}) == set()
        assert selected_aliases({"vlm": None}) == set()
        assert selected_aliases({"vlm": "not-a-mapping"}) == set()
        assert selected_aliases({"vlm": {"name": ""}}) == set()

    def test_non_model_vlm_name_is_not_an_alias(self):
        assert selected_aliases({"vlm": {"name": "msp_point"}}) == set()


# ---------------------------------------------------------------------------
# bare_backend_price_gap: the false green that used to report PASS
# ---------------------------------------------------------------------------

class TestBareBackendPriceGap:
    def test_bare_backend_alone_cannot_be_price_checked(self):
        # The regression. A bare selector names no concrete model, so
        # nothing can be priced, and check (h) previously said PASS
        # while verifying nothing at all.
        gap = bare_backend_price_gap({"gemini"})
        assert gap is not None
        assert "gemini" in gap
        assert "no concrete model" in gap

    def test_concrete_model_has_no_gap(self):
        assert bare_backend_price_gap({"claude-opus-4-6"}) is None

    def test_bare_plus_concrete_has_no_gap(self):
        # One concrete model is enough to make the price check
        # meaningful, so a bare name alongside it is not a gap.
        assert bare_backend_price_gap({"claude", "claude-opus-4-6"}) is None

    def test_no_backends_at_all_is_not_a_gap(self):
        # Nothing selected means the governor is not required; that is
        # a different condition and must not be reported as this one.
        assert bare_backend_price_gap(set()) is None
        assert bare_backend_price_gap({"msp_point"}) is None

    def test_gap_names_the_provider_not_the_alias(self):
        # The message has to say which cap key is unverifiable, since
        # that is what the reader has to go fix.
        assert "openai" in bare_backend_price_gap({"gpt"})


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

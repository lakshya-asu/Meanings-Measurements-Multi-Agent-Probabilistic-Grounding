#!/usr/bin/env python3
"""Preflight checks for benchmark runs (gate 3 of the research harness).

Run as:

    python3 -m src.scripts.preflight --cfg mapg_benchmark

Works both inside the Habitat container and on a bare host checkout.
Every check prints PASS, FAIL, SKIP, or INFO with a specific message and
the command exits nonzero if anything FAILs, so a broken setup is caught
before any API budget is burned.

Checks:
  (a) config file loads and its data paths resolve
  (b) frozen split exists and SHA-verifies via src/splits.py
  (c) every scene in the split exists under the HM3D scenes dir and has
      semantic annotations (this replaces the runners' silent skipping)
  (d) scene init poses file exists and covers the split's scene_floor pairs
  (e) required API env keys present for the backends the cfg selects
  (f) every model alias the cfg uses has a non-empty pin in model_pins
  (g) numpy / habitat_sim importability (FAIL inside the container,
      INFO outside)
  (h) cost governor (MAPG-11): cost_caps present with a cap for every
      enabled provider plus 'total', and model_prices pins usable
      $/Mtok rows for the capped models

Dependencies: stdlib. Uses OmegaConf or PyYAML for the cfg if available
and python-dotenv for .env if available; degrades to manual parsing.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from src.paths import REPO_ROOT, resolve_data_path
from src.results.governor import normalize_price_row, provider_of, resolve_price_key

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"
INFO = "INFO"

# Model alias prefixes mapped to the env keys that back them. For each
# backend, ANY listed key being non-empty satisfies the check.
BACKEND_ENV_KEYS = {
    "claude": ["CLAUDE_API_KEY", "ANTHROPIC_API_KEY"],
    "gpt": ["OPENAI_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    "qwen": ["DASHSCOPE_API_KEY"],
}

# Strings that select a backend without naming a concrete model snapshot.
# These need env keys (check e) but not a pin (check f).
BARE_BACKEND_NAMES = {"claude", "gpt", "openai", "gemini", "google", "qwen"}

# Allow a digit or separator right after the prefix (qwen3-vl-plus,
# gpt-5.2-chat-latest) but not arbitrary letters (so 'training' or
# 'googleplex' are not mistaken for aliases).
_ALIAS_RE = re.compile(r"^(claude|gpt|gemini|google|qwen|o[134])(?:[-_.\d].*)?$")

SEM_LIST_BASENAME = "train-semantic-annots-files.json"


# ---------------------------------------------------------------------------
# Pure helpers (unit tested in tests/test_preflight.py)
# ---------------------------------------------------------------------------

def parse_env_file(path) -> dict:
    """Parse a .env file into a dict. Manual fallback for python-dotenv.

    Handles blank lines, comments, optional 'export ' prefixes, and
    single or double quoted values. Lines without '=' are ignored.
    """
    out = {}
    path = Path(path)
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key:
            out[key] = val
    return out


def load_env(repo_root=REPO_ROOT) -> dict:
    """Merge .env values under the live environment (live env wins)."""
    env_path = Path(repo_root) / ".env"
    try:
        from dotenv import dotenv_values
        file_vals = {k: (v or "") for k, v in dotenv_values(env_path).items()}
    except ImportError:
        file_vals = parse_env_file(env_path)
    merged = dict(file_vals)
    merged.update(os.environ)
    return merged


def collect_aliases(node) -> set:
    """Walk a plain cfg structure and collect model alias looking strings.

    Also includes the keys of any 'model_pins' mapping, since those are
    by definition aliases the run intends to use.
    """
    found = set()

    def walk(x, key=None):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "model_pins" and isinstance(v, dict):
                    found.update(str(a) for a in v.keys())
                walk(v, key=k)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            s = x.strip()
            if s and _ALIAS_RE.match(s):
                found.add(s)

    walk(node)
    return found


def selected_aliases(cfg) -> set:
    """The model aliases this run will actually use, or an empty set.

    A run picks its backend with vlm.name, plus any non-null
    vlm.model_tiers override. collect_aliases deliberately walks the
    WHOLE cfg, which is right for auditing but wrong for gating: this
    cfg permanently documents every arm of the factorial, so pins and
    prices for gpt, gemini and qwen sit in the file even when nobody
    is running them. Gating on that walk means every single-arm run
    demands every other provider's API key, pin and price, and a
    claude-only smoke run can never pass no matter what is filled in.

    Returning an empty set means the selection could not be read; the
    caller then falls back to the conservative whole-cfg walk, so a
    cfg shape this function does not understand fails loudly rather
    than checking nothing.
    """
    vlm = cfg.get("vlm") or {}
    if not isinstance(vlm, dict):
        return set()
    out = set()

    name = str(vlm.get("name") or "").strip()
    if name and _ALIAS_RE.match(name):
        out.add(name)

    tiers = vlm.get("model_tiers") or {}
    if isinstance(tiers, dict):
        for v in tiers.values():
            if v is None:
                continue
            s = str(v).strip()
            if s and s.lower() not in ("none", "null") and _ALIAS_RE.match(s):
                out.add(s)

    return out


def pinned_aliases_needed(aliases) -> set:
    """Aliases that must carry a pin: concrete model names, not bare
    backend selectors like 'gemini'."""
    return {a for a in aliases if a not in BARE_BACKEND_NAMES}


def unpinned_aliases(aliases, model_pins) -> list:
    """Return sorted aliases whose pin is missing or empty.

    An alias that is itself the resolved value of some pin counts as
    pinned. Without that, pinning can never converge: collect_aliases
    also picks up pin VALUES, so writing the correct pin
    claude-haiku-4-5 -> claude-haiku-4-5-20251001 immediately creates a
    brand new unpinned alias out of the snapshot id it just pinned to.
    A pinned snapshot id is the most pinned thing in the file.
    """
    pins = model_pins or {}
    resolved = {
        str(v).strip() for v in pins.values()
        if v is not None and str(v).strip()
    }
    bad = []
    for a in sorted(pinned_aliases_needed(aliases)):
        if a in resolved:
            continue
        pin = pins.get(a)
        if pin is None or not str(pin).strip():
            bad.append(a)
    return bad


def backends_for(aliases) -> set:
    """Map aliases to backend prefixes ('claude', 'gpt', ...)."""
    out = set()
    for a in aliases:
        m = _ALIAS_RE.match(a)
        if not m:
            continue
        prefix = m.group(1)
        if prefix.startswith("o") and prefix not in BACKEND_ENV_KEYS:
            prefix = "gpt"  # o-series models are OpenAI
        out.add(prefix)
    return out


def missing_env_backends(aliases, env) -> list:
    """Return [(backend, keys_tried)] for backends with no usable key."""
    missing = []
    for backend in sorted(backends_for(aliases)):
        keys = BACKEND_ENV_KEYS.get(backend, [])
        if not keys:
            continue
        if not any(str(env.get(k, "")).strip() for k in keys):
            missing.append((backend, keys))
    return missing


# Preflight backend prefixes -> cost governor provider cap keys.
_BACKEND_TO_PROVIDER = {"gpt": "openai", "google": "gemini"}


def cost_governor_problems(cfg, aliases=None) -> list:
    """Problems with cfg cost_caps / model_prices (MAPG-11); [] when
    the enabled backends are fully governable. Pure, unit tested.

    Requires: a positive USD cap for every enabled provider plus
    'total', numeric price rows, and a usable price row for every
    concrete model alias (or its pinned snapshot) whose provider is
    capped. Unpriceable spend cannot be governed.
    """
    problems = []
    aliases = collect_aliases(cfg) if aliases is None else set(aliases)
    providers = {_BACKEND_TO_PROVIDER.get(b, b) for b in backends_for(aliases)}
    if not providers:
        return problems

    caps = cfg.get("cost_caps") if isinstance(cfg, dict) else None
    if not isinstance(caps, dict) or not caps:
        problems.append(
            "cost_caps missing: add per-provider USD caps plus 'total' "
            "(no LLM run without the cost governor)"
        )
        caps = {}
    else:
        for key, value in caps.items():
            try:
                if float(value) < 0:
                    raise ValueError
            except (TypeError, ValueError):
                problems.append(f"cost_caps['{key}'] is not a usable USD "
                                f"amount: {value!r}")
        for p in sorted(providers):
            if p not in caps:
                problems.append(f"cost_caps has no cap for enabled "
                                f"provider '{p}'")
        if "total" not in caps:
            problems.append("cost_caps has no 'total' cap")

    prices = cfg.get("model_prices") if isinstance(cfg, dict) else None
    prices = prices if isinstance(prices, dict) else {}
    parsed = {}
    for model, row in prices.items():
        try:
            normalize_price_row(row)
            parsed[model] = row
        except ValueError as e:
            problems.append(f"model_prices['{model}']: {e}")

    pins = cfg.get("model_pins") if isinstance(cfg, dict) else None
    pins = pins if isinstance(pins, dict) else {}
    for alias in sorted(pinned_aliases_needed(aliases)):
        provider = _BACKEND_TO_PROVIDER.get(provider_of(alias), provider_of(alias))
        if provider is None or (caps and provider not in caps and "total" not in caps):
            continue
        candidates = [alias]
        pin = pins.get(alias)
        if pin is not None and str(pin).strip():
            candidates.append(str(pin).strip())
        if not any(resolve_price_key(parsed, c) is not None for c in candidates):
            problems.append(
                f"model_prices has no usable row for '{alias}': "
                "unpriceable spend cannot be governed"
            )
    return problems


def in_container() -> bool:
    """Detect the Habitat container via /.dockerenv or the /workspace mount."""
    return os.path.exists("/.dockerenv") or os.path.isdir("/workspace")


# ---------------------------------------------------------------------------
# Cfg loading
# ---------------------------------------------------------------------------

def load_cfg(cfg_arg: str):
    """Load src/cfg/<name>.yaml (or an explicit path) as a plain dict.

    Uses OmegaConf if available (without resolving interpolations, which
    is fine because preflight only reads literal fields), else PyYAML.
    """
    if cfg_arg.endswith((".yaml", ".yml")):
        cfg_path = Path(resolve_data_path(cfg_arg))
    else:
        cfg_path = REPO_ROOT / "src" / "cfg" / f"{cfg_arg}.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(cfg_path)
        return OmegaConf.to_container(cfg, resolve=False), cfg_path
    except ImportError:
        pass
    try:
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            return yaml.safe_load(f), cfg_path
    except ImportError:
        raise RuntimeError(
            "neither OmegaConf nor PyYAML is importable; install one of "
            "them to run preflight"
        )


# ---------------------------------------------------------------------------
# Check runner
# ---------------------------------------------------------------------------

class Reporter:
    def __init__(self):
        self.failures = 0

    def report(self, status: str, label: str, msg: str):
        if status == FAIL:
            self.failures += 1
        print(f"[{status}] {label}: {msg}")


def run_preflight(cfg_arg: str, select=None) -> int:
    rep = Reporter()

    # (a) cfg loads and paths resolve
    try:
        cfg, cfg_path = load_cfg(cfg_arg)
    except Exception as e:
        rep.report(FAIL, "cfg", f"could not load config '{cfg_arg}': {e}")
        print("PREFLIGHT FAILED: 1 check failed (cannot continue without cfg)")
        return 1
    data = cfg.get("data") or {}
    resolved = {}
    for key in ("question_data_path", "init_pose_data_path",
                "scene_data_path", "semantic_annot_data_path"):
        raw = data.get(key)
        if raw is None:
            rep.report(FAIL, "cfg", f"data.{key} missing from {cfg_path.name}")
            continue
        resolved[key] = resolve_data_path(raw)
    if len(resolved) == 4:
        rep.report(PASS, "cfg", f"loaded {cfg_path} and resolved 4 data paths")

    # (b) split exists and SHA-verifies
    split_rows = None
    split_name = cfg.get("split")
    if not split_name:
        rep.report(FAIL, "split",
                   f"{cfg_path.name} has no 'split' field; add e.g. "
                   "split: bench_v1_98")
    else:
        try:
            from src.splits import load_split, split_sha
            split_rows = load_split(split_name)
            rep.report(PASS, "split",
                       f"'{split_name}' verified, {len(split_rows)} rows, "
                       f"sha256 {split_sha(split_name)[:12]}...")
        except Exception as e:
            rep.report(FAIL, "split", f"'{split_name}' failed to load: {e}")

    # Also confirm the cfg question path IS the frozen split file.
    qpath = resolved.get("question_data_path")
    if qpath is not None:
        if Path(qpath).is_file():
            rep.report(PASS, "questions", f"question_data_path resolves to {qpath}")
        else:
            rep.report(FAIL, "questions",
                       f"question_data_path resolves to missing file {qpath}")

    # (c) scenes exist and have semantics
    scenes_dir = resolved.get("scene_data_path")
    if split_rows is None:
        rep.report(SKIP, "scenes", "split unavailable, scene checks skipped")
    elif scenes_dir is None or not Path(scenes_dir).is_dir():
        rep.report(SKIP, "scenes",
                   f"HM3D dataset dir absent at '{scenes_dir}'. Scene and "
                   "semantics coverage NOT verified. Do not start a run "
                   "from this machine.")
    else:
        scenes = sorted({r["scene"] for r in split_rows})
        missing_dirs = [s for s in scenes
                        if not (Path(scenes_dir) / s).is_dir()]
        if missing_dirs:
            rep.report(FAIL, "scenes",
                       f"{len(missing_dirs)} of {len(scenes)} split scenes "
                       f"missing under {scenes_dir}: {', '.join(missing_dirs)}")
        else:
            rep.report(PASS, "scenes",
                       f"all {len(scenes)} split scenes present under {scenes_dir}")

        # Semantic annotations list. Prefer the json list the runners use;
        # fall back to scanning scene dirs for *.semantic.glb files.
        sem_dir = resolved.get("semantic_annot_data_path", scenes_dir)
        sem_json = Path(sem_dir) / SEM_LIST_BASENAME
        source = None
        semantic_ok = set()
        if sem_json.is_file():
            import json as _json
            try:
                with open(sem_json) as f:
                    for p in _json.load(f):
                        semantic_ok.add(os.path.basename(str(p)).split(".")[0])
                source = f"list {sem_json}"
            except Exception as e:
                rep.report(FAIL, "semantics", f"could not parse {sem_json}: {e}")
        if source is None:
            for s in scenes:
                sdir = Path(scenes_dir) / s
                if sdir.is_dir() and list(sdir.glob("*.semantic.glb")):
                    semantic_ok.add(s.split("-", 1)[-1])
            source = (f"scan of scene dirs ({SEM_LIST_BASENAME} not found "
                      f"at {sem_json})")
        # Runners strip the '00410-' index prefix: scene_id = scene[6:].
        no_sem = [s for s in scenes if s.split("-", 1)[-1] not in semantic_ok]
        if no_sem:
            rep.report(FAIL, "semantics",
                       f"{len(no_sem)} of {len(scenes)} split scenes lack "
                       f"semantic annotations (source: {source}): "
                       f"{', '.join(no_sem)}. The runners used to skip "
                       "these silently; fix the dataset or the split.")
        else:
            rep.report(PASS, "semantics",
                       f"all {len(scenes)} split scenes have semantic "
                       f"annotations (source: {source})")

    # (d) init poses exist and cover the split
    pose_path = resolved.get("init_pose_data_path")
    if pose_path is None or not Path(pose_path).is_file():
        rep.report(FAIL, "init_poses", f"init pose file missing: {pose_path}")
    elif split_rows is None:
        rep.report(SKIP, "init_poses", "split unavailable, coverage not checked")
    else:
        import csv as _csv
        with open(pose_path, newline="", encoding="utf-8") as f:
            have = {r["scene_floor"] for r in _csv.DictReader(f)}
        need = sorted({f"{r['scene']}_{r['floor']}" for r in split_rows})
        missing = [n for n in need if n not in have]
        if missing:
            rep.report(FAIL, "init_poses",
                       f"{pose_path} covers {len(need) - len(missing)} of "
                       f"{len(need)} split scene_floor pairs; missing: "
                       f"{', '.join(missing)}")
        else:
            rep.report(PASS, "init_poses",
                       f"{pose_path} covers all {len(need)} scene_floor pairs")

    # (e) env keys for the selected backends
    #
    # Gate on what this run actually selects (vlm.name plus non-null
    # model_tiers), not on every model-looking string in the cfg. The
    # cfg documents the whole factorial permanently, so the wider walk
    # would demand OpenAI, Google and DashScope keys for a claude-only
    # run. If the selection cannot be read, fall back to the wide walk
    # so an unfamiliar cfg over-reports instead of under-reporting.
    all_aliases = collect_aliases(cfg)
    cli_select = {str(s).strip() for s in (select or []) if str(s).strip()}
    if cli_select:
        aliases = cli_select
        how = "--select"
    else:
        from_cfg = selected_aliases(cfg)
        aliases = from_cfg or all_aliases
        how = "cfg vlm.name" if from_cfg else "whole-cfg walk (fallback)"
    unselected = sorted(all_aliases - aliases)
    rep.report(INFO, "selection",
               f"gating via {how} on backend(s) "
               f"{', '.join(sorted(backends_for(aliases))) or 'none'}"
               + (f"; documented but not selected this run: "
                  f"{', '.join(unselected)}" if unselected else ""))
    env = load_env()
    missing_env = missing_env_backends(aliases, env)
    if not aliases:
        rep.report(INFO, "env_keys", "cfg selects no model backends")
    elif missing_env:
        parts = [f"{b} (need one of: {', '.join(keys)})"
                 for b, keys in missing_env]
        rep.report(FAIL, "env_keys",
                   "missing API keys for backend(s): " + "; ".join(parts))
    else:
        note = ""
        if "claude" in backends_for(aliases) and \
                not str(env.get("CLAUDE_API_KEY", "")).strip():
            note = (" (warning: agents read CLAUDE_API_KEY specifically; "
                    "only ANTHROPIC_API_KEY is set, copy it over)")
        rep.report(PASS, "env_keys",
                   f"API keys present for backends: "
                   f"{', '.join(sorted(backends_for(aliases)))}{note}")

    # (f) every used alias has a non-empty pin
    pins = cfg.get("model_pins") or {}
    need_pin = pinned_aliases_needed(aliases)
    bad = unpinned_aliases(aliases, pins)
    if not need_pin:
        rep.report(INFO, "model_pins", "no concrete model aliases in cfg")
    elif bad:
        rep.report(FAIL, "model_pins",
                   f"{len(bad)} alias(es) without a pinned snapshot id in "
                   f"model_pins: {', '.join(bad)}. Fill model_pins before "
                   "running; results from floating aliases are not "
                   "reproducible.")
    else:
        rep.report(PASS, "model_pins",
                   f"all {len(need_pin)} model aliases pinned")

    # (g) heavy deps importable (required in the container only)
    inside = in_container()
    for mod in ("numpy", "habitat_sim"):
        try:
            __import__(mod)
            rep.report(PASS if inside else INFO, "imports", f"{mod} importable")
        except Exception as e:
            if inside:
                rep.report(FAIL, "imports",
                           f"{mod} not importable inside the container: {e}")
            else:
                rep.report(INFO, "imports",
                           f"{mod} not importable outside the container "
                           "(expected on the host)")

    # (h) cost governor: caps + prices complete for enabled backends
    # (MAPG-11). No LLM run without the governor.
    if not backends_for(aliases):
        rep.report(INFO, "cost_governor",
                   "no model backends enabled; cost governor not required")
    else:
        gov_problems = cost_governor_problems(cfg, aliases)
        capped = sorted(
            {_BACKEND_TO_PROVIDER.get(b, b) for b in backends_for(aliases)}
        )
        if gov_problems:
            rep.report(FAIL, "cost_governor", "; ".join(gov_problems))
        elif not pinned_aliases_needed(aliases):
            # Every selected alias is a bare backend name, so there is
            # no concrete model to look up a price for and nothing was
            # actually verified. Saying PASS here would be a false
            # green: the run would reach the governor and only then
            # discover it cannot price what it is spending. Name a
            # concrete model (vlm.name or --select) to get a real check.
            rep.report(FAIL, "cost_governor",
                       "selected backend(s) "
                       + ", ".join(capped)
                       + " name no concrete model, so no price could be "
                         "verified; set vlm.name (or --select) to a pinned "
                         "model id rather than a bare backend name")
        else:
            rep.report(PASS, "cost_governor",
                       "caps and pinned prices cover enabled provider(s): "
                       + ", ".join(capped))

    if rep.failures:
        print(f"PREFLIGHT FAILED: {rep.failures} check(s) failed")
        return 1
    print("PREFLIGHT OK: all checks passed")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fail-loud preflight for MAPG benchmark runs")
    parser.add_argument("--cfg", default="mapg_benchmark",
                        help="cfg name under src/cfg (or a yaml path)")
    parser.add_argument("--select", action="append", metavar="ALIAS",
                        help="model alias this run will use, repeatable. "
                             "Overrides the cfg's vlm.name for gating, so "
                             "you can preflight the arm you are about to "
                             "launch: a runner invoked with vlm.name=X must "
                             "be preflighted with --select X, otherwise you "
                             "are checking a different arm than you run.")
    args = parser.parse_args(argv)
    return run_preflight(args.cfg, select=args.select)


if __name__ == "__main__":
    sys.exit(main())

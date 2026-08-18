"""Golden prompt proof (MAPG-09).

Two guarantees, both executable:

1. Version pinning: every role prompt rendered under the frozen
   context in tests/golden/context.py matches its snapshot byte for
   byte. Any wording change is an explicit snapshot diff
   (regenerate with ``python3 tests/golden/regen.py`` and review the
   diff in the commit).
2. Backend identity: the same rendered text packaged through all four
   backend configs (claude, openai, alibaba, gemini) is extracted
   back out of each provider request payload byte-identical, with the
   image riding as a separate non-text part. This is the precondition
   for the pipeline x backend factorial (EXPERIMENT_PLAN I1 / P1);
   the legacy families failed it (claude/gemini carried extra schema
   blocks the openai/alibaba copies had lost).

``build_request`` is pure request shaping: no SDK, no network, no API
key, so this runs on the host.
"""

import os

import pytest

from src.agents.backends.claude import ClaudeBackend
from src.agents.backends.gemini import GeminiBackend
from src.agents.backends.openai_compat import OpenAICompatBackend
from src.agents.base import image_part, text_part
from tests.golden.context import render_all

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")

RENDERED = render_all()
ROLES = [row[0] for row in RENDERED]


def _snapshot(role, kind):
    path = os.path.join(GOLDEN_DIR, f"{role}_{kind}.txt")
    assert os.path.exists(path), (
        f"missing snapshot {path}; run: python3 tests/golden/regen.py"
    )
    with open(path, "rb") as f:
        return f.read().decode("utf-8")


def _backends():
    return [
        ("claude", ClaudeBackend()),
        ("openai", OpenAICompatBackend(provider="openai")),
        ("alibaba", OpenAICompatBackend(provider="alibaba")),
        ("gemini", GeminiBackend()),
    ]


def _extract_texts(provider, request):
    """(system_text, [user_text_parts], n_image_parts) from a payload.

    MAPG-10: caching annotations are request structure, not text. The
    claude system may be a block list (each block carrying
    cache_control) and user text may arrive as several blocks; the
    extraction joins the text and ignores cache_control, so the
    equality assertions below still compare prompt BYTES.
    """
    if provider == "claude":
        content = request["messages"][0]["content"]
        texts = [c["text"] for c in content if c.get("type") == "text"]
        images = [c for c in content if c.get("type") == "image"]
        system = request["system"]
        if isinstance(system, list):
            system = "".join(b["text"] for b in system)
        return system, texts, len(images)
    if provider in ("openai", "alibaba"):
        system = request["messages"][0]["content"]
        content = request["messages"][1]["content"]
        texts = [c["text"] for c in content if c.get("type") == "text"]
        images = [c for c in content if c.get("type") == "image_url"]
        return system, texts, len(images)
    if provider == "gemini":
        parts = request["contents"][0]["parts"]
        texts = [p["text"] for p in parts if "text" in p]
        images = [p for p in parts if "inline_data" in p]
        return request["system_instruction"], texts, len(images)
    raise AssertionError(provider)


# ----------------------------------------------------------------------
# 1. Snapshot pinning
# ----------------------------------------------------------------------
@pytest.mark.parametrize("role,system,user", RENDERED, ids=ROLES)
def test_rendered_prompt_matches_snapshot(role, system, user):
    assert system == _snapshot(role, "system")
    assert user == _snapshot(role, "user")


# ----------------------------------------------------------------------
# 2. Byte-identical text through every backend adapter
# ----------------------------------------------------------------------
@pytest.mark.parametrize("role,system,user", RENDERED, ids=ROLES)
def test_prompt_text_identical_across_all_backends(role, system, user):
    golden_system = _snapshot(role, "system")
    golden_user = _snapshot(role, "user")
    for provider, backend in _backends():
        request = backend.build_request(system, [text_part(user)])
        got_system, got_texts, n_images = _extract_texts(provider, request)
        assert got_system == golden_system, f"{provider}: system text diverged"
        assert "".join(got_texts) == golden_user, f"{provider}: user text diverged"
        assert n_images == 0


@pytest.mark.parametrize("role,system,user", RENDERED, ids=ROLES)
def test_image_attachment_stays_outside_the_text(role, system, user, tmp_path):
    img = tmp_path / "current_img_0.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\nnot-a-real-png")
    golden_user = _snapshot(role, "user")
    for provider, backend in _backends():
        request = backend.build_request(
            system, [text_part(user), image_part(str(img))]
        )
        got_system, got_texts, n_images = _extract_texts(provider, request)
        # Attaching the image changes nothing about the text parts.
        assert got_system == _snapshot(role, "system")
        assert "".join(got_texts) == golden_user, f"{provider}: image leaked into text"
        assert n_images == 1, f"{provider}: image part missing"

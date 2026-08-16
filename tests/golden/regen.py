"""Regenerate the golden prompt snapshots (MAPG-09).

Usage, from the repo root:

    python3 tests/golden/regen.py

Writes tests/golden/<role>_system.txt and <role>_user.txt for every
role from the frozen context in tests/golden/context.py. Run this
after any DELIBERATE prompt or schema wording change, then review the
snapshot diff in the commit: the diff IS the prompt version history
(EXPERIMENT_PLAN P1, "prompts versioned").
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _REPO)

from tests.golden.context import render_all  # noqa: E402


def main() -> None:
    for role, system, user in render_all():
        for kind, text in (("system", system), ("user", user)):
            path = os.path.join(_HERE, f"{role}_{kind}.txt")
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            print(f"wrote {os.path.relpath(path, _REPO)} ({len(text)} chars)")


if __name__ == "__main__":
    main()

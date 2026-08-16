"""Unified agent stack (MAPG-09 / EXPERIMENT_PLAN P1).

One prompt per role, shared across backends; typed outputs; thin
backend adapters that return provider usage. Replaces the 24
per-backend agent files under src/multi_agent/agents/.

Layout (EXPERIMENT_PLAN section 1, "Target architecture"):

    src/agents/
      base.py       # backend protocol, prompt containers, JSON helpers
      schemas.py    # one typed output schema per role + validator
      prompts/      # ONE prompt template per role, backend-independent
      backends/     # thin adapters: claude, openai_compat, gemini
      roles/        # role classes taking a backend instance
      factory.py    # create_backend / create_role

Import surface: use ``from src.agents.factory import create_role``.
This package imports no provider SDK at module scope; SDKs load
lazily at transport time so the host test suite runs without them.
"""

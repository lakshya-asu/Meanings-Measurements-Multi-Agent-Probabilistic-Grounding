"""Legacy factory shim (MAPG-09).

The 24 per-backend agent files under src/multi_agent/agents/ were
deleted after the unified stack in src/agents reached prompt and
behavior parity (golden proof: tests/test_golden_prompts.py).
AgentFactory keeps its historical call shape and now delegates to the
unified factory so existing planner call sites keep working; new code
should import src.agents.factory.create_role directly.

The 'logical' role no longer exists: it was constructed and never
called by any planner (orphan; EXPERIMENT_PLAN section 1).
"""

from src.agents.factory import create_role


class AgentFactory:
    @staticmethod
    def create_agent(role: str, provider: str = "gemini", **kwargs):
        return create_role(role, provider=provider, **kwargs)

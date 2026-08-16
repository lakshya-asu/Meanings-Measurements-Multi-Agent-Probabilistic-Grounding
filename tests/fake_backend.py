"""Fake backend for unified-agent tests (MAPG-09).

Implements the Backend.send contract with canned payloads keyed by
schema title, fixed usage numbers, and full capture of everything
sent, so tests can assert on prompt routing and usage plumbing with
no network and no SDK.
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base import Backend


class FakeBackend(Backend):
    provider = "fake"

    def __init__(
        self,
        responses: Dict[str, Any],
        usage: Optional[Dict[str, Optional[int]]] = None,
        model_name: str = "fake-model-1",
    ):
        super().__init__(model_name)
        # schema title -> payload, list of payloads (popped per call),
        # or an Exception instance to raise.
        self.responses = dict(responses)
        self.usage = dict(usage or {"prompt_tokens": 111, "completion_tokens": 22})
        self.sent: List[Tuple[str, List[Dict[str, Any]], Optional[Dict[str, Any]]]] = []

    def send(
        self,
        system: str,
        user_parts: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Optional[int]], float]:
        self.sent.append((system, list(user_parts), schema))
        key = (schema or {}).get("title")
        if key not in self.responses:
            raise AssertionError(f"FakeBackend has no canned response for {key!r}")
        payload = self.responses[key]
        if isinstance(payload, list):
            payload = payload.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return copy.deepcopy(payload), dict(self.usage), 1.0

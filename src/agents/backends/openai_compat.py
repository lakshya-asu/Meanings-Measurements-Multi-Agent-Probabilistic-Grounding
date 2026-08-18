"""OpenAI-compatible adapter: OpenAI proper and Alibaba/DashScope.

One adapter, two configs, exactly as the legacy openai_* and
alibaba_* families were one client with a base_url override.

Schema enforcement: the schema text sits in the rendered prompt
(identical for all backends); the request additionally asks for JSON
mode (``response_format={"type": "json_object"}``) when the endpoint
accepts it. Some DashScope models reject response_format; the adapter
then drops the parameter for the retried request and remembers the
capability, so the degradation is one extra transport attempt once per
process, not per call. Replies are json-parsed here and validated in
the role. The legacy ``beta.chat.completions.parse`` pydantic path was
dropped deliberately: enforcement machinery is now uniform across
backends (prompted schema + JSON mode + shared validator), so the
factorial does not compare different enforcement stacks.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    Backend,
    BackendError,
    encode_image_b64,
    guess_mime,
    usage_dict,
)

DASHSCOPE_BASE_URL = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"

PROVIDER_DEFAULTS = {
    "openai": {
        "model": "gpt-5.2-chat-latest",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
    },
    "alibaba": {
        "model": "qwen3-vl-plus",
        "api_key_env": "ALIBABA_API_KEY",
        "base_url": DASHSCOPE_BASE_URL,
    },
}


class OpenAICompatBackend(Backend):
    def __init__(
        self,
        provider: str = "openai",
        model_name: Optional[str] = None,
        api_key_env: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        provider = str(provider).lower()
        if provider not in PROVIDER_DEFAULTS:
            raise BackendError(f"unknown openai-compatible provider: {provider!r}")
        defaults = PROVIDER_DEFAULTS[provider]
        super().__init__(model_name or defaults["model"])
        self.provider = provider
        self.api_key_env = api_key_env or defaults["api_key_env"]
        self.base_url = base_url if base_url is not None else defaults["base_url"]
        self._client = None
        self._json_mode_supported = True

    # ------------------------------------------------------------------
    def build_request(
        self,
        system: str,
        user_parts: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = []
        for part in user_parts:
            if part.get("type") == "text":
                content.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_path":
                path = part["path"]
                mime = guess_mime(path)
                b64_img = encode_image_b64(path)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64_img}"},
                    }
                )
            else:
                raise BackendError(f"unknown user part type: {part.get('type')!r}")
        request: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": DEFAULT_TEMPERATURE,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        }
        if self._json_mode_supported:
            request["response_format"] = {"type": "json_object"}
        return request

    # ------------------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            if self.api_key_env not in os.environ:
                raise BackendError(
                    f"{self.api_key_env} must be set in the environment."
                )
            kwargs: Dict[str, Any] = {"api_key": os.environ[self.api_key_env]}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _transport(self, request: Dict[str, Any]) -> Tuple[str, Dict[str, Optional[int]]]:
        client = self._get_client()
        try:
            completion = client.chat.completions.create(**request)
        except Exception as e:
            # Capability fallback, not a semantic retry: some DashScope
            # models reject response_format. Drop it once, remember.
            if "response_format" in request and "response_format" in str(e):
                self._json_mode_supported = False
                retried = {k: v for k, v in request.items() if k != "response_format"}
                completion = client.chat.completions.create(**retried)
            else:
                raise
        return completion.choices[0].message.content, usage_dict(completion)

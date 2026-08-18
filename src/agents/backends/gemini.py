"""Google Gemini adapter (google.generativeai).

The system prompt travels as ``system_instruction`` so the user text
stays byte-identical to what every other backend receives. (The legacy
gemini family concatenated the system text into the user prompt, which
was one of the divergences MAPG-09 removes.) If the installed SDK
predates ``system_instruction``, the adapter falls back to the legacy
concatenation and records ``system_via='concat'`` in the request dict
so the degradation is visible in traces.

Schema enforcement: prompted schema + JSON mime type + the shared
role validator, uniform with the other backends. The legacy
hand-built ``genai.protos.Schema`` objects are gone with the legacy
files; parity is preserved by the validator being at least as strict.
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

DEFAULT_MODEL = "models/gemini-3-pro-preview"
API_KEY_ENV = "GOOGLE_API_KEY"


class GeminiBackend(Backend):
    provider = "gemini"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        super().__init__(model_name)
        self._configured = False

    # ------------------------------------------------------------------
    def build_request(
        self,
        system: str,
        user_parts: List[Dict[str, Any]],
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        parts: List[Dict[str, Any]] = []
        for part in user_parts:
            if part.get("type") == "text":
                parts.append({"text": part["text"]})
            elif part.get("type") == "image_path":
                path = part["path"]
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": guess_mime(path),
                            "data": encode_image_b64(path),
                        }
                    }
                )
            else:
                raise BackendError(f"unknown user part type: {part.get('type')!r}")
        return {
            "model": self.model_name,
            "system_instruction": system,
            "system_via": "system_instruction",
            "contents": [{"role": "user", "parts": parts}],
            "generation_config": {
                "response_mime_type": "application/json",
                "temperature": DEFAULT_TEMPERATURE,
                "max_output_tokens": DEFAULT_MAX_TOKENS,
            },
        }

    # ------------------------------------------------------------------
    def _configure(self):
        if not self._configured:
            import google.generativeai as genai

            if API_KEY_ENV not in os.environ:
                raise BackendError(f"{API_KEY_ENV} must be set in the environment.")
            genai.configure(api_key=os.environ[API_KEY_ENV])
            self._configured = True

    def _transport(self, request: Dict[str, Any]) -> Tuple[str, Dict[str, Optional[int]]]:
        import google.generativeai as genai

        self._configure()
        contents = request["contents"]
        try:
            model = genai.GenerativeModel(
                model_name=request["model"],
                system_instruction=request["system_instruction"],
            )
        except TypeError:
            # SDK predates system_instruction: legacy concatenation,
            # recorded so traces show the degraded shape.
            request["system_via"] = "concat"
            model = genai.GenerativeModel(model_name=request["model"])
            first_text = {
                "text": request["system_instruction"] + "\n" + contents[0]["parts"][0]["text"]
            }
            contents = [
                {"role": "user", "parts": [first_text] + list(contents[0]["parts"][1:])}
            ]
        response = model.generate_content(
            contents,
            generation_config=genai.GenerationConfig(**request["generation_config"]),
        )
        return response.text, usage_dict(response)

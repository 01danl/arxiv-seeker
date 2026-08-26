"""Shared LLM client used by RagChat and the search agents."""
from __future__ import annotations

import logging
import time

from arxiv_seeker.config import get_settings

logger = logging.getLogger(__name__)

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}


class LLMClient:
    def __init__(self):
        self.settings = get_settings()

    def chat(self, system: str, user: str, json_mode: bool = False) -> str:
        provider = self.settings.llm_provider
        if provider == "ollama":
            return self._ollama(system, user, json_mode)
        if provider == "openai":
            return self._openai(system, user, json_mode)
        if provider == "anthropic":
            return self._anthropic(system, user)
        raise ValueError(f"Unknown provider: {provider}")

    def _openai(self, system, user, json_mode, max_retries: int = 3):
        from openai import OpenAI, APIStatusError

        client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            timeout=60.0,
        )
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}

        last_error = None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    **kwargs,
                )
                return resp.choices[0].message.content
            except APIStatusError as exc:
                last_error = exc
                if exc.status_code in _RETRYABLE_STATUSES:
                    wait = 2 ** attempt
                    logger.warning(
                        "OpenAI API error %d (attempt %d/%d), retrying in %ds",
                        exc.status_code, attempt + 1, max_retries, wait,
                    )
                    time.sleep(wait)
                else:
                    raise  # non-retryable (401, 403, etc.)
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "OpenAI request failed (attempt %d/%d): %s, retrying in %ds",
                        attempt + 1, max_retries, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(
            f"OpenAI call failed after {max_retries} retries: {last_error}"
        ) from last_error

    def _ollama(self, system, user, json_mode):
        import requests
        payload = {
            "model": self.settings.ollama_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"
        resp = requests.post(f"{self.settings.ollama_base_url}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def _anthropic(self, system, user):
        import anthropic
        client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        resp = client.messages.create(
            model=self.settings.anthropic_model, max_tokens=1024,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text"))
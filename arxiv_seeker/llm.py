"""Shared LLM client used by RagChat and the search agents."""
from __future__ import annotations
from arxiv_seeker.config import get_settings

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

    def _openai(self, system, user, json_mode):
        from openai import OpenAI
        client = OpenAI(api_key=self.settings.openai_api_key, base_url=self.settings.openai_base_url)
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )
        return resp.choices[0].message.content

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
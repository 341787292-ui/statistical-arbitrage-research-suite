from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LLMResult:
    text: str
    model: str


class OpenAITextClient:
    """Small wrapper so the rest of the app can run without an API key."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str) -> LLMResult:
        if not self.available:
            raise RuntimeError("OPENAI_API_KEY is not set.")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package to use LLM extraction.") from exc

        client = OpenAI(api_key=self.api_key)

        if hasattr(client, "responses"):
            response = client.responses.create(model=self.model, input=prompt)
            return LLMResult(text=response.output_text, model=self.model)

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResult(text=response.choices[0].message.content or "", model=self.model)

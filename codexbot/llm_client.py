from __future__ import annotations

import os
from typing import Any, Dict, Optional

from openai import OpenAI

# Default system prompt (can be overridden)
SYSTEM_PROMPT = "You are a helpful assistant with access to a vector database of source code. Use the provided context to answer the user's question."


class LLMClient:
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None):
        self.base_url = api_url or os.getenv("LLM_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "qwen-flash")

        if not self.api_key:
            raise RuntimeError("Set DASHSCOPE_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def complete(self, prompt: str, max_tokens: int = 2000, temperature: float = 0.1) -> str:
        # The new implementation expects a system prompt and a user prompt.
        # We'll split the provided prompt into system and user parts.
        # For simplicity, we assume the user_prompt is the whole prompt here.
        user_prompt = prompt

        completion = self.client.chat.completions.create(
            model=self.model,
            stream=False,
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            timeout=120,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content or ""

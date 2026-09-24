"""
Shared LLM client using OpenAI-compatible API.
Supports: vLLM, OpenAI, Azure, and other compatible endpoints.

Usage:
    from llm_client import LLMClient
    client = LLMClient(base_url="http://localhost:8000/v1", model="qwen3-8b")
    result = client.chat(messages=[{"role": "user", "content": "Hello"}])
"""

import os
import json
import time
from openai import OpenAI


class LLMClient:
    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        api_key: str = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        enable_thinking: bool = False,
        top_p: float = None,
        top_k: int = None,
    ):
        self.base_url = base_url or os.environ.get(
            "VLLM_BASE_URL", "http://localhost:8000/v1"
        )
        self.model = model or os.environ.get("VLLM_MODEL", "qwen3-8b")
        self.api_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_thinking = enable_thinking
        self.top_p = top_p
        self.top_k = top_k

        # OpenAI client requires base_url to include /v1; auto-append if missing
        base_url_fixed = self.base_url.rstrip("/")
        if not base_url_fixed.endswith("/v1"):
            base_url_fixed += "/v1"

        self.client = OpenAI(base_url=base_url_fixed, api_key=self.api_key)
        print(f"[LLMClient] endpoint={base_url_fixed}, model={self.model}")

    def chat(self, messages: list, **kwargs):
        """Send a chat request and return text (or text plus finish metadata)."""
        return_metadata = bool(kwargs.get("return_metadata", False))
        last_error = None
        for attempt in range(self.max_retries):
            try:
                extra_body = {
                    "enable_thinking": self.enable_thinking,
                    "chat_template_kwargs": {
                        "enable_thinking": self.enable_thinking
                    },
                }
                if self.top_k is not None:
                    extra_body["top_k"] = self.top_k

                request_kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "extra_body": extra_body,
                }
                top_p = kwargs.get("top_p", self.top_p)
                if top_p is not None:
                    request_kwargs["top_p"] = top_p

                response = self.client.chat.completions.create(
                    **request_kwargs
                )
                content = response.choices[0].message.content
                if content is None or not content.strip():
                    reasoning = getattr(
                        response.choices[0].message, "reasoning_content", None
                    )
                    if reasoning is None:
                        reasoning = getattr(
                            response.choices[0].message, "reasoning", None
                        )
                    if reasoning is None:
                        message_extra = getattr(
                            response.choices[0].message, "model_extra", None
                        ) or {}
                        reasoning = message_extra.get(
                            "reasoning_content", message_extra.get("reasoning")
                        )
                    if reasoning and not self.enable_thinking:
                        # Backward-compatible fallback for existing non-thinking
                        # jobs. Thinking jobs must never persist private reasoning
                        # as the requested final artifact.
                        if return_metadata:
                            return {
                                "content": reasoning,
                                "finish_reason": response.choices[0].finish_reason,
                            }
                        return reasoning
                    finish = response.choices[0].finish_reason
                    reasoning_chars = len(reasoning) if reasoning else 0
                    raise RuntimeError(
                        "Empty final response "
                        f"(finish_reason={finish}, reasoning_chars={reasoning_chars})"
                    )
                if return_metadata:
                    return {
                        "content": content,
                        "finish_reason": response.choices[0].finish_reason,
                    }
                return content
            except Exception as e:
                last_error = e
                print(
                    f"[LLMClient] Attempt {attempt + 1} failed "
                    f"(max_tokens={request_kwargs.get('max_tokens', self.max_tokens)}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
        raise RuntimeError(f"All {self.max_retries} retries failed: {last_error}")

    def chat_json(self, messages: list, **kwargs) -> dict:
        """Send a chat request expecting JSON output, parse and return it."""
        text = self.chat(messages, **kwargs)
        if not text:
            raise ValueError("Empty response from LLM")
        # Try to extract JSON from markdown code blocks first
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()
        return json.loads(text)

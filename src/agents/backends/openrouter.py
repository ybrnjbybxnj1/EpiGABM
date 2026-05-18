
from __future__ import annotations

import os
import time

from loguru import logger
from openai import (
    APIConnectionError, APITimeoutError, InternalServerError, OpenAI,
    RateLimitError,
)
from tenacity import (
    before_sleep_log, retry, retry_if_exception_type, stop_after_attempt,
    wait_exponential,
)

from src.agents.base import BaseLLMBackend

import logging as _stdlib_logging

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterBackend(BaseLLMBackend):
    def __init__(self, config: dict) -> None:
        self.model: str = config["model"]
        self.temperature: float = config.get("temperature", 0.3)
        self.max_tokens: int = config.get("max_tokens", 500)
        self.timeout_s: int = config.get("timeout_s", 60)
        self.cost_per_1m_input: float = config.get("cost_per_1m_input", 0.0)
        self.cost_per_1m_output: float = config.get("cost_per_1m_output", 0.0)
        api_key = config.get("api_key") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set; export it or put it in .env"
            )
        default_headers = {
            "HTTP-Referer": config.get("referer", "https://github.com/ybrnjbybxnj1/nir4"),
            "X-Title": config.get("title", "EpiGABM"),
        }
        self.client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers=default_headers,
            timeout=self.timeout_s,
            max_retries=5,
        )
        self.last_tokens_input: int = 0
        self.last_tokens_output: int = 0
        self.last_latency_ms: int = 0

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    @retry(
        stop=stop_after_attempt(12),
        wait=wait_exponential(multiplier=2, min=2, max=120),
        retry=retry_if_exception_type((
            RateLimitError, APIConnectionError, APITimeoutError, InternalServerError,
        )),
        before_sleep=before_sleep_log(_stdlib_logging.getLogger("openrouter.retry"),
                                      _stdlib_logging.WARNING),
        reraise=True,
    )
    def _call_api(self, prompt: str) -> str:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        self.last_tokens_input = (
            usage.prompt_tokens if usage and usage.prompt_tokens
            else self._estimate_tokens(prompt)
        )
        self.last_tokens_output = (
            usage.completion_tokens if usage and usage.completion_tokens
            else self._estimate_tokens(content)
        )
        self.last_latency_ms = elapsed_ms
        return content

    def query(self, prompt: str) -> str:
        logger.debug(
            "openrouter query, model={}, prompt length={}", self.model, len(prompt),
        )
        return self._call_api(prompt)

    def estimate_cost(self) -> float:
        input_cost = self.last_tokens_input * self.cost_per_1m_input / 1_000_000
        output_cost = self.last_tokens_output * self.cost_per_1m_output / 1_000_000
        return input_cost + output_cost
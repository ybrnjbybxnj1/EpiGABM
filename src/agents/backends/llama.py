import logging as _stdlib_logging
import time

from loguru import logger
from ollama import Client
from tenacity import (
    before_sleep_log, retry, stop_after_attempt, wait_exponential,
)

from src.agents.base import BaseLLMBackend


class LlamaBackend(BaseLLMBackend):
    def __init__(self, config: dict) -> None:
        self.model: str = config["model"]
        self.temperature: float = config["temperature"]
        self.timeout_s: int = config.get("timeout_s", 180)
        endpoint: str = config.get("endpoint", "http://localhost:11434")
        self.client = Client(host=endpoint, timeout=self.timeout_s)
        self.last_tokens_input: int = 0
        self.last_tokens_output: int = 0
        self.last_latency_ms: int = 0

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=2, min=2, max=120),
        before_sleep=before_sleep_log(_stdlib_logging.getLogger("llama.retry"),
                                      _stdlib_logging.WARNING),
        reraise=True,
    )
    def _call_api(self, prompt: str) -> str:
        start = time.perf_counter()
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        content = response.message.content or ""
        self.last_tokens_input = (
            response.prompt_eval_count
            if response.prompt_eval_count is not None
            else self._estimate_tokens(prompt)
        )
        self.last_tokens_output = (
            response.eval_count
            if response.eval_count is not None
            else self._estimate_tokens(content)
        )
        self.last_latency_ms = elapsed_ms
        return content

    def query(self, prompt: str) -> str:
        logger.debug(
            "llama query, model={}, prompt length={}", self.model, len(prompt)
        )
        return self._call_api(prompt)

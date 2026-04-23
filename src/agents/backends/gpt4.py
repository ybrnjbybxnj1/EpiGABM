import os
import time

import tiktoken
from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.agents.base import BaseLLMBackend

class GPT4Backend(BaseLLMBackend):
    def __init__(self, config: dict) -> None:
        self.model: str = config["model"]
        self.temperature: float = config["temperature"]
        self.max_tokens: int = config["max_tokens"]
        self.timeout_s: int = config.get("timeout_s", 30)
        self.cost_per_1m_input: float = config.get("cost_per_1m_input", 2.50)
        self.cost_per_1m_output: float = config.get("cost_per_1m_output", 10.0)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        try:
            self._encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        self.last_tokens_input: int = 0
        self.last_tokens_output: int = 0
        self.last_latency_ms: int = 0

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _call_api(self, prompt: str) -> str:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout_s,
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        content = response.choices[0].message.content or ""
        usage = response.usage
        self.last_tokens_input = (
            usage.prompt_tokens if usage else self.count_tokens(prompt)
        )
        self.last_tokens_output = (
            usage.completion_tokens if usage else self.count_tokens(content)
        )
        self.last_latency_ms = elapsed_ms
        return content

    def query(self, prompt: str) -> str:
        logger.debug("gpt4 query, prompt length={}", len(prompt))
        return self._call_api(prompt)

    def estimate_cost(self) -> float:
        input_cost = self.last_tokens_input * self.cost_per_1m_input / 1_000_000
        output_cost = self.last_tokens_output * self.cost_per_1m_output / 1_000_000
        return input_cost + output_cost

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from loguru import logger

from src.agents.base import BaseLLMBackend

def _cache_key(model_id: str, prompt: str) -> str:
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()

class CachedBackend(BaseLLMBackend):
    def __init__(
        self,
        inner: BaseLLMBackend,
        cache_dir: str | Path = "cache",
        read_only: bool = False,
    ) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.read_only = read_only
        self.model = getattr(inner, "model", type(inner).__name__)
        self._hits = 0
        self._misses = 0
        self._forwarded_attrs = {
            "last_tokens_input", "last_tokens_output",
            "last_latency_ms",
        }

    def __getattr__(self, name: str):
        if name in self.__dict__.get("_forwarded_attrs", set()):
            return getattr(self.inner, name, 0)
        raise AttributeError(name)

    def estimate_cost(self) -> float:
        return getattr(self.inner, "estimate_cost", lambda: 0.0)()

    def query(self, prompt: str) -> str:
        key = _cache_key(self.model, prompt)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    record = json.load(f)
                self._hits += 1
                return record["response"]
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("cache read failed for {}: {}", key[:8], e)
        if self.read_only:
            raise RuntimeError(
                f"cache miss for {key[:8]} and backend is read_only"
            )
        response = self.inner.query(prompt)
        self._misses += 1
        try:
            record = {
                "model": self.model,
                "prompt": prompt,
                "response": response,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
        except OSError as e:
            logger.warning("cache write failed for {}: {}", key[:8], e)
        return response

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": (self._hits / total) if total else 0.0,
        }

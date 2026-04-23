from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL,
)

class BehaviorDecision(BaseModel):
    isolate: bool
    isolate_confidence: float = Field(ge=0.0, le=1.0)
    mask: bool
    mask_confidence: float = Field(ge=0.0, le=1.0)
    reduce_contacts: bool
    reduce_contacts_confidence: float = Field(ge=0.0, le=1.0)
    see_doctor: bool
    see_doctor_confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str

FALLBACK_DECISION = BehaviorDecision(
    isolate=False,
    isolate_confidence=0.0,
    mask=False,
    mask_confidence=0.0,
    reduce_contacts=False,
    reduce_contacts_confidence=0.0,
    see_doctor=False,
    see_doctor_confidence=0.0,
    reasoning="fallback: LLM unavailable",
)

class BaseLLMBackend(ABC):
    @abstractmethod
    def query(self, prompt: str) -> str:
        ...

def _strip_code_fence(raw: str) -> str:
    m = _CODE_FENCE_RE.match(raw)
    if m:
        return m.group(1).strip()
    return raw

def parse_decision(raw: str) -> BehaviorDecision:
    cleaned = _strip_code_fence(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid json: {e}") from e
    try:
        return BehaviorDecision.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"schema validation failed: {e}") from e

def query_with_fallback(
    backend: BaseLLMBackend,
    prompt: str,
    retry_suffix: str = "\n\nОтветь ТОЛЬКО валидным JSON объектом.",
) -> tuple[BehaviorDecision, str, bool]:
    raw = backend.query(prompt)
    try:
        decision = parse_decision(raw)
        return decision, raw, True
    except ValueError:
        logger.warning("first parse failed, retrying with json reminder")
    raw_retry = backend.query(prompt + retry_suffix)
    try:
        decision = parse_decision(raw_retry)
        return decision, raw_retry, True
    except ValueError:
        logger.warning("retry also failed, using fallback decision")
    return FALLBACK_DECISION, raw, False

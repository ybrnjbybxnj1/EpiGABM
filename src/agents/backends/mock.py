from src.agents.base import BaseLLMBackend, BehaviorDecision

class MockBackend(BaseLLMBackend):
    def __init__(self, default_decision: BehaviorDecision | None = None) -> None:
        self.default_decision = default_decision or BehaviorDecision(
            isolate=False,
            isolate_confidence=0.5,
            mask=False,
            mask_confidence=0.5,
            reduce_contacts=False,
            reduce_contacts_confidence=0.5,
            see_doctor=False,
            see_doctor_confidence=0.5,
            reasoning="mock response",
        )
        self.call_count: int = 0
        self.call_log: list[str] = []

    def query(self, prompt: str) -> str:
        self.call_count += 1
        self.call_log.append(prompt)
        return self.default_decision.model_dump_json()

class AllIsolateMockBackend(MockBackend):
    def __init__(self) -> None:
        decision = BehaviorDecision(
            isolate=True,
            isolate_confidence=0.95,
            mask=True,
            mask_confidence=0.95,
            reduce_contacts=True,
            reduce_contacts_confidence=0.9,
            see_doctor=True,
            see_doctor_confidence=0.8,
            reasoning="mock: max caution",
        )
        super().__init__(default_decision=decision)

class InvalidJsonMockBackend(BaseLLMBackend):
    def __init__(self) -> None:
        self.call_count: int = 0
        self.call_log: list[str] = []
        self._fallback = BehaviorDecision(
            isolate=False,
            isolate_confidence=0.5,
            mask=False,
            mask_confidence=0.5,
            reduce_contacts=False,
            reduce_contacts_confidence=0.5,
            see_doctor=False,
            see_doctor_confidence=0.5,
            reasoning="recovered after retry",
        )

    def query(self, prompt: str) -> str:
        self.call_count += 1
        self.call_log.append(prompt)
        if self.call_count % 2 == 1:
            return "this is not json at all {broken"
        return self._fallback.model_dump_json()

class AlwaysInvalidMockBackend(BaseLLMBackend):
    def __init__(self) -> None:
        self.call_count: int = 0
        self.call_log: list[str] = []

    def query(self, prompt: str) -> str:
        self.call_count += 1
        self.call_log.append(prompt)
        return "not json {{"

import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class Intent(str, Enum):
    MEDICAL_INFORMATION = "medical_information"
    MEDICAL_ADVICE = "medical_advice"


class RiskLevel(str, Enum):
    NORMAL = "normal"
    HIGH = "high"


class Action(str, Enum):
    SEARCH_MEDICAL_KNOWLEDGE = "search_medical_knowledge"
    ESCALATE_HIGH_RISK_REQUEST = "escalate_high_risk_request"


@dataclass(frozen=True)
class RouteDecision:
    intent: Intent
    risk: RiskLevel
    action: Action
    reason: str
    allow_rag: bool

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["intent"] = self.intent.value
        result["risk"] = self.risk.value
        result["action"] = self.action.value
        return result


@dataclass(frozen=True)
class TraceEvent:
    step: str
    detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return asdict(self)


HIGH_RISK_RULES = (
    (
        "severe chest pain",
        re.compile(r"\b(severe|crushing|intense)\s+chest\s+pain\b", re.IGNORECASE),
    ),
    (
        "possible stroke symptoms",
        re.compile(
            r"\b(face droop(?:ing)?|one[- ]sided weakness|sudden slurred speech|"
            r"signs? of (?:a )?stroke|having (?:a )?stroke)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "severe breathing difficulty",
        re.compile(
            r"\b(severe (?:breathing difficulty|shortness of breath)|"
            r"cannot breathe|can(?:not|'t) breathe|gasping for (?:air|breath))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "possible anaphylaxis",
        re.compile(
            r"\b(anaphylaxis|throat (?:is )?(?:closing|swelling)|"
            r"severe allergic reaction)\b",
            re.IGNORECASE,
        ),
    ),
)


def new_trace() -> List[TraceEvent]:
    return [TraceEvent("Request received")]


def append_trace(
    trace: List[TraceEvent], step: str, detail: Optional[str] = None
) -> None:
    trace.append(TraceEvent(step, detail))


def route_request(
    redacted_question: str,
    classification: Optional[Dict[str, Any]] = None,
) -> RouteDecision:
    for reason, pattern in HIGH_RISK_RULES:
        if pattern.search(redacted_question):
            return RouteDecision(
                intent=Intent.MEDICAL_ADVICE,
                risk=RiskLevel.HIGH,
                action=Action.ESCALATE_HIGH_RISK_REQUEST,
                reason=f"Deterministic emergency rule matched: {reason}",
                allow_rag=False,
            )

    if str((classification or {}).get("risk_level", "")).strip().lower() == "high":
        return RouteDecision(
            intent=Intent.MEDICAL_ADVICE,
            risk=RiskLevel.HIGH,
            action=Action.ESCALATE_HIGH_RISK_REQUEST,
            reason="Classifier marked the request as high risk",
            allow_rag=False,
        )

    return RouteDecision(
        intent=Intent.MEDICAL_INFORMATION,
        risk=RiskLevel.NORMAL,
        action=Action.SEARCH_MEDICAL_KNOWLEDGE,
        reason="No deterministic emergency rule or high-risk classification matched",
        allow_rag=True,
    )


def record_decision(trace: List[TraceEvent], decision: RouteDecision) -> None:
    append_trace(trace, "Intent classified", decision.intent.value)
    append_trace(trace, "Risk classified", f"{decision.risk.value}: {decision.reason}")
    append_trace(trace, "Selected action", decision.action.value)


def escalate_high_risk_request(
    decision: RouteDecision,
    trace: List[TraceEvent],
) -> Dict[str, Any]:
    append_trace(trace, "Normal RAG generation skipped")
    append_trace(trace, "Autonomous treatment recommendation not produced")
    append_trace(trace, "Controlled escalation response returned")
    append_trace(trace, "Request completed")

    response = (
        "This may be a medical emergency. I cannot diagnose the cause or recommend "
        "medicine for it. Call emergency services now (111 in New Zealand), or ask "
        "someone nearby to call for you. Do not drive yourself. If you are outside "
        "New Zealand, call your local emergency number."
    )
    return {
        "answer": response,
        "sources": [],
        "classification": {},
        "decision": decision.to_dict(),
        "trace": [event.to_dict() for event in trace],
        "rewritten_query": None,
        "retrieval_count": 0,
    }

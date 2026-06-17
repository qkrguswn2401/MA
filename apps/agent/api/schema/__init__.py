"""Pydantic request/response models for the agent HTTP API (``apps.agent.api.server``).

Kept separate from the route handlers so the wire contract is in one place.
"""

from __future__ import annotations

from pydantic import BaseModel


# Request inputs for /ask and /ask/stream are declared as explicit ``Query(...)`` params on the
# route handlers (both endpoints are GET); only the response shape lives here as a model.


class TraceStep(BaseModel):
    step: int
    agent: str  # which pipeline agent ran: planner|router|retriever|verifier|synthesizer
    action: str
    arg: str
    thought: str


class AskResponse(BaseModel):
    question: str
    answer: str
    steps: int
    source: str = "wiki"  # which backend answered: wiki | dart
    dataset: str | None = None  # which wiki dataset answered (None for the dart backend)
    trace: list[TraceStep] | None = None


__all__ = ["AskResponse", "TraceStep"]

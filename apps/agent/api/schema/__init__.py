"""Pydantic request/response models for the agent HTTP API (``apps.agent.api.server``).

Kept separate from the route handlers so the wire contract is in one place.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., description="KO/EN question about the Centroid valuation.")
    max_steps: int = Field(3, ge=1, le=20, description="Per-branch read budget (initial read + retries).")
    include_trace: bool = Field(True, description="Return the routing trace.")


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
    trace: list[TraceStep] | None = None


__all__ = ["AskRequest", "AskResponse", "TraceStep"]

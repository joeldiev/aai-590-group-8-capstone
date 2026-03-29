"""Pydantic schemas for the severity evaluation component."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SeverityResponse(BaseModel):
    """Response from the severity evaluation service."""

    severity_tier: str = Field(
        ...,
        description="Severity tier: low, medium, high, or critical",
    )
    severity_score: int = Field(
        ...,
        description="Raw severity score (0-10)",
    )
    threat_type: str = Field(
        ...,
        description="Classified threat type: injection, jailbreak, exfiltration, or unknown_malicious",
    )
    threat_confidence: float = Field(
        ...,
        description="Confidence of threat type classification (0-1)",
    )
    threat_class_probabilities: dict[str, float] = Field(
        default_factory=dict,
        description="Probability distribution across threat types",
    )
    threat_intel: dict = Field(
        default_factory=dict,
        description="Threat intelligence from MITRE ATLAS and OWASP",
    )
    scoring_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Point-by-point breakdown of severity score",
    )
    latency_ms: float = Field(
        ...,
        description="Severity evaluation latency in milliseconds",
    )

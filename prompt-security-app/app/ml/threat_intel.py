"""
Threat intelligence lookup service.

Maps classified threat types to structured intelligence from:
  - MITRE ATLAS (Adversarial Threat Landscape for AI Systems)
  - OWASP Top 10 for LLM Applications (2025)

Supports online refresh with local cache fallback.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Default cache location (relative to project root)
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent.parent.parent / (
    "models/severity/threat_intel/threat_mapping.json"
)


class ThreatIntelService:
    """Threat intelligence lookup with online + cache fallback."""

    def __init__(
        self,
        cache_path: str | Path | None = None,
        online_timeout: float = 5.0,
    ):
        self.cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
        self.online_timeout = online_timeout
        self._mapping: dict = {}
        self._loaded = False

    def load(self) -> None:
        """Load threat mapping from local cache."""
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                self._mapping = json.load(f)
            logger.info(
                "Loaded threat intel cache from %s (%d entries)",
                self.cache_path, len(self._mapping),
            )
        else:
            logger.warning(
                "Threat intel cache not found at %s, using built-in defaults",
                self.cache_path,
            )
            self._mapping = self._builtin_mapping()

        self._loaded = True

    def lookup(self, threat_type: str) -> dict:
        """Look up threat intelligence for a given threat type.

        Tries online first (if configured), falls back to local cache.

        Args:
            threat_type: One of injection, jailbreak, exfiltration, unknown_malicious.

        Returns:
            Dict with mitre_atlas and owasp entries, plus general guidance.
        """
        if not self._loaded:
            self.load()

        # Try online refresh (non-blocking, with timeout)
        online_intel = self._try_online_fetch(threat_type)
        if online_intel:
            return online_intel

        # Fall back to cache
        return self._mapping.get(threat_type, self._mapping.get("unknown_malicious", {}))

    def _try_online_fetch(self, threat_type: str) -> dict | None:
        """Attempt to fetch latest threat intel from online sources.

        Returns None on failure (network error, timeout, etc.).
        Falls back gracefully to cached data.
        """
        # For now, online fetch is a placeholder.
        # In production, this would hit MITRE ATLAS STIX API
        # or OWASP data feeds. The cache is the primary source.
        #
        # TODO: Implement MITRE ATLAS STIX feed parser
        # URL: https://github.com/mitre-atlas/atlas-data
        return None

    def _builtin_mapping(self) -> dict:
        """Return hardcoded threat intel mapping as ultimate fallback."""
        return {
            "injection": {
                "mitre_atlas": {
                    "technique_id": "AML.T0051",
                    "name": "LLM Prompt Injection",
                    "url": "https://atlas.mitre.org/techniques/AML.T0051",
                    "tactics": ["ML Attack Staging"],
                    "description": (
                        "Adversary crafts input to override or manipulate "
                        "the intended behavior of an LLM by injecting "
                        "instructions that the model interprets as commands."
                    ),
                    "mitigations": [
                        "AML.M0016 — Adversarial Input Detection",
                        "AML.M0015 — Adversarial Input Detection, Filtering",
                    ],
                },
                "owasp": {
                    "id": "LLM01",
                    "name": "Prompt Injection",
                    "url": "https://genai.owasp.org/llmrisk/llm01-prompt-injection/",
                    "description": (
                        "Prompt injection occurs when user input alters the "
                        "LLM's behavior in unintended ways, potentially "
                        "causing it to ignore previous instructions."
                    ),
                    "prevention": [
                        "Constrain model behavior with strict system prompts",
                        "Implement input validation and sanitization",
                        "Use privilege control for LLM access to backend systems",
                        "Monitor and log LLM interactions for anomalous patterns",
                    ],
                },
                "recommended_actions": [
                    "Block prompt from reaching the LLM",
                    "Log full prompt for security audit",
                    "Alert security team if pattern is novel",
                ],
            },
            "jailbreak": {
                "mitre_atlas": {
                    "technique_id": "AML.T0054",
                    "name": "LLM Jailbreak",
                    "url": "https://atlas.mitre.org/techniques/AML.T0054",
                    "tactics": ["Defense Evasion"],
                    "description": (
                        "Adversary crafts prompts designed to bypass the "
                        "safety filters and content policies of an LLM, "
                        "causing it to generate restricted content."
                    ),
                    "mitigations": [
                        "AML.M0016 — Adversarial Input Detection",
                        "AML.M0019 — Use an Ensemble of Models",
                    ],
                },
                "owasp": {
                    "id": "LLM01",
                    "name": "Prompt Injection (Jailbreak variant)",
                    "url": "https://genai.owasp.org/llmrisk/llm01-prompt-injection/",
                    "description": (
                        "Jailbreaking is a subtype of prompt injection where "
                        "the adversary's goal is to bypass safety alignment "
                        "rather than redirect task behavior."
                    ),
                    "prevention": [
                        "Layer multiple safety classifiers",
                        "Implement output filtering in addition to input filtering",
                        "Use red-teaming to identify jailbreak vulnerabilities",
                        "Keep safety training data current with emerging techniques",
                    ],
                },
                "recommended_actions": [
                    "Block prompt from reaching the LLM",
                    "Classify jailbreak technique for threat tracking",
                    "Update blocklist patterns if novel technique detected",
                ],
            },
            "exfiltration": {
                "mitre_atlas": {
                    "technique_id": "AML.T0048",
                    "name": "LLM Data Leakage",
                    "url": "https://atlas.mitre.org/techniques/AML.T0048",
                    "tactics": ["Exfiltration", "Collection"],
                    "description": (
                        "Adversary crafts prompts to extract confidential "
                        "information from an LLM, including system prompts, "
                        "training data, or proprietary instructions."
                    ),
                    "mitigations": [
                        "AML.M0016 — Adversarial Input Detection",
                        "AML.M0004 — Restrict Number of ML Model Queries",
                    ],
                },
                "owasp": {
                    "id": "LLM06",
                    "name": "Sensitive Information Disclosure",
                    "url": "https://genai.owasp.org/llmrisk/llm06-sensitive-information-disclosure/",
                    "description": (
                        "LLM applications may reveal sensitive information, "
                        "proprietary algorithms, or confidential data through "
                        "their responses when prompted by an adversary."
                    ),
                    "prevention": [
                        "Apply strict output filtering for sensitive patterns",
                        "Implement data sanitization in training pipelines",
                        "Use access controls to limit what data LLM can reference",
                        "Audit system prompts for leakable confidential content",
                    ],
                },
                "recommended_actions": [
                    "Block prompt immediately — high-risk exfiltration attempt",
                    "Log and alert security team",
                    "Review what data the LLM has access to",
                    "Check if system prompt or training data was partially leaked",
                ],
            },
            "unknown_malicious": {
                "mitre_atlas": {
                    "technique_id": "AML.T0000",
                    "name": "Unknown/Novel Attack Pattern",
                    "url": "https://atlas.mitre.org/techniques/",
                    "tactics": ["Reconnaissance", "Initial Access"],
                    "description": (
                        "Prompt does not match known attack patterns but "
                        "exhibits anomalous characteristics suggesting "
                        "malicious intent. May represent a novel technique."
                    ),
                    "mitigations": [
                        "AML.M0016 — Adversarial Input Detection",
                        "AML.M0013 — Code Signing for ML Artifacts",
                    ],
                },
                "owasp": {
                    "id": "LLM09",
                    "name": "Overreliance",
                    "url": "https://genai.owasp.org/llmrisk/llm09-overreliance/",
                    "description": (
                        "Unknown attack patterns require human review. "
                        "Automated systems should flag but not solely "
                        "decide on novel threats."
                    ),
                    "prevention": [
                        "Implement human-in-the-loop review for low-confidence detections",
                        "Log novel patterns for future model retraining",
                        "Cross-reference with emerging threat intelligence feeds",
                    ],
                },
                "recommended_actions": [
                    "Flag for human review — potential novel attack",
                    "Quarantine prompt and log full context",
                    "Add to retraining dataset after manual labeling",
                ],
            },
        }

    def save_cache(self, mapping: dict | None = None) -> None:
        """Save threat mapping to local cache file.

        Args:
            mapping: Mapping dict to save. If None, saves current mapping.
        """
        to_save = mapping or self._mapping
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(to_save, f, indent=2)
        logger.info("Saved threat intel cache to %s", self.cache_path)

"""
Unify per-source labels → {0: Benign, 1: Injection, 2: Jailbreak, 3: Exfiltration}.

Decision rules:
  - If the prompt's GOAL is data extraction → Exfiltration
  - If the goal is bypassing safety filters → Jailbreak
  - If the goal is overriding system instructions → Injection
  - Otherwise → Benign
"""

import pandas as pd

from src.config import LABEL2ID

# ── Per-source mapping tables ──────────────────────────────────────────────

_DEEPSET_MAP = {
    "benign": "Benign",
    "injection": "Injection",
}

_JAILBREAKV28K_MAP = {
    "jailbreak": "Jailbreak",
}

_LAKERA_EXFIL_KEYWORDS = {"password", "extract", "leak", "exfiltrate", "reveal"}

_WILDGUARD_MAP = {
    "unharmful": "Benign",
    "harmful_adversarial": "Jailbreak",
    "harmful_nonadversarial": "Jailbreak",
}

_HACKAPROMPT_MAP = {
    "injection": "Injection",
}

_AWESOME_CHATGPT_MAP = {
    "benign": "Benign",
}

_VERAZUO_MAP = {
    "jailbreak": "Jailbreak",
}


# ── Mapper function ────────────────────────────────────────────────────────

def unify_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map original_label → unified label (str) and label (int).

    Args:
        df: Must have columns [text, source, original_label].

    Returns:
        df with added columns: unified_label (str), label (int).
        Rows that cannot be mapped are dropped with a warning.
    """
    df = df.copy()
    df["unified_label"] = df.apply(_map_row, axis=1)

    n_unmapped = df["unified_label"].isna().sum()
    if n_unmapped > 0:
        print(f"[label_mapping] Dropping {n_unmapped} unmappable rows")
        df = df.dropna(subset=["unified_label"])

    df["label"] = df["unified_label"].map(LABEL2ID)
    return df


def _map_row(row) -> str | None:
    """Route to per-source mapping logic."""
    source = row["source"]
    original = row["original_label"]

    if source == "deepset":
        return _DEEPSET_MAP.get(original)

    if source == "jailbreakv28k":
        return _JAILBREAKV28K_MAP.get(original)

    if source == "lakera_mosaic":
        return _map_lakera(row)

    if source == "hackaprompt":
        return _HACKAPROMPT_MAP.get(original)

    if source == "wildguardmix":
        return _WILDGUARD_MAP.get(original)

    if source == "awesome_chatgpt":
        return _AWESOME_CHATGPT_MAP.get(original)

    if source == "verazuo":
        return _VERAZUO_MAP.get(original)

    if source == "synthetic_exfiltration":
        return "Exfiltration"

    return None


def _map_lakera(row) -> str | None:
    """Lakera/mosaic: password/extraction categories → Exfiltration, rest → Injection."""
    text_lower = str(row.get("text", "")).lower()
    original = str(row.get("original_label", "")).lower()

    # Check if this is an exfiltration-type prompt
    if any(kw in original for kw in _LAKERA_EXFIL_KEYWORDS):
        return "Exfiltration"
    if any(kw in text_lower for kw in _LAKERA_EXFIL_KEYWORDS):
        return "Exfiltration"

    # Default: treat as injection
    return "Injection"

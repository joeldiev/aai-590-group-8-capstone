"""
Dataset loaders — each function returns a pd.DataFrame with columns:
    text (str), source (str), original_label (str)

Phase A (MVP): load_deepset(), load_jailbreakv28k()
Phase B: remaining loaders added incrementally.
"""

import pandas as pd
from datasets import load_dataset


# ── Phase A loaders ────────────────────────────────────────────────────────

def load_deepset() -> pd.DataFrame:
    """deepset/prompt-injections — 662 samples, binary (0=benign, 1=injection)."""
    ds = load_dataset("deepset/prompt-injections")
    frames = []
    for split in ds:
        df = ds[split].to_pandas()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"text": "text", "label": "original_label"})
    df["original_label"] = df["original_label"].map({0: "benign", 1: "injection"})
    df["source"] = "deepset"
    return df[["text", "source", "original_label"]]


def load_jailbreakv28k() -> pd.DataFrame:
    """JailBreakV-28K — jailbreak_query column, all are jailbreak prompts."""
    df = pd.read_csv(
        "hf://datasets/JailbreakV-28K/JailBreakV-28k/JailBreakV_28K/JailBreakV_28K.csv"
    )
    df = df.rename(columns={"jailbreak_query": "text"})
    df["source"] = "jailbreakv28k"
    df["original_label"] = "jailbreak"
    return df[["text", "source", "original_label"]]


# ── Phase B loaders ────────────────────────────────────────────────────────

def load_hackaprompt() -> pd.DataFrame:
    """hackaprompt — filter to correct=True (successful injections only).
    Uses 'user_input' as the prompt text."""
    ds = load_dataset("hackaprompt/hackaprompt-dataset")
    df = ds["train"].to_pandas()
    # Only keep successful injection attempts
    df = df[df["correct"] == True].copy()
    df = df.rename(columns={"user_input": "text"})
    df["source"] = "hackaprompt"
    df["original_label"] = "injection"
    # Drop rows with empty text
    df = df[df["text"].str.strip().astype(bool)]
    return df[["text", "source", "original_label"]]


def load_wildguardmix() -> pd.DataFrame:
    """allenai/wildguardmix — uses prompt_harm_label to split benign/harmful.
    adversarial+harmful prompts → inspect subcategory for label mapping."""
    train = pd.read_parquet(
        "hf://datasets/allenai/wildguardmix/train/wildguard_train.parquet"
    )
    test = pd.read_parquet(
        "hf://datasets/allenai/wildguardmix/test/wildguard_test.parquet"
    )
    df = pd.concat([train, test], ignore_index=True)
    df = df.rename(columns={"prompt": "text"})
    # Keep original columns needed for label mapping
    df["source"] = "wildguardmix"
    df["original_label"] = df.apply(
        lambda r: _wildguard_label(r), axis=1
    )
    return df[["text", "source", "original_label"]]


def _wildguard_label(row) -> str:
    """Map WildGuardMix row to a label string for downstream mapping."""
    if row.get("prompt_harm_label") == "unharmful":
        return "unharmful"
    # harmful or adversarial — encode both flags
    adv = "adversarial" if row.get("adversarial") else "nonadversarial"
    return f"harmful_{adv}"


def load_awesome_chatgpt_prompts() -> pd.DataFrame:
    """fka/awesome-chatgpt-prompts — all benign role-play prompts."""
    ds = load_dataset("fka/awesome-chatgpt-prompts")
    df = ds["train"].to_pandas()
    df = df.rename(columns={"prompt": "text"})
    df["source"] = "awesome_chatgpt"
    df["original_label"] = "benign"
    return df[["text", "source", "original_label"]]


# ── Registry ───────────────────────────────────────────────────────────────

# Phase A: minimal viable dataset
MVP_LOADERS = {
    "deepset": load_deepset,
    "jailbreakv28k": load_jailbreakv28k,
}

# Phase B: full dataset
ALL_LOADERS = {
    **MVP_LOADERS,
    "hackaprompt": load_hackaprompt,
    "wildguardmix": load_wildguardmix,
    "awesome_chatgpt": load_awesome_chatgpt_prompts,
}

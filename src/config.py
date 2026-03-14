"""
Central configuration for the AGL project.
Paths, hyperparameters, label maps, and dataset metadata.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ── Label map ──────────────────────────────────────────────────────────────
LABEL2ID = {
    "Benign": 0,
    "Injection": 1,
    "Jailbreak": 2,
    "Exfiltration": 3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)
LABEL_NAMES = list(LABEL2ID.keys())

# ── Model / training hyperparameters ───────────────────────────────────────
MODEL_NAME = "roberta-base"
MAX_SEQ_LENGTH = 256
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2  # effective batch size = 32
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 2
SEED = 42

# ── Anomaly detection ─────────────────────────────────────────────────────
PCA_COMPONENTS = 100
# Calibrated on val set so 95% of in-distribution samples are below threshold
OOD_RECALL_TARGET = 0.95

# ── Dataset split ratios ──────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── Dataset sources (HuggingFace repo IDs) ────────────────────────────────
HF_DATASETS = {
    "deepset": "deepset/prompt-injections",
    "jailbreakv28k": "JailbreakV-28K/JailBreakV-28k",
    "lakera_mosaic": "Lakera/mosaic_prompt_injection",
    "hackaprompt": "hackaprompt/hackaprompt-dataset",
    "wildguardmix": "allenai/wildguardmix",
    "awesome_chatgpt": "fka/awesome-chatgpt-prompts",
    "verazuo": "verazuo/jailbreak_llms",
}

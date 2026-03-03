#!/usr/bin/env python3
"""
Download all candidate datasets for the AGL Capstone Project.
Run this script from the repo root: python3 scripts/download_datasets.py
"""

import subprocess
import sys

def ensure_deps():
    """Install huggingface_hub if not present."""
    try:
        import huggingface_hub
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "--quiet"])

def download_dataset(repo_id, local_dir):
    from huggingface_hub import snapshot_download
    print(f"\n{'='*60}")
    print(f"Downloading: {repo_id}")
    print(f"  -> {local_dir}")
    print(f"{'='*60}")
    try:
        snapshot_download(repo_id, repo_type="dataset", local_dir=local_dir)
        print(f"  DONE: {repo_id}")
        return True
    except Exception as e:
        print(f"  FAILED: {repo_id} — {e}")
        return False

def main():
    ensure_deps()

    datasets = [
        # Priority datasets
        ("deepset/prompt-injections",                       "data/raw/deepset-prompt-injections"),
        ("Lakera/mosaic_prompt_injection",                  "data/raw/lakera-mosaic-prompt-injection"),
        ("hackaprompt/hackaprompt-dataset",                  "data/raw/hackaprompt"),
        ("JailbreakV-28K/JailBreakV-28k",                   "data/raw/jailbreakv-28k"),
        ("GeneralAnalysis/GA_Jailbreak_Benchmark",          "data/raw/ga-jailbreak-benchmark"),
        # Additional sources
        ("rubend18/ChatGPT-Jailbreak-Prompts",              "data/raw/chatgpt-jailbreak-prompts"),
        ("Harelix/Prompt-Injection-Mixed-Techniques-2024",  "data/raw/harelix-injection-mixed"),
        ("nvidia/Aegis-AI-Content-Safety-Dataset-1.0",      "data/raw/nvidia-aegis"),
        ("fka/awesome-chatgpt-prompts",                     "data/raw/awesome-chatgpt-prompts"),
        ("jackhhao/jailbreak-classification",               "data/raw/jackhhao-jailbreak-classification"),
    ]

    results = {"success": [], "failed": []}

    for repo_id, local_dir in datasets:
        ok = download_dataset(repo_id, local_dir)
        if ok:
            results["success"].append(repo_id)
        else:
            results["failed"].append(repo_id)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Downloaded: {len(results['success'])}/{len(datasets)}")
    for name in results["success"]:
        print(f"    ✓ {name}")
    if results["failed"]:
        print(f"  Failed: {len(results['failed'])}")
        for name in results["failed"]:
            print(f"    ✗ {name}")

if __name__ == "__main__":
    main()

"""
Parse L1B3RT4S jailbreak prompt repository into structured training data.

Reads .mkd files from the L1B3RT4S repo, extracts individual jailbreak prompts
separated by markdown headers, and produces a labeled CSV for training.

Usage:
    python -m src.data.parse_l1b3rt4s --input-dir /path/to/L1B3RT4S-main --output data/processed/l1b3rt4s_prompts.csv
"""

import argparse
import csv
import re
from pathlib import Path


# Vendor file → target_vendor mapping
VENDOR_MAP = {
    "ANTHROPIC": "anthropic",
    "OPENAI": "openai",
    "CHATGPT": "openai",
    "GOOGLE": "google",
    "META": "meta",
    "MICROSOFT": "microsoft",
    "APPLE": "apple",
    "XAI": "xai",
    "DEEPSEEK": "deepseek",
    "ALIBABA": "alibaba",
    "PERPLEXITY": "perplexity",
    "MISTRAL": "mistral",
    "COHERE": "cohere",
    "CURSOR": "cursor",
    "BRAVE": "brave",
    "GROK-MEGA": "xai",
    "MIDJOURNEY": "midjourney",
    "NVIDIA": "nvidia",
    "INFLECTION": "inflection",
    "REKA": "reka",
    "HUME": "hume",
    "INCEPTION": "inception",
    "LIQUIDAI": "liquidai",
    "MOONSHOT": "moonshot",
    "MULTION": "multion",
    "NOUS": "nous",
    "REFLECTION": "reflection",
    "WINDSURF": "windsurf",
    "FETCHAI": "fetchai",
    "GRAYSWAN": "grayswan",
    "ZAI": "zai",
    "ZYPHRA": "zyphra",
    "AAA": "multi",
    "1337": "multi",
    "TOKEN80M8": "multi",
    "TOKENADE": "multi",
    "-MISCELLANEOUS-": "multi",
    "SYSTEMPROMPTS": "multi",
}

# Minimum character length to consider a block a valid prompt
MIN_PROMPT_LENGTH = 50


def extract_prompts_from_mkd(filepath: Path) -> list[dict]:
    """Extract individual jailbreak prompts from a .mkd file.

    Prompts are separated by top-level markdown headers (# Header).
    Each header names the target model variant.

    Args:
        filepath: Path to the .mkd file.

    Returns:
        List of dicts with keys: text, target_model, target_vendor, source_file.
    """
    text = filepath.read_text(encoding="utf-8", errors="replace")
    stem = filepath.stem

    vendor = VENDOR_MAP.get(stem, "unknown")

    # Split on top-level headers
    sections = re.split(r"^# (.+)$", text, flags=re.MULTILINE)

    prompts = []

    # sections[0] is content before the first header (often empty)
    if sections[0].strip() and len(sections[0].strip()) >= MIN_PROMPT_LENGTH:
        prompts.append({
            "text": _clean_prompt(sections[0]),
            "target_model": "unknown",
            "target_vendor": vendor,
            "source_file": filepath.name,
        })

    # Iterate header/content pairs
    for i in range(1, len(sections), 2):
        header = sections[i].strip()
        content = sections[i + 1].strip() if i + 1 < len(sections) else ""

        if len(content) < MIN_PROMPT_LENGTH:
            continue

        # Some sections contain multiple prompts separated by sub-conversations
        # (>> user messages). Split those into separate prompts where possible.
        sub_prompts = _split_sub_prompts(content)

        for sp in sub_prompts:
            if len(sp) < MIN_PROMPT_LENGTH:
                continue
            prompts.append({
                "text": _clean_prompt(sp),
                "target_model": header.lower().strip(),
                "target_vendor": vendor,
                "source_file": filepath.name,
            })

    return prompts


def _split_sub_prompts(content: str) -> list[str]:
    """Split a section that may contain multiple distinct prompt templates.

    Heuristic: if the section has multiple divider patterns or >> markers
    indicating separate prompts, split them. Otherwise return as-is.
    """
    # Check for clear dividers between prompts
    divider_patterns = [
        r"\n---+\n",
        r"\n\*{3,}\n",
        r"\n-\.-\.-",
    ]

    for pattern in divider_patterns:
        parts = re.split(pattern, content)
        if len(parts) > 1:
            return [p.strip() for p in parts if p.strip()]

    return [content]


def _clean_prompt(text: str) -> str:
    """Clean a raw prompt block for use as training data."""
    # Remove leading/trailing whitespace
    text = text.strip()

    # Remove markdown image references
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

    # Collapse excessive newlines
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text


def parse_repository(input_dir: Path) -> list[dict]:
    """Parse all .mkd files in the L1B3RT4S repository.

    Args:
        input_dir: Path to the L1B3RT4S-main directory.

    Returns:
        List of all extracted prompt dicts.
    """
    all_prompts = []

    for mkd_file in sorted(input_dir.glob("*.mkd")):
        prompts = extract_prompts_from_mkd(mkd_file)
        all_prompts.extend(prompts)
        print(f"  {mkd_file.name}: {len(prompts)} prompts extracted")

    # Also parse the motherload text file
    motherload = input_dir / "#MOTHERLOAD.txt"
    if motherload.exists():
        text = motherload.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) >= MIN_PROMPT_LENGTH:
            all_prompts.append({
                "text": _clean_prompt(text),
                "target_model": "multi",
                "target_vendor": "multi",
                "source_file": motherload.name,
            })

    return all_prompts


def save_prompts_csv(prompts: list[dict], output_path: Path) -> None:
    """Save extracted prompts to CSV with training labels.

    All prompts from L1B3RT4S are labeled as:
      - label: 1 (malicious) for binary classification
      - threat_type: jailbreak for multi-class classification

    Args:
        prompts: List of prompt dicts.
        output_path: Where to write the CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "text",
        "label",
        "threat_type",
        "target_model",
        "target_vendor",
        "source",
        "source_file",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for prompt in prompts:
            writer.writerow({
                "text": prompt["text"],
                "label": 1,  # malicious
                "threat_type": "jailbreak",
                "target_model": prompt["target_model"],
                "target_vendor": prompt["target_vendor"],
                "source": "l1b3rt4s",
                "source_file": prompt["source_file"],
            })

    print(f"\nSaved {len(prompts)} prompts to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse L1B3RT4S jailbreak repository into training CSV"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to L1B3RT4S-main directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/l1b3rt4s_prompts.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: {input_dir} does not exist")
        return

    print(f"Parsing L1B3RT4S repository from {input_dir}...")
    prompts = parse_repository(input_dir)
    print(f"\nTotal prompts extracted: {len(prompts)}")

    save_prompts_csv(prompts, Path(args.output))


if __name__ == "__main__":
    main()

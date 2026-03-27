import re


def normalize_prompt(prompt: str, max_length: int = 20000) -> str:
    """
    Basic text normalization that should stay stable between training and inference.
    Keep this aligned with the notebook export logic if you used a specific normalization pipeline.
    """
    if prompt is None:
        prompt = ""

    text = str(prompt)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    if max_length > 0:
        text = text[:max_length]

    return text

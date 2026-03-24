from __future__ import annotations


def normalize_prompt(prompt: str) -> str:
    normalized = " ".join(prompt.strip().split())
    if not normalized:
        raise ValueError("Prompt must not be empty.")
    if not normalized.endswith("."):
        normalized = f"{normalized}."
    return normalized

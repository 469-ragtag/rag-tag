from __future__ import annotations

import re

_SEPARATOR_RE = re.compile(r"[_./-]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")
_GROUND_RE = re.compile(r"\bground\s*(?:floor|level)\b|\bgroundfloor\b|\bgroundlevel\b")
_BASEMENT_NUMBER_RE = re.compile(r"\bbasement\s*0*([0-9]+)\b")
_BASEMENT_RE = re.compile(r"\bbasement\b")
_LEVEL_NUMBER_RE = re.compile(r"\b(?:level|storey|story|floor)\s*0*([0-9]+)\b")


def normalize_level_text(value: str | None) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    text = _SEPARATOR_RE.sub(" ", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text or None


def canonicalize_level(value: str | None) -> str | None:
    normalized = normalize_level_text(value)
    if normalized is None:
        return None

    if _GROUND_RE.search(normalized):
        return "ground floor"

    basement_number = _BASEMENT_NUMBER_RE.search(normalized)
    if basement_number is not None:
        return f"basement {int(basement_number.group(1))}"

    if _BASEMENT_RE.search(normalized):
        return "basement"

    level_number = _LEVEL_NUMBER_RE.search(normalized)
    if level_number is not None:
        return f"level {int(level_number.group(1))}"

    return normalized

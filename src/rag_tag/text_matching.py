from __future__ import annotations

import re
from typing import Any, Iterable

_TEXT_MATCH_TERM_RE = re.compile(r"[a-z0-9]+")


def text_match_terms(text: str) -> tuple[str, ...]:
    """Return ordered unique lowercase alphanumeric terms."""
    ordered_terms: list[str] = []
    seen_terms: set[str] = set()
    for term in _TEXT_MATCH_TERM_RE.findall(text.lower()):
        if term in seen_terms:
            continue
        seen_terms.add(term)
        ordered_terms.append(term)
    return tuple(ordered_terms)


def normalize_text_match_text(text: str) -> str:
    """Return a canonical token-normalized text string."""
    return " ".join(text_match_terms(text))


def combined_text_match_terms(values: Iterable[Any]) -> tuple[str, ...]:
    """Return ordered unique terms across a sequence of values."""
    ordered_terms: list[str] = []
    seen_terms: set[str] = set()
    for value in values:
        if value in {None, "", "None"}:
            continue
        for term in _TEXT_MATCH_TERM_RE.findall(str(value).lower()):
            if term in seen_terms:
                continue
            seen_terms.add(term)
            ordered_terms.append(term)
    return tuple(ordered_terms)


def text_matches_query(
    searchable_text: str,
    query: str | tuple[str, ...],
) -> bool:
    """Return True when every query term is present as a token."""
    query_terms = query if isinstance(query, tuple) else text_match_terms(query)
    if not query_terms:
        return False
    searchable_term_set = set(text_match_terms(searchable_text))
    return all(term in searchable_term_set for term in query_terms)

"""Deterministic, dependency-light tokenisation for public benchmarks."""

from __future__ import annotations

import re

# Unicode letters (without underscore), plus numbers. Apostrophes/hyphens inside
# words are retained as boundaries by the regex and therefore split consistently.
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*|[^\W_]+", re.UNICODE)


def tokenise(text: str) -> list[str]:
    """Case-fold Unicode text into a deterministic lexical token stream."""
    return [match.group(0).casefold() for match in _TOKEN.finditer(text or "")]


def tokenise_with_positions(text: str) -> list[tuple[str, int]]:
    """Return every token with its zero-based position; no tokens are dropped."""
    return [(token, position) for position, token in enumerate(tokenise(text))]

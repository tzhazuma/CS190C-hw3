from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


HASH_ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")
FRACTION_RE = re.compile(r"[-+]?\d+\s*/\s*\d+")
NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\.\d+")


def _normalize_decimal(value: Decimal) -> str:
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return "0" if normalized == "-0" else normalized


def normalize_numeric_string(text: str | None) -> str | None:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None

    fraction_matches = list(FRACTION_RE.finditer(stripped))
    number_matches = list(NUMBER_RE.finditer(stripped))
    all_matches = sorted(fraction_matches + number_matches, key=lambda match: match.start())
    if not all_matches:
        return None

    token = all_matches[-1].group(0).replace(",", "")
    try:
        if "/" in token:
            numerator_text, denominator_text = [piece.strip() for piece in token.split("/", maxsplit=1)]
            value = Decimal(numerator_text) / Decimal(denominator_text)
        else:
            value = Decimal(token)
    except (InvalidOperation, ZeroDivisionError):
        return None
    return _normalize_decimal(value)


def extract_final_answer(text: str | None) -> str | None:
    if text is None:
        return None

    hash_matches = HASH_ANSWER_RE.findall(text)
    if hash_matches:
        for candidate in reversed(hash_matches):
            normalized = normalize_numeric_string(candidate)
            if normalized is not None:
                return normalized

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        normalized = normalize_numeric_string(line)
        if normalized is not None:
            return normalized

    return normalize_numeric_string(text)


def answers_match(prediction: str | None, ground_truth: str | None) -> bool:
    return prediction is not None and ground_truth is not None and prediction == ground_truth

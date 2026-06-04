"""Composition type inference for AIHub X-ray research splits.

The logic is intentionally isolated here so dataset-specific naming quirks can
be edited without touching conversion or split code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


UNKNOWN = "unknown"
SINGLE_BASIC = "single_basic"
SINGLE_WITH_NON_TARGET = "single_with_non_target"
COMPLEX_TARGET = "complex_target"
COMPLEX_WITH_NON_TARGET = "complex_with_non_target"

_KNOWN = [
    SINGLE_BASIC,
    SINGLE_WITH_NON_TARGET,
    COMPLEX_TARGET,
    COMPLEX_WITH_NON_TARGET,
]

_EXACT_PATTERNS = {
    "단일기본": SINGLE_BASIC,
    "single_basic": SINGLE_BASIC,
    "single-basic": SINGLE_BASIC,
    "single basic": SINGLE_BASIC,
    "단일비품목": SINGLE_WITH_NON_TARGET,
    "single_with_non_target": SINGLE_WITH_NON_TARGET,
    "single-with-non-target": SINGLE_WITH_NON_TARGET,
    "single non target": SINGLE_WITH_NON_TARGET,
    "single-non-target": SINGLE_WITH_NON_TARGET,
    "single_non_target": SINGLE_WITH_NON_TARGET,
    "복합품목": COMPLEX_TARGET,
    "complex_target": COMPLEX_TARGET,
    "complex-target": COMPLEX_TARGET,
    "complex target": COMPLEX_TARGET,
    "복합비품목": COMPLEX_WITH_NON_TARGET,
    "complex_with_non_target": COMPLEX_WITH_NON_TARGET,
    "complex-with-non-target": COMPLEX_WITH_NON_TARGET,
    "complex non target": COMPLEX_WITH_NON_TARGET,
    "complex-non-target": COMPLEX_WITH_NON_TARGET,
    "complex_non_target": COMPLEX_WITH_NON_TARGET,
    "pidray_easy": SINGLE_BASIC,
    "pidray-easy": SINGLE_BASIC,
    "easy": SINGLE_BASIC,
    "test_easy": SINGLE_BASIC,
    "easy_test": SINGLE_BASIC,
    "pidray_hard": COMPLEX_TARGET,
    "pidray-hard": COMPLEX_TARGET,
    "hard": COMPLEX_TARGET,
    "test_hard": COMPLEX_TARGET,
    "hard_test": COMPLEX_TARGET,
    "pidray_hidden": COMPLEX_WITH_NON_TARGET,
    "pidray-hidden": COMPLEX_WITH_NON_TARGET,
    "hidden": COMPLEX_WITH_NON_TARGET,
    "test_hidden": COMPLEX_WITH_NON_TARGET,
    "hidden_test": COMPLEX_WITH_NON_TARGET,
    "ol1": SINGLE_BASIC,
    "occlusion level 1": SINGLE_BASIC,
    "ol2": COMPLEX_TARGET,
    "occlusion level 2": COMPLEX_TARGET,
    "ol3": COMPLEX_WITH_NON_TARGET,
    "occlusion level 3": COMPLEX_WITH_NON_TARGET,
}


def known_compositions() -> list[str]:
    """Return composition labels used by this project."""

    return list(_KNOWN)


def is_complex_composition(composition_type: str) -> bool:
    """Return True for complex baggage compositions."""

    return composition_type in {COMPLEX_TARGET, COMPLEX_WITH_NON_TARGET}


def _metadata_text(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""

    parts: list[str] = []

    def collect(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                collect(item)
        else:
            parts.append(str(value))

    collect(metadata)
    return " ".join(parts)


def infer_composition_type(path: str, metadata: dict[str, Any] | None = None) -> str:
    """Infer composition type from a path and optional metadata.

    Returns one of:
    single_basic, single_with_non_target, complex_target,
    complex_with_non_target, or unknown.
    """

    raw = f"{Path(path).as_posix()} {_metadata_text(metadata)}"
    text = raw.lower().replace("\\", "/")
    compact = (
        text.replace("_", "-")
        .replace("/", " ")
        .replace(".", " ")
        .replace("(", " ")
        .replace(")", " ")
    )

    for pattern, composition in _EXACT_PATTERNS.items():
        if pattern.lower() in text or pattern.lower() in compact:
            return composition

    has_korean_complex = "복합" in text
    has_korean_single = "단일" in text
    has_korean_non_target = "비품목" in text

    if has_korean_complex and has_korean_non_target:
        return COMPLEX_WITH_NON_TARGET
    if has_korean_complex:
        return COMPLEX_TARGET
    if has_korean_single and has_korean_non_target:
        return SINGLE_WITH_NON_TARGET
    if has_korean_single:
        return SINGLE_BASIC

    normalized = compact.replace("_", "-")
    has_non_target = any(
        token in normalized
        for token in [
            "non-target",
            "non target",
            "nontarget",
            "with-non-target",
            "with non target",
            "non prohibited",
            "non-prohibited",
        ]
    )
    has_complex = "complex" in normalized
    has_single = "single" in normalized

    if has_complex and has_non_target:
        return COMPLEX_WITH_NON_TARGET
    if has_complex:
        return COMPLEX_TARGET
    if has_single and has_non_target:
        return SINGLE_WITH_NON_TARGET
    if has_single:
        return SINGLE_BASIC

    return UNKNOWN

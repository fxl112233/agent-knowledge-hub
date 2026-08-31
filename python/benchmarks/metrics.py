"""Dependency-light benchmark metrics and paired bootstrap confidence intervals."""

from __future__ import annotations

import math
import random
import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass


def normalize_answer(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^\w\u3400-\u9fff.%-]+", "", text)


def exact_match(prediction: object, reference: object) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def _tokens(value: object) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(value or "")).lower()
    return re.findall(r"[\u3400-\u9fff]|[a-z0-9]+(?:[.%-][a-z0-9]+)*", normalized)


def token_f1(prediction: object, reference: object) -> float:
    predicted, expected = _tokens(prediction), _tokens(reference)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def multihop_official_accuracy(prediction: object, reference: object) -> float:
    """Compatibility metric used by MultiHop-RAG's published QA evaluator."""
    predicted = set(str(prediction or "").lower().split())
    expected = set(str(reference or "").lower().split())
    return float(bool(predicted & expected))


def citation_precision_recall(
    cited: Iterable[str], expected: Iterable[str]
) -> tuple[float | None, float | None]:
    """Return citation precision/recall, excluding cases without gold evidence."""
    cited_set, expected_set = set(cited), set(expected)
    if not expected_set:
        return None, None
    hits = len(cited_set & expected_set)
    precision = hits / len(cited_set) if cited_set else 0.0
    return precision, hits / len(expected_set)


def character_error_rate(prediction: object, reference: object) -> float:
    predicted = normalize_answer(prediction)
    expected = normalize_answer(reference)
    if not expected:
        return 0.0 if not predicted else 1.0
    previous = list(range(len(predicted) + 1))
    for row, expected_char in enumerate(expected, start=1):
        current = [row]
        for column, predicted_char in enumerate(predicted, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_char != predicted_char),
                )
            )
        previous = current
    return previous[-1] / len(expected)


def numeric_equal(prediction: object, reference: object, tolerance: float = 1e-4) -> float:
    def number(value: object) -> tuple[float, int]:
        match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", str(value))
        if not match:
            raise ValueError("no number")
        raw = match.group(0).replace(",", "")
        decimals = len(raw.rsplit(".", 1)[1]) if "." in raw else 0
        return float(raw), decimals

    try:
        (left, _prediction_decimals), (right, reference_decimals) = (
            number(prediction),
            number(reference),
        )
    except ValueError:
        return exact_match(prediction, reference)
    # Dataset references such as TAT-QA are often rounded. Accept values that
    # round to the published precision, while keeping a small relative floor.
    rounding_tolerance = 0.5 * (10 ** (-reference_decimals)) if reference_decimals else tolerance
    return float(
        math.isclose(
            left,
            right,
            rel_tol=tolerance,
            abs_tol=max(tolerance, rounding_tolerance),
        )
    )


def evidence_metrics(retrieved: Iterable[str], expected: Iterable[str]) -> dict[str, float]:
    retrieved_set, expected_set = set(retrieved), set(expected)
    if not expected_set:
        return {"evidence_recall": 1.0, "all_evidence_hit": 1.0}
    hits = len(retrieved_set & expected_set)
    return {
        "evidence_recall": hits / len(expected_set),
        "all_evidence_hit": float(expected_set <= retrieved_set),
    }


def binary_f1(predictions: Sequence[bool], references: Sequence[bool]) -> float:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    pairs = list(zip(predictions, references, strict=True))
    true_positive = sum(prediction and reference for prediction, reference in pairs)
    false_positive = sum(prediction and not reference for prediction, reference in pairs)
    false_negative = sum(not prediction and reference for prediction, reference in pairs)
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 1.0


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    samples: int


def paired_bootstrap_difference[T](
    baseline: Sequence[T],
    candidate: Sequence[T],
    metric: Callable[[Sequence[T]], float] = statistics.fmean,
    *,
    iterations: int = 2000,
    seed: int = 42,
) -> ConfidenceInterval:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired non-empty samples of equal length are required")
    differences = [float(right) - float(left) for left, right in zip(baseline, candidate, strict=True)]
    estimate = metric(differences)
    generator = random.Random(seed)
    bootstrapped = []
    for _ in range(iterations):
        sample = [differences[generator.randrange(len(differences))] for _ in differences]
        bootstrapped.append(metric(sample))
    return ConfidenceInterval(
        estimate=estimate,
        lower=percentile(bootstrapped, 0.025),
        upper=percentile(bootstrapped, 0.975),
        samples=len(differences),
    )

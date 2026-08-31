from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.metrics import (
    binary_f1,
    character_error_rate,
    citation_precision_recall,
    evidence_metrics,
    exact_match,
    multihop_official_accuracy,
    numeric_equal,
    paired_bootstrap_difference,
    percentile,
    token_f1,
)
from benchmarks.runner import ResultCache


def test_text_and_numeric_metrics() -> None:
    assert exact_match("Revenue: 1,000", "revenue 1000") == 1
    assert token_f1("北京公司 revenue", "北京 revenue") > 0.7
    assert character_error_rate("abcd", "abxd") == 0.25
    assert numeric_equal("1,000.00 yuan", 1000) == 1
    assert numeric_equal("0.306149", "0.31") == 1
    assert numeric_equal("0.304", "0.31") == 0
    assert numeric_equal("1000.4", "1000") == 0
    assert numeric_equal("unknown", "unknown") == 1


def test_retrieval_and_binary_metrics() -> None:
    assert evidence_metrics(["a", "x"], ["a", "b"]) == {
        "evidence_recall": 0.5,
        "all_evidence_hit": 0.0,
    }
    assert binary_f1([True, False, True], [True, True, False]) == 0.5
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert citation_precision_recall(["a", "x"], ["a", "b"]) == (0.5, 0.5)
    assert citation_precision_recall([], []) == (None, None)
    assert multihop_official_accuracy("The answer is Atlas", "Atlas") == 1
    assert multihop_official_accuracy("Yes", "Same") == 0


def test_bootstrap_is_deterministic_and_validates_pairs() -> None:
    interval = paired_bootstrap_difference([0, 0, 1], [1, 1, 1], iterations=200, seed=42)
    assert interval.estimate == pytest.approx(2 / 3)
    assert interval.lower <= interval.estimate <= interval.upper
    with pytest.raises(ValueError):
        paired_bootstrap_difference([], [])


def test_result_cache_no_resume_starts_a_clean_file(tmp_path: Path) -> None:
    path = tmp_path / "result.jsonl"
    path.write_text('{"cache_key":"old"}\n', encoding="utf-8")
    cache = ResultCache(path, resume=False)
    assert not cache.has("old")
    assert not path.exists()
    cache.append("new", {"value": 1})
    assert cache.has("new")
    assert cache.get("new") == {"cache_key": "new", "value": 1}

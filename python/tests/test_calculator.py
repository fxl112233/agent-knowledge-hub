from __future__ import annotations

import pytest

from utils.calculator import safe_calculate


def test_safe_arithmetic() -> None:
    assert safe_calculate("(10 + 2) * 3 / 4") == 9
    assert safe_calculate("2 ** 8") == 256


@pytest.mark.parametrize("expression", ["__import__('os')", "x + 1", "10 ** 1000", "1 << 2"])
def test_rejects_unsafe_or_unbounded_expressions(expression: str) -> None:
    with pytest.raises((SyntaxError, ValueError, OverflowError)):
        safe_calculate(expression)

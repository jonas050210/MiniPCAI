"""Tests for the safe arithmetic engine, especially unsafe inputs."""

from __future__ import annotations

import pytest

from minipcai.calc_engine import CalcError, evaluate, format_number, validate


class TestValidExpressions:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("2+3", "5"),
            ("2+3*4", "14"),
            ("(2+3)*4", "20"),
            ("10/4", "2.5"),
            ("2**10", "1024"),
            ("-5+10", "5"),
            ("100 % 7", "2"),
            ("100 // 7", "14"),
            ("0.5 + 0.5", "1"),
            ("2**1000", str(2**1000)),
            ("17.5 + 2.5", "20"),
            ("3.5 * 2", "7"),
        ],
    )
    def test_evaluates(self, expression, expected):
        assert evaluate(expression) == expected

    def test_validate_returns_ast(self):
        tree = validate("1+1")
        assert tree is not None


class TestUnsafeInputs:
    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os')",          # Python constructs
            "open('x')",
            "1 if True else 2",
            "[1,2]",
            "{'a': 1}",
            "x + 1",                     # names
            "f(1)",
            "1 << 2",                    # shift operators
            "~5",                        # bitwise not
            "2**2**2**2**2",             # exponent is not a small literal
            "2**(2+3)",                  # computed exponent
            "2 ** 1001",                 # exponent too large
            "9" * 20,                    # literal too large
            "1" + "+1" * 100,            # too many nodes
            "1e400",                     # scientific notation not allowed
            "abc",
            "",
            "   ",
            "(2+3",                      # syntax error
            "2+3)",
        ],
    )
    def test_rejects_expression(self, expression):
        with pytest.raises(CalcError):
            evaluate(expression)

    def test_expression_too_long(self):
        with pytest.raises(CalcError, match="too long"):
            evaluate("1+1" * 50)

    def test_division_by_zero(self):
        with pytest.raises(CalcError, match="division by zero"):
            evaluate("10 / 0")

    def test_modulo_by_zero(self):
        with pytest.raises(CalcError, match="modulo by zero"):
            evaluate("10 % 0")

    def test_result_must_be_finite(self):
        # Division can overflow to infinity even with allowed literals.
        with pytest.raises(CalcError, match="finite"):
            evaluate("(2**1000) / 0.000000000001")

    def test_float_power_overflow_is_a_clean_error(self):
        # A float power beyond the double range raises OverflowError in
        # CPython; it must surface as a CalcError, never as a crash.
        with pytest.raises(CalcError, match="too large"):
            evaluate("999999999999999.0 ** 1000")

    def test_huge_integer_result_is_refused(self):
        # int -> str conversion is quadratic; such results are rejected
        # before formatting instead of hanging or producing a huge string.
        with pytest.raises(CalcError, match="digits"):
            evaluate("999999999999999 ** 1000")

    def test_large_but_reasonable_result_still_works(self):
        assert evaluate("2**100") == str(2**100)

    def test_computed_exponent_rejected(self):
        # Exponent chains like 2**(2**1000) are structurally impossible.
        with pytest.raises(CalcError, match="exponent"):
            evaluate("2 ** (2 * 3)")


class TestFormatting:
    def test_integral_float_becomes_int(self):
        assert format_number(4.0) == "4"

    def test_float_trimmed(self):
        assert format_number(100 / 7) == "14.2857142857"

    def test_integer_unchanged(self):
        assert format_number(48) == "48"

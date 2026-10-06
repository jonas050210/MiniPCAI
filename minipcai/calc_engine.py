"""Safe arithmetic evaluation for the ``calc`` intent.

Expressions are extracted by ``textutils.extract_math_expression`` and then
strictly validated here against an AST whitelist before evaluation:

* only numeric literals, ``+ - * / // % **``, unary ``+/-`` and parentheses;
* no names, calls, attributes, subscripts or any other Python construct;
* hard limits on expression length, node count, literal magnitude and
  exponents to make denial-of-service inputs impossible;
* division/modulo by zero and non-finite results are rejected as errors.

``eval``/``exec`` are never used; evaluation walks the validated AST directly.
"""

from __future__ import annotations

import ast
import math

from minipcai.config import (
    CALC_MAX_EXPONENT,
    CALC_MAX_EXPRESSION_LENGTH,
    CALC_MAX_LITERAL,
    CALC_MAX_NODES,
)

_ALLOWED_CHARS = frozenset("0123456789+-*/().% ")

_ALLOWED_BINOPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


class CalcError(ValueError):
    """Raised when an expression is invalid, unsafe or cannot be evaluated."""


def validate(expression: str) -> ast.Expression:
    """Validate an arithmetic expression and return its parsed AST."""
    expression = expression.strip()
    if not expression:
        raise CalcError("empty expression")
    if len(expression) > CALC_MAX_EXPRESSION_LENGTH:
        raise CalcError("expression is too long")
    if any(ch not in _ALLOWED_CHARS for ch in expression):
        raise CalcError("expression contains forbidden characters")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"invalid arithmetic expression: {exc.msg}") from exc

    node_count = 0
    for node in ast.walk(tree):
        node_count += 1
        if node_count > CALC_MAX_NODES:
            raise CalcError("expression is too complex")
        if isinstance(node, ast.Expression):
            continue
        # ast.walk also yields operator/context helper nodes; their validity is
        # checked within the BinOp/UnaryOp branches below.
        if isinstance(node, (ast.operator, ast.unaryop, ast.expr_context)):
            continue
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINOPS):
                raise CalcError("operator is not allowed")
            if isinstance(node.op, ast.Pow):
                exponent = node.right
                if not isinstance(exponent, ast.Constant) or not isinstance(
                    exponent.value, (int, float)
                ) or abs(exponent.value) > CALC_MAX_EXPONENT:
                    raise CalcError("exponent must be a small literal number")
            continue
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARYOPS):
                raise CalcError("unary operator is not allowed")
            continue
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise CalcError("only numeric literals are allowed")
            if abs(node.value) > CALC_MAX_LITERAL:
                raise CalcError("number is too large")
            continue
        raise CalcError(f"construct is not allowed: {type(node).__name__}")

    return tree


def _eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        return value
    if isinstance(node, ast.BinOp):
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise CalcError("division by zero")
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise CalcError("division by zero")
            return left // right
        if isinstance(node.op, ast.Mod):
            if right == 0:
                raise CalcError("modulo by zero")
            return left % right
        if isinstance(node.op, ast.Pow):
            return left**right
    raise CalcError("unsupported expression node")


def format_number(value: float | int) -> str:
    """Format a numeric result for display (int when integral)."""
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        raise CalcError("result is not a finite number")
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.10f}".rstrip("0").rstrip(".")


def evaluate(expression: str) -> str:
    """Validate and evaluate an arithmetic expression, returning the result."""
    tree = validate(expression)
    result = _eval(tree)
    if isinstance(result, float) and not math.isfinite(result):
        raise CalcError("result is not a finite number")
    return format_number(result)

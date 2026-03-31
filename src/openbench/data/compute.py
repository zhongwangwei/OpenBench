"""Compute expression executor for derived variables.

Evaluates Python expressions from model profile YAML to compute
derived variables from xarray Datasets.

Supports:
- Simple: "ds['f_xy_rain'] + ds['f_xy_snow']"
- Multi-step with semicolons: "total = ds['a'] + ds['b']; total / 100"
- All xarray/numpy operations available

Security: expressions are validated against an allowlist of safe AST node types
before evaluation to prevent code injection.
"""

from __future__ import annotations

import ast
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_SAFE_NODES = {
    ast.Expression, ast.Module,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.FloorDiv, ast.Mod,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Subscript, ast.Attribute, ast.Index, ast.Slice,
    ast.Name, ast.Load, ast.Constant, ast.Num, ast.Str,
    ast.Call, ast.keyword, ast.Starred,
    ast.Tuple, ast.List,
    ast.IfExp,
}


def _validate_expression(expr: str) -> None:
    """Validate that expression only contains safe AST nodes."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ComputeError(f"Invalid expression syntax: {e}") from e

    for node in ast.walk(tree):
        if type(node) not in _SAFE_NODES:
            raise ComputeError(
                f"Unsafe expression: {type(node).__name__} not allowed. "
                f"Only arithmetic, subscript, attribute access, and function calls are permitted."
            )


def execute_compute(ds: Any, expression: str, var_name: str = "") -> Any:
    """Execute a compute expression against an xarray Dataset.

    Args:
        ds: xarray Dataset with source variables.
        expression: Python expression string. May contain semicolons
            for intermediate assignments. The last expression is the result.
        var_name: Variable name for logging.

    Returns:
        Computed xarray DataArray.

    Raises:
        ComputeError: If expression evaluation fails.
    """
    if not expression or not expression.strip():
        raise ComputeError(f"Empty compute expression for {var_name}")

    # Split on semicolons for multi-step expressions
    # "total_area = ds['a'] + ds['b']; (prod / total_area) * factor"
    steps = [s.strip() for s in expression.split(";") if s.strip()]

    # Build execution namespace
    namespace: dict[str, Any] = {
        "ds": ds,
        "np": np,
    }

    try:
        # Execute intermediate assignments
        for step in steps[:-1]:
            if "=" in step and not step.strip().startswith("("):
                # Assignment: "total_area = ds['a'] + ds['b']"
                var, expr = step.split("=", 1)
                expr_stripped = expr.strip()
                _validate_expression(expr_stripped)
                namespace[var.strip()] = eval(expr_stripped, {"__builtins__": {}}, namespace)  # noqa: S307
            else:
                # Expression without assignment (side effect)
                _validate_expression(step)
                eval(step, {"__builtins__": {}}, namespace)  # noqa: S307

        # Evaluate final expression — this is the return value
        _validate_expression(steps[-1])
        result = eval(steps[-1], {"__builtins__": {}}, namespace)  # noqa: S307

        logger.debug("Computed %s successfully", var_name)
        return result

    except KeyError as e:
        raise ComputeError(
            f"Variable {e} not found in dataset when computing {var_name}. "
            f"Available: {list(ds.data_vars)[:10]}..."
        ) from e
    except Exception as e:
        raise ComputeError(f"Failed to compute {var_name}: {e}") from e


class ComputeError(Exception):
    """Raised when a compute expression fails."""

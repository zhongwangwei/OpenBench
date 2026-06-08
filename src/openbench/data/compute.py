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
from collections.abc import Iterable
from typing import Any

import numpy as np
import xarray as xr

from openbench.util.names import get_xarray_key_case_insensitive

logger = logging.getLogger(__name__)


class _CaseInsensitiveDatasetProxy:
    """Exact-first case-insensitive ``ds['var']`` proxy for compute expressions."""

    def __init__(self, dataset: Any):
        self._dataset = dataset

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            actual = get_xarray_key_case_insensitive(self._dataset, key)
            if actual is not None:
                return self._dataset[actual]
        return self._dataset[key]

    def __contains__(self, key: Any) -> bool:
        """Support profile expressions such as ``'var' in ds``.

        Several bundled model profiles use membership checks to choose between
        alternative native variable names.  Without ``__contains__``, Python
        falls back to integer iteration through ``__getitem__`` and xarray raises
        on ``ds[0]`` before the expression can select the valid branch.
        """
        if not isinstance(key, str):
            return False
        return get_xarray_key_case_insensitive(self._dataset, key) is not None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)


_SAFE_NODES = {
    ast.Expression,
    ast.Module,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.FloorDiv,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Subscript,
    ast.Attribute,
    ast.Index,
    ast.Slice,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Call,
    ast.keyword,
    ast.Starred,
    ast.Tuple,
    ast.List,
    ast.IfExp,
}

# Names that may legitimately appear as the root of an attribute / call chain.
# Anything else (especially names starting with `_`) is rejected — this
# prevents `__import__(...)` style escapes even though __builtins__ is empty.
_SAFE_ROOT_NAMES = frozenset({"ds", "np", "xr"})


def _validate_expression(expr: str, allowed_names: Iterable[str] | None = None) -> None:
    """Validate that expression only contains safe AST nodes.

    In addition to the node-type allowlist, this rejects private/dunder
    attribute access and any free identifier that is not present in the
    evaluation namespace.  Callers that expose extra arrays (for example
    fallback conversion expressions using ``value`` and peer variables)
    must pass those names via ``allowed_names``.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ComputeError(f"Invalid expression syntax: {e}") from e

    allowed = set(_SAFE_ROOT_NAMES)
    if allowed_names is not None:
        allowed.update(str(name) for name in allowed_names)

    for node in ast.walk(tree):
        if type(node) not in _SAFE_NODES:
            raise ComputeError(
                f"Unsafe expression: {type(node).__name__} not allowed. "
                f"Only arithmetic, subscript, attribute access, and function calls are permitted."
            )
        if isinstance(node, ast.Attribute):
            # Reject dunder / private attribute access. Without this the
            # Call+Attribute combo lets an attacker walk into the runtime
            # via `ds.__class__.__init__.__globals__["os"].system(...)`.
            if node.attr.startswith("_"):
                raise ComputeError(f"Unsafe expression: attribute '{node.attr}' starts with underscore.")
        if isinstance(node, ast.Name):
            if node.id not in allowed:
                raise ComputeError(f"Unsafe expression: identifier '{node.id}' is not allowed.")


def _split_assignment(step: str) -> tuple[str, str] | None:
    """Return ``(target, expression)`` for a simple assignment step."""
    try:
        tree = ast.parse(step, mode="exec")
    except SyntaxError as exc:
        raise ComputeError(f"Invalid expression syntax: {exc}") from exc

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        return None

    assign = tree.body[0]
    if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
        raise ComputeError(f"Invalid assignment target in compute expression: {step}")

    target = assign.targets[0].id
    if target in _SAFE_ROOT_NAMES or target.startswith("_"):
        raise ComputeError(f"Invalid assignment target in compute expression: {target}")

    return target, ast.unparse(assign.value)


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
        "ds": _CaseInsensitiveDatasetProxy(ds),
        "np": np,
        "xr": xr,
    }

    try:
        # Execute intermediate assignments
        for step in steps[:-1]:
            assignment = _split_assignment(step)
            if assignment is not None:
                # Assignment: "total_area = ds['a'] + ds['b']"
                var, expr_stripped = assignment
                _validate_expression(expr_stripped, allowed_names=namespace.keys())
                namespace[var] = eval(expr_stripped, {"__builtins__": {}}, namespace)  # noqa: S307
            else:
                # Expression without assignment (side effect)
                _validate_expression(step, allowed_names=namespace.keys())
                eval(step, {"__builtins__": {}}, namespace)  # noqa: S307

        # Evaluate final expression — this is the return value
        _validate_expression(steps[-1], allowed_names=namespace.keys())
        result = eval(steps[-1], {"__builtins__": {}}, namespace)  # noqa: S307

        logger.debug("Computed %s successfully", var_name)
        return result

    except KeyError as e:
        raise ComputeError(
            f"Variable {e} not found in dataset when computing {var_name}. Available: {list(ds.data_vars)[:10]}..."
        ) from e
    except Exception as e:
        raise ComputeError(f"Failed to compute {var_name}: {e}") from e


class ComputeError(Exception):
    """Raised when a compute expression fails."""

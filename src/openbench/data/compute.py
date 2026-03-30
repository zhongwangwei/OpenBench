"""Compute expression executor for derived variables.

Evaluates Python expressions from model profile YAML to compute
derived variables from xarray Datasets.

Supports:
- Simple: "ds['f_xy_rain'] + ds['f_xy_snow']"
- Multi-step with semicolons: "total = ds['a'] + ds['b']; total / 100"
- All xarray/numpy operations available

Security: only used with trusted model profile YAML, not user input.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


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
                namespace[var.strip()] = eval(expr.strip(), {"__builtins__": {}}, namespace)  # noqa: S307
            else:
                # Expression without assignment (side effect)
                eval(step, {"__builtins__": {}}, namespace)  # noqa: S307

        # Evaluate final expression — this is the return value
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

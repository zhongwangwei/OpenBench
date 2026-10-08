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
from functools import lru_cache
from typing import Any

import numpy as np
import xarray as xr

from openbench.util.names import get_xarray_key_case_insensitive

logger = logging.getLogger(__name__)


def _parsed_steps(expression: Any) -> list[ast.AST]:
    """Parse each ``;`` step the way :func:`execute_compute` runs it.

    Steps are stripped first, so leading whitespace or an indented line after
    ``;`` (both fine at evaluation) does not hide the expression's inputs. A
    step that does not parse is skipped; evaluation reports it as an error.
    """
    trees = []
    for step in str(expression or "").split(";"):
        step = step.strip()
        if not step:
            continue
        try:
            trees.append(ast.parse(step))
        except SyntaxError:
            continue
    return trees


# Dataset attributes that describe the dataset without reading a variable.
_METADATA_ATTRIBUTES = frozenset({"dims", "sizes", "attrs"})


def _string_constant(node: Any) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _listed_dataset_use(node: ast.Name, parents: dict[int, ast.AST]) -> bool:
    """Whether this use of ``ds`` reads only what :func:`compute_dependency_names` lists."""
    parent = parents.get(id(node))
    if isinstance(parent, ast.Subscript) and parent.value is node:
        return _string_constant(parent.slice)
    if isinstance(parent, ast.Compare):
        # 'name' in ds tests membership without reading a value.
        return any(
            comparator is node and isinstance(op, (ast.In, ast.NotIn))
            for op, comparator in zip(parent.ops, parent.comparators)
        )
    if not (isinstance(parent, ast.Attribute) and parent.value is node):
        return False
    call = parents.get(id(parent))
    if isinstance(call, ast.Call) and call.func is parent:
        kwargs = {kw.arg: kw.value for kw in call.keywords}
        if parent.attr == "get":
            return _string_constant(call.args[0] if call.args else kwargs.get("key"))
        if parent.attr == "sum_prefix":
            prefix = call.args[0] if call.args else kwargs.get("prefix")
            count = call.args[1] if len(call.args) > 1 else kwargs.get("parts")
            return (
                _string_constant(prefix)
                and isinstance(count, ast.Constant)
                and type(count.value) is int
                and count.value > 0
            )
        return False
    return parent.attr in _METADATA_ATTRIBUTES or not hasattr(xr.Dataset, parent.attr)


@lru_cache(maxsize=256)
def _inputs_known(expression: str) -> bool:
    steps = [step.strip() for step in expression.split(";") if step.strip()]
    trees = _parsed_steps(expression)
    if len(trees) != len(steps):
        return False
    for tree in trees:
        parents = {id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "ds" and not _listed_dataset_use(node, parents):
                return False
    return True


def compute_inputs_known(expression: Any) -> bool:
    """True when :func:`compute_dependency_names` lists every variable the expression can read.

    Every ``;`` step must parse, and ``ds`` may appear only as ``ds['name']``,
    ``ds.name``, ``ds.get('name')``, ``ds.sum_prefix('prefix', n)``,
    ``'name' in ds`` or a metadata attribute such as ``ds.dims``. A computed
    key (``ds[key]``), a list (``ds[['a']]``), ``ds.data_vars[...]`` or ``ds``
    passed on may read any variable, so no raw variable is provably independent.
    """
    return _inputs_known(str(expression or ""))


def compute_required_inputs(expression: Any) -> tuple[list[str], list[list[str]]]:
    """Statically certain reads, excluding conditional branches and optional get().

    This is a lower bound, not validation of arbitrary expressions. It lets
    preflight reject demonstrably incomplete per-file inputs without requiring
    every alternative in a conditional expression to exist. The second result
    groups mandatory sum_prefix parts: partial sums disallow raw fallback.
    """
    required = []
    sum_groups = []
    pending = list(reversed(_parsed_steps(expression)))
    while pending:
        node = pending.pop()
        if isinstance(node, ast.IfExp):
            children = [node.test]
        elif isinstance(node, ast.BoolOp):
            children = node.values[:1]
        elif isinstance(node, ast.Compare):
            children = [node.left, *node.comparators[:1]]
        else:
            children = list(ast.iter_child_nodes(node))
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "ds"
            ):
                children.remove(node.func)  # A dataset method is not a variable read.
                if node.func.attr == "sum_prefix" or (
                    node.func.attr == "get"
                    and len(node.args) < 2
                    and not any(keyword.arg == "default" for keyword in node.keywords)
                ):
                    required.extend(compute_dependency_names(ast.unparse(node)))
                    if node.func.attr == "sum_prefix":
                        sum_groups.append(compute_dependency_names(ast.unparse(node)))
        if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(node.value, ast.Name):
            if node.value.id == "ds":
                required.extend(compute_dependency_names(ast.unparse(node)))
        pending.extend(reversed(children))
    return list(dict.fromkeys(required)), sum_groups


def compute_dependency_names(expression: Any) -> list[str]:
    """Names read through dataset indexing, attributes, get(), and sum_prefix()."""
    trees = _parsed_steps(expression)
    calls = {id(node.func) for tree in trees for node in ast.walk(tree) if isinstance(node, ast.Call)}
    names = []
    parts = []
    for node in (node for tree in trees for node in ast.walk(tree)):
        key = None
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "ds":
            key = node.slice
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "ds":
            if id(node) not in calls and not hasattr(xr.Dataset, node.attr):
                names.append(node.attr)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ds"
        ):
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            if node.func.attr == "get":
                key = node.args[0] if node.args else kwargs.get("key")
            elif node.func.attr == "sum_prefix":
                prefix = node.args[0] if node.args else kwargs.get("prefix")
                count = node.args[1] if len(node.args) > 1 else kwargs.get("parts")
                if (
                    isinstance(prefix, ast.Constant)
                    and isinstance(prefix.value, str)
                    and isinstance(count, ast.Constant)
                    and type(count.value) is int
                    and count.value > 0
                ):
                    parts.extend(f"{prefix.value}{index}" for index in range(1, count.value + 1))
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            names.append(key.value)
    return list(dict.fromkeys(name for name in [*names, *parts] if name))


_NO_DEFAULT = object()


class _CaseInsensitiveDatasetProxy:
    """Exact-first case-insensitive ``ds['var']`` proxy for compute expressions."""

    def __init__(self, dataset: Any):
        self._dataset = dataset

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            actual = get_xarray_key_case_insensitive(self._dataset, key)
            try:
                return self._dataset[actual if actual is not None else key]
            except KeyError as exc:
                raise MissingComputeVariable(f"Variable {key!r} not found in dataset") from exc
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

    def sum_prefix(self, prefix: str, parts: int) -> Any:
        """Sum the numbered variables ``<prefix>1`` .. ``<prefix><parts>`` (case-insensitive).

        For outputs split into a configured number of parts, such as CoLM's
        ``nsed`` sediment size classes ``f_sedcon_1``, ``f_sedcon_2``, ...
        Every part must be in the dataset and no higher-numbered part may be,
        so a part kept in another file, or a run with another number of parts,
        stops the computation instead of giving a partial sum.
        """
        if isinstance(parts, bool) or not isinstance(parts, int) or parts < 1:
            raise ComputeIntegrityError(
                f"sum_prefix({prefix!r}, parts) needs a positive whole number of parts, got {parts!r}"
            )
        wanted = str(prefix).lower()
        numbered = {}
        for name in self._dataset.data_vars:
            suffix = str(name).lower()[len(wanted) :]
            if str(name).lower().startswith(wanted) and suffix.isdigit():
                if int(suffix) in numbered:
                    raise ComputeIntegrityError(
                        f"{numbered[int(suffix)]} and {name} are both part {int(suffix)} of {prefix!r}; "
                        "keep one name per part"
                    )
                numbered[int(suffix)] = str(name)
        if not numbered:
            raise MissingComputeVariable(f"No numbered variables with prefix {prefix!r} found in dataset")
        missing = [f"{prefix}{index}" for index in range(1, parts + 1) if index not in numbered]
        if missing:
            raise ComputeIntegrityError(
                f"Variable(s) {', '.join(missing)} not found in dataset; "
                "check the configured part count and input files"
            )
        extra = sorted(index for index in numbered if index > parts)
        if extra:
            raise ComputeIntegrityError(
                f"{numbered[extra[0]]} found beyond the {parts} parts summed for {prefix!r}; "
                "set the part count to the run's number of parts"
            )
        total = self._dataset[numbered[1]]
        for index in range(2, parts + 1):
            total = total + self._dataset[numbered[index]]
        return total

    def get(self, key: Any, default: Any = _NO_DEFAULT) -> Any:
        """``ds.get('var'[, default])``: a missing variable without a default is a missing input."""
        if isinstance(key, str) and get_xarray_key_case_insensitive(self._dataset, key) is not None:
            return self[key]
        if default is not _NO_DEFAULT:
            return default
        if isinstance(key, str):
            raise MissingComputeVariable(f"Variable {key!r} not found in dataset")
        return self._dataset.get(key)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return getattr(self._dataset, name)
        except AttributeError:
            # ``ds.var`` reads a data variable like ``ds['var']``: case-insensitive,
            # and a missing one is a missing input, not an arbitrary failure.
            actual = get_xarray_key_case_insensitive(self._dataset, name)
            if actual is not None:
                return self._dataset[actual]
            raise _MissingComputeAttribute(f"Variable {name!r} not found in dataset") from None


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
# Keep this deliberately small: catalog expressions occasionally need
# trigonometry for spherical grid-cell geometry (for example, converting a
# volume flux to an areal runoff depth), while the evaluator must remain free
# of arbitrary NumPy execution.
_SAFE_NUMPY_FUNCTIONS = frozenset({"sin", "sqrt"})
_SAFE_DATA_METHODS = frozenset(
    {"diff", "fillna", "get", "isel", "lower", "mean", "squeeze", "sum", "sum_prefix", "where"}
)


def _root_name(node: ast.AST) -> str | None:
    """Return the root identifier for an attribute/call/subscript chain."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _root_name(node.value)
    if isinstance(node, ast.Subscript):
        return _root_name(node.value)
    if isinstance(node, ast.Call):
        return _root_name(node.func)
    return None


def _validate_call(node: ast.Call, allowed_names: set[str]) -> None:
    """Allow only pure catalog compute calls, not arbitrary module/object APIs."""
    func = node.func
    if isinstance(func, ast.Name):
        raise ComputeError(f"Unsafe expression: function '{func.id}' is not allowed.")
    if not isinstance(func, ast.Attribute):
        raise ComputeError("Unsafe expression: only whitelisted method calls are allowed.")

    root = _root_name(func)
    if root is not None and root not in allowed_names:
        raise ComputeError(f"Unsafe expression: identifier '{root}' is not allowed.")
    if root == "np":
        if func.attr not in _SAFE_NUMPY_FUNCTIONS:
            raise ComputeError(f"Unsafe expression: numpy function '{func.attr}' is not allowed.")
        return
    if root == "xr":
        raise ComputeError(f"Unsafe expression: xarray function '{func.attr}' is not allowed.")
    if func.attr not in _SAFE_DATA_METHODS:
        raise ComputeError(f"Unsafe expression: method '{func.attr}' is not allowed.")


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
        if isinstance(node, ast.Call):
            _validate_call(node, allowed)
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

        _validate_expression(steps[-1], allowed_names=namespace.keys())
        result = eval(steps[-1], {"__builtins__": {}}, namespace)  # noqa: S307

        logger.debug("Computed %s successfully", var_name)
        return result

    except ComputeError as e:
        if isinstance(e, MissingComputeVariable):
            e.dependencies = tuple(name.casefold() for name in compute_dependency_names(expression))
            e.inputs_known = compute_inputs_known(expression)
        e.args = (f"Computing {var_name}: {e}", *e.args[1:])
        raise
    except KeyError as e:
        raise ComputeError(
            f"Variable {e} not found in dataset when computing {var_name}. Available: {list(ds.data_vars)[:10]}..."
        ) from e
    except Exception as e:
        raise ComputeError(f"Failed to compute {var_name}: {e}") from e


class ComputeError(Exception):
    """Raised when a compute expression fails."""


class MissingComputeVariable(ComputeError):
    """A compute expression references a variable absent from the source dataset."""

    dependencies: tuple[str, ...] = ()
    # False when the expression reads variables the parser cannot list
    # (``ds[key]``); any raw variable may then be one of its inputs.
    inputs_known: bool = True

    def may_read(self, name: str) -> bool:
        """Whether the failed expression may read ``name``, so it cannot stand in for the result."""
        return not self.inputs_known or str(name).casefold() in self.dependencies


class _MissingComputeAttribute(MissingComputeVariable, AttributeError):
    """``ds.var`` for an absent variable; still an AttributeError for ``hasattr``."""


class ComputeIntegrityError(ComputeError):
    """A required aggregate is incomplete or inconsistent; raw fallback is unsafe."""

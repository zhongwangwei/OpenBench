import ast
import importlib
import logging
import subprocess
import sys
import warnings
from pathlib import Path

_VIS_DIR = Path(__file__).resolve().parents[1] / "src" / "openbench" / "visualization"


def _visualization_python_files():
    return sorted(_VIS_DIR.rglob("*.py"))


def test_visualization_imports_do_not_install_global_runtime_warning_filters():
    before = list(warnings.filters)

    for module_name in [
        "openbench.visualization.Fig_Basic_Plot",
        "openbench.visualization.Fig_Relative_Score",
        "openbench.visualization.Fig_geo_plot_index",
        "openbench.visualization.Mod_Only_Drawing",
    ]:
        module = importlib.import_module(module_name)
        importlib.reload(module)

    added_filters = [warning_filter for warning_filter in warnings.filters if warning_filter not in before]
    forbidden_categories = (RuntimeWarning, FutureWarning, UserWarning)
    assert not [
        warning_filter
        for warning_filter in added_filters
        if warning_filter[0] == "ignore" and issubclass(warning_filter[2], forbidden_categories)
    ]


def test_core_visualization_imports_do_not_override_dependency_logger_levels():
    before = {name: logging.getLogger(name).level for name in ["xarray", "dask"]}

    for module_name in [
        "openbench.core.scores",
        "openbench.core.evaluation",
        "openbench.core.comparison",
        "openbench.visualization.Mod_Only_Drawing",
    ]:
        module = importlib.import_module(module_name)
        importlib.reload(module)

    assert {name: logging.getLogger(name).level for name in before} == before


def test_visualization_import_does_not_override_caller_selected_backend():
    script = """
import matplotlib
matplotlib.use("svg", force=True)
before = matplotlib.get_backend()
import openbench.visualization
after = matplotlib.get_backend()
assert before.lower() == "svg", before
assert after.lower() == "svg", after
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_visualization_exceptions_do_not_log_errors_without_raising():
    for path in _visualization_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for handler in [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]:
            handler_module = ast.Module(body=handler.body, type_ignores=[])
            raises = any(isinstance(child, ast.Raise) for child in ast.walk(handler_module))
            logs_error = any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr in {"error", "exception"}
                and (
                    (isinstance(child.func.value, ast.Name) and child.func.value.id in {"logging", "logger"})
                    or (isinstance(child.func.value, ast.Name) and "log" in child.func.value.id.lower())
                )
                for child in ast.walk(handler_module)
            )
            assert not (logs_error and not raises), (
                f"{path.name}:{handler.lineno} logs an error in except but does not re-raise"
            )


def test_visualization_renderers_do_not_close_unrelated_figures():
    for path in _visualization_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "close":
                continue
            if not isinstance(node.func.value, ast.Name) or node.func.value.id != "plt":
                continue
            assert not (node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "all"), (
                f"{path.name}:{node.lineno} must close its own figure, not all figures"
            )


def test_diagram_outputs_use_safe_filename_components():
    for filename, raw_prefix in [
        ("Fig_taylor_diagram.py", "Taylor_Diagram_"),
        ("Fig_target_diagram.py", "Target_Diagram_"),
    ]:
        source = (_VIS_DIR / filename).read_text(encoding="utf-8")
        assert "join_filename_components" in source
        assert raw_prefix + "{evaluation_item}" not in source


def test_visualization_combination_filenames_do_not_join_with_plain_underscores():
    for path in [
        _VIS_DIR / "Fig_parallel_coordinates.py",
        _VIS_DIR / "Fig_portrait_plot_seasonal.py",
    ]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "join"
                and isinstance(node.func.value, ast.Constant)
                and node.func.value.value == "_"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "item_combination"
            ):
                raise AssertionError(f"{path.name}:{node.lineno} must use join_filename_components for plot names")


def test_visualization_renderers_save_and_close_explicit_figures():
    for path in _visualization_python_files():
        if path.name == "_figure_io.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "plt":
                assert node.func.attr != "savefig", f"{path.name}:{node.lineno} must save an explicit figure"
                assert not (node.func.attr == "close" and not node.args), (
                    f"{path.name}:{node.lineno} must close an explicit figure"
                )
            assert node.func.attr != "savefig", f"{path.name}:{node.lineno} must use save_figure()"


def test_visualization_renderers_do_not_use_pyplot_current_axes_state():
    forbidden_pyplot_methods = {
        "fill_between",
        "get_cmap",
        "grid",
        "legend",
        "plot",
        "subplots_adjust",
        "tight_layout",
        "title",
        "xlabel",
        "xlim",
        "xticks",
        "ylabel",
        "ylim",
        "yticks",
        "gca",
        "gcf",
        "show",
        "setp",
        "subplot",
        "boxplot",
        "colorbar",
        "scatter",
        "MultipleLocator",
        "Polygon",
    }
    for path in _visualization_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "plt"
                and node.func.attr in forbidden_pyplot_methods
            ):
                raise AssertionError(f"{path.name}:{node.lineno} must use explicit fig/ax, not plt.{node.func.attr}")


def test_functions_that_open_figures_are_failure_isolated():
    allowed_unisolated = {
        ("Fig_target_diagram.py", "_get_target_diagram_arguments"),
        ("Fig_taylor_diagram.py", "_get_taylor_diagram_arguments"),
    }
    for path in _visualization_python_files():
        if not path.name.startswith("Fig_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            opens_figure = any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "plt"
                and child.func.attr in {"figure", "subplots"}
                for child in ast.walk(node)
            )
            if not opens_figure or (path.name, node.name) in allowed_unisolated:
                continue
            decorators = {ast.unparse(decorator) for decorator in node.decorator_list}
            assert "with_isolated_rc" in decorators, f"{path.name}:{node.lineno} must close opened figures on failure"


def test_renderers_that_save_figures_also_close_them():
    for path in _visualization_python_files():
        if path.name == "_figure_io.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            saves_figure = any(
                isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "save_figure"
                for child in ast.walk(node)
            )
            if not saves_figure:
                continue
            closes_figure = any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "close"
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "plt"
                and child.args
                for child in ast.walk(node)
            )
            assert closes_figure, f"{path.name}:{node.lineno} saves a figure but does not close it"


def test_visualization_renderers_do_not_create_unused_pyplot_figures():
    for path in _visualization_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            func = node.value.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "plt"
                and func.attr in {"figure", "subplots"}
            ):
                raise AssertionError(f"{path.name}:{node.lineno} creates an unused matplotlib figure")


def test_save_figure_creates_parent_and_replaces_atomically(tmp_path):
    from openbench.visualization._figure_io import save_figure

    output_path = tmp_path / "missing" / "plot.txt"

    class DummyFigure:
        def savefig(self, path, **kwargs):
            Path(path).write_text(f"{kwargs['format']}:{kwargs['dpi']}")

    save_figure(DummyFigure(), output_path, format="txt", dpi=42)

    assert output_path.read_text(encoding="utf-8") == "txt:42"
    assert not list(output_path.parent.glob(f".{output_path.name}.*"))


def test_save_figure_removes_temporary_file_on_failure(tmp_path):
    from openbench.visualization._figure_io import save_figure

    output_path = tmp_path / "plot.txt"

    class FailingFigure:
        def savefig(self, path, **kwargs):
            Path(path).write_text("partial")
            raise RuntimeError("boom")

    try:
        save_figure(FailingFigure(), output_path, format="txt")
    except RuntimeError:
        pass
    else:
        raise AssertionError("save_figure should propagate save failures")

    assert not output_path.exists()
    assert not list(tmp_path.glob(f".{output_path.name}.*"))


def test_station_index_plot_saves_each_ref_and_sim_iteration():
    tree = ast.parse((_VIS_DIR / "Fig_stn_plot_index.py").read_text(encoding="utf-8"))
    function = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "make_stn_plot_index"
    )
    loop = next(node for node in function.body if isinstance(node, ast.For))

    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "save_figure"
        for node in ast.walk(loop)
    ), "station index plot must save each ref/sim figure inside the loop"
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "close" and node.args
        for node in ast.walk(loop)
    ), "station index plot must close each ref/sim figure inside the loop"

import ast
from pathlib import Path

_VIS_DIR = Path(__file__).resolve().parents[1] / "src" / "openbench" / "visualization"


_OPTION_ISOLATED_FUNCTIONS = [
    ("Fig_ANOVA.py", "map"),
    ("Fig_ANOVA.py", "make_ANOVA"),
    ("Fig_Basic_Plot.py", "plot_map_grid"),
    ("Fig_Basic_Plot.py", "plot_stn_map"),
    ("Fig_Basic_Plot.py", "make_Basic"),
    ("Fig_Correlation.py", "make_Correlation"),
    ("Fig_Diff_Plot.py", "plot_grid_map"),
    ("Fig_Diff_Plot.py", "plot_stn_map"),
    ("Fig_Functional_Response.py", "make_Functional_Response"),
    ("Fig_Hellinger_Distance.py", "make_Hellinger_Distance"),
    ("Fig_LC_based_heat_map.py", "make_LC_based_heat_map"),
    ("Fig_LC_based_heat_map.py", "make_CZ_based_heat_map"),
    ("Fig_Mann_Kendall_Trend_Test.py", "map"),
    ("Fig_Mann_Kendall_Trend_Test.py", "make_Mann_Kendall_Trend_Test"),
    ("Fig_Partial_Least_Squares_Regression.py", "map"),
    ("Fig_Partial_Least_Squares_Regression.py", "make_Partial_Least_Squares_Regression"),
    ("Fig_Relative_Score.py", "make_stn_plot_index"),
    ("Fig_Relative_Score.py", "make_geo_plot_index"),
    ("Fig_Standard_Deviation.py", "make_Standard_Deviation"),
    ("Fig_Three_Cornered_Hat.py", "map"),
    ("Fig_Three_Cornered_Hat.py", "make_Three_Cornered_Hat"),
    ("Fig_Whisker_Plot.py", "make_scenarios_comparison_Whisker_Plot"),
    ("Fig_heatmap.py", "make_scenarios_scores_comparison_heat_map"),
    ("Fig_parallel_coordinates.py", "make_scenarios_comparison_parallel_coordinates"),
    ("Fig_portrait_plot_seasonal.py", "make_scenarios_comparison_Portrait_Plot_seasonal"),
    ("Fig_stn_plot_index.py", "make_stn_plot_index"),
    ("Fig_target_diagram.py", "make_scenarios_comparison_Target_Diagram"),
    ("Fig_taylor_diagram.py", "make_scenarios_comparison_Taylor_Diagram"),
]


def _function_node(filename: str, function_name: str) -> ast.FunctionDef:
    tree = ast.parse((_VIS_DIR / filename).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"{filename}:{function_name} not found")


def _copies_option(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "option"
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == "copy"
        and isinstance(statement.value.func.value, ast.Name)
        and statement.value.func.value.id == "option"
    )


def test_visualization_renderers_copy_option_before_mutating_it():
    for filename, function_name in _OPTION_ISOLATED_FUNCTIONS:
        node = _function_node(filename, function_name)
        assert _copies_option(node.body[0]), f"{filename}:{function_name} must not mutate caller option"

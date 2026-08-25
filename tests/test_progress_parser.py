from openbench.gui.pages.page_run_monitor import count_evaluation_tasks
from openbench.gui.progress_parser import parse_progress_line


def _state(**overrides):
    state = {
        "current_variable": "",
        "current_ref": "",
        "current_sim": "",
        "started_preprocess_tasks": set(),
        "completed_preprocess_tasks": set(),
        "completed_eval_tasks": set(),
        "completed_groupby_tasks": set(),
        "completed_comparison_tasks": set(),
        "total_tasks": 0,
        "num_comparisons": 0,
        "num_variables": 0,
    }
    state.update(overrides)
    return state


CONSTANTS = {
    "PROGRESS_INIT": 5,
    "PROGRESS_WORK": 90,
    "PROGRESS_MAX": 95,
    "PROGRESS_INCREMENT": 0.5,
}


def test_progress_parser_does_not_move_backwards_when_total_tasks_known():
    state = _state(total_tasks=10)

    progress, var, stage = parse_progress_line("Processing Latent_Heat", 50, state, CONSTANTS)

    assert progress == 50
    assert var == "Latent_Heat"
    assert stage == ""


def test_progress_parser_ignores_exception_names_as_reference_or_simulation_sources():
    state = _state(current_ref="GLEAM", current_sim="CoLM")

    progress, var, stage = parse_progress_line(
        "ReferenceError: variable is not defined; simulation traceback follows", 12, state, CONSTANTS
    )

    assert progress == 12
    assert var == ""
    assert stage == ""
    assert state["current_ref"] == "GLEAM"
    assert state["current_sim"] == "CoLM"


def test_progress_parser_accepts_structured_ref_and_sim_markers():
    state = _state()

    parse_progress_line("Processing Latent_Heat - ref: GLEAM_v4.2a - sim: CoLM2024", 0, state, CONSTANTS)

    assert state["current_variable"] == "Latent_Heat"
    assert state["current_ref"] == "GLEAM_v4.2a"
    assert state["current_sim"] == "CoLM2024"


def test_progress_parser_counts_structured_completed_evaluation_line():
    state = _state(total_tasks=1)

    progress, var, stage = parse_progress_line(
        "Completed Latent_Heat: sim=CoLM2024 ref=GLEAM_v4.2a", 5, state, CONSTANTS
    )

    assert progress == 95
    assert var == "Latent_Heat"
    assert stage == "Evaluation"
    assert ("Latent_Heat", "GLEAM_v4.2a", "CoLM2024") in state["completed_eval_tasks"]


def test_progress_parser_advances_during_structured_preprocessing():
    state = _state(total_tasks=2)

    progress, variable, stage = parse_progress_line(
        'OPENBENCH_PROGRESS {"event":"preprocessing_completed","variable":"Runoff","sim":"SimA","ref":"RefA"}',
        5,
        state,
        CONSTANTS,
    )

    assert progress > 5
    assert variable == "Runoff"
    assert stage == "Preprocessing"
    assert ("Runoff", "RefA", "SimA") in state["completed_preprocess_tasks"]


def test_progress_parser_leaves_five_percent_when_preprocessing_starts():
    state = _state(total_tasks=2)

    progress, variable, stage = parse_progress_line(
        'OPENBENCH_PROGRESS {"event":"preprocessing_started","variable":"Runoff","sim":"SimA","ref":"RefA"}',
        5,
        state,
        CONSTANTS,
    )

    assert progress > 5
    assert variable == "Runoff"
    assert stage == "Preprocessing"


def test_progress_parser_advances_report_stage_without_backtracking():
    state = _state(total_tasks=1)

    progress, _var, stage = parse_progress_line("Starting report generation...", 90, state, CONSTANTS)
    done, _var, done_stage = parse_progress_line("Report generation completed successfully", progress, state, CONSTANTS)

    assert stage == "Report"
    assert done_stage == "Report"
    assert done > progress > 90


def test_progress_parser_counts_actual_comparison_completion_line():
    state = _state(total_tasks=1)

    progress, _var, stage = parse_progress_line("Completed Taylor_Diagram comparison", 5, state, CONSTANTS)

    assert progress == 95
    assert stage == "Comparison"
    assert "taylor_diagram" in state["completed_comparison_tasks"]


def test_progress_parser_counts_actual_groupby_complete_lines():
    for line, key in [
        ("IGBP groupby complete", "igbp"),
        ("PFT groupby complete", "pft"),
        ("Climate zone groupby complete", "climate"),
    ]:
        state = _state(total_tasks=1, current_variable="Runoff")

        progress, _var, stage = parse_progress_line(line, 5, state, CONSTANTS)

        assert progress == 95
        assert stage == "Comparison"
        assert ("Runoff", key) in state["completed_groupby_tasks"]


def test_evaluation_task_count_uses_each_variables_bound_sources():
    config = {
        "ref_data": {
            "general": {
                "Runoff_ref_source": "RefA",
                "GPP_ref_source": "RefB",
            }
        },
        "sim_data": {
            "general": {
                "Runoff_sim_source": ["SimA"],
                "GPP_sim_source": ["SimB"],
            }
        },
    }

    assert count_evaluation_tasks(config, ["Runoff", "GPP"]) == 2


def test_progress_parser_preserves_spaces_in_structured_source_names():
    state = _state(total_tasks=1)
    progress, variable, stage = parse_progress_line(
        'OPENBENCH_PROGRESS {"event":"evaluation_completed","variable":"Runoff Basin","sim":"ERA5 Land",'
        '"ref":"GLDAS Comparison 2"}',
        5,
        state,
        CONSTANTS,
    )

    assert progress == 95
    assert variable == "Runoff Basin"
    assert stage == "Evaluation"
    assert ("Runoff Basin", "GLDAS Comparison 2", "ERA5 Land") in state["completed_eval_tasks"]

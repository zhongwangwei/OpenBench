from openbench.gui.progress_parser import parse_progress_line


def _state(**overrides):
    state = {
        "current_variable": "",
        "current_ref": "",
        "current_sim": "",
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

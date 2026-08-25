"""Machine-readable task progress emitted only for GUI-launched runs."""

import json
import os

GUI_PROGRESS_PREFIX = "OPENBENCH_PROGRESS "


def _emit_gui_event(event: str, *, variable: str, sim: str, ref: str) -> None:
    if os.environ.get("OPENBENCH_GUI_PROGRESS") != "1":
        return
    payload = json.dumps(
        {
            "event": event,
            "variable": variable,
            "sim": sim,
            "ref": ref,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    print(f"{GUI_PROGRESS_PREFIX}{payload}", flush=True)


def emit_gui_task_completion(result: dict) -> None:
    if result.get("status") not in {"success", "skipped"}:
        return
    _emit_gui_event(
        "evaluation_completed",
        variable=result["variable"],
        sim=result["sim"],
        ref=result["ref"],
    )


def emit_gui_preprocessing_completion(task: dict) -> None:
    _emit_gui_event(
        "preprocessing_completed",
        variable=task["var_name"],
        sim=task["sim_source"],
        ref=task["ref_source"],
    )


def emit_gui_preprocessing_started(task: dict) -> None:
    _emit_gui_event(
        "preprocessing_started",
        variable=task["var_name"],
        sim=task["sim_source"],
        ref=task["ref_source"],
    )

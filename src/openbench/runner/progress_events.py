"""Machine-readable task progress emitted only for GUI-launched runs."""

import json
import os

GUI_PROGRESS_PREFIX = "OPENBENCH_PROGRESS "


def emit_gui_task_completion(result: dict) -> None:
    if os.environ.get("OPENBENCH_GUI_PROGRESS") != "1" or result.get("status") not in {"success", "skipped"}:
        return
    payload = json.dumps(
        {
            "event": "evaluation_completed",
            "variable": result["variable"],
            "sim": result["sim"],
            "ref": result["ref"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    print(f"{GUI_PROGRESS_PREFIX}{payload}", flush=True)

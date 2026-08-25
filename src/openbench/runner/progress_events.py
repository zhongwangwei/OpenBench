"""Machine-readable task progress emitted only for GUI-launched runs."""

import os


def emit_gui_task_completion(result: dict) -> None:
    if os.environ.get("OPENBENCH_GUI_PROGRESS") != "1":
        return
    print(
        f"Completed {result['variable']}: sim={result['sim']} ref={result['ref']}",
        flush=True,
    )

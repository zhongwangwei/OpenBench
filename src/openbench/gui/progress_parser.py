# -*- coding: utf-8 -*-
"""Shared log-line progress parser for local and remote runners."""

import re


def parse_progress_line(
    line: str,
    current_progress: float,
    state: dict,
    constants: dict,
) -> tuple:
    """Parse progress from a log line with detailed task tracking.

    Args:
        line: Log line to parse.
        current_progress: Current progress value.
        state: Mutable dict with keys: current_variable, current_ref, current_sim,
               completed_eval_tasks (set), completed_groupby_tasks (set),
               completed_comparison_tasks (set), total_tasks, num_comparisons,
               num_variables.
        constants: Dict with keys: PROGRESS_INIT, PROGRESS_WORK, PROGRESS_MAX,
                   PROGRESS_INCREMENT.

    Returns:
        Tuple of (progress, variable, stage).
    """
    start_progress = current_progress
    var = state.get("current_variable", "")
    stage = ""

    line_lower = line.lower()

    # Detect variable being processed
    if "processing" in line_lower or "evaluating" in line_lower:
        for keyword in ["Processing", "Evaluating", "processing", "evaluating"]:
            if keyword in line:
                parts = line.split(keyword)
                if len(parts) > 1:
                    remaining = parts[1].strip()
                    if remaining:
                        var_name = remaining.split()[0].strip(".:,")
                        if var_name and len(var_name) > 2:
                            state["current_variable"] = var_name
                            var = var_name
                    break

    # Detect reference/simulation source being processed. Keep this
    # deliberately structured: broad substring checks such as
    # "reference" + split(":") misclassified exception names like
    # "ReferenceError: ..." as a new reference source.
    ref_match = re.search(
        r"(?:^|[-\s])ref:\s*(\S+)|\bref_source\b\s*[:=]\s*(\S+)|\breference(?:\s+source)?\b\s*[:=]\s*(\S+)",
        line,
        re.IGNORECASE,
    )
    if ref_match:
        state["current_ref"] = next(group for group in ref_match.groups() if group).strip(":,")

    sim_match = re.search(
        r"(?:^|[-\s])sim:\s*(\S+)|\bsim_source\b\s*[:=]\s*(\S+)|\bsimulation(?:\s+source)?\b\s*[:=]\s*(\S+)",
        line,
        re.IGNORECASE,
    )
    if sim_match:
        state["current_sim"] = next(group for group in sim_match.groups() if group).strip(":,")

    # Detect stage
    if "evaluation" in line_lower and "item" not in line_lower:
        stage = "Evaluation"
    elif "comparison" in line_lower or "groupby" in line_lower:
        stage = "Comparison"
        if "done running" in line_lower and "comparison" in line_lower:
            match = re.search(r"done running\s+(\w+)\s+comparison", line_lower)
            if match:
                comp_name = match.group(1)
                state["completed_comparison_tasks"].add(comp_name)
    elif "statistic" in line_lower:
        stage = "Statistics"

    # Detect task completions
    task_completed = False

    if stage == "Evaluation" and ("completed" in line_lower or "finished" in line_lower or "done" in line_lower):
        task_key = (state.get("current_variable", ""), state.get("current_ref", ""), state.get("current_sim", ""))
        if task_key not in state["completed_eval_tasks"] and state.get("current_variable"):
            state["completed_eval_tasks"].add(task_key)
            task_completed = True

    for groupby_type in ["igbp", "pft", "climate", "landcover"]:
        if groupby_type in line_lower and (
            "completed" in line_lower or "finished" in line_lower or "done" in line_lower
        ):
            task_key = (state.get("current_variable", ""), groupby_type)
            if task_key not in state["completed_groupby_tasks"]:
                state["completed_groupby_tasks"].add(task_key)
                task_completed = True

    if stage == "Statistics" and ("completed" in line_lower or "finished" in line_lower):
        comp_name = state.get("current_variable") or "comparison"
        if comp_name not in state["completed_comparison_tasks"]:
            state["completed_comparison_tasks"].add(comp_name)
            task_completed = True

    # Calculate progress
    total_tasks = state.get("total_tasks", 0)
    num_comparisons = state.get("num_comparisons", 0)
    num_variables = state.get("num_variables", 0)

    P_INIT = constants["PROGRESS_INIT"]
    P_WORK = constants["PROGRESS_WORK"]
    P_MAX = constants["PROGRESS_MAX"]
    P_INC = constants["PROGRESS_INCREMENT"]

    if total_tasks > 0:
        total_completed = (
            len(state["completed_eval_tasks"])
            + len(state["completed_groupby_tasks"])
            + len(state["completed_comparison_tasks"])
        )
        task_progress = (total_completed / max(1, total_tasks)) * P_WORK
        current_progress = min(P_INIT + task_progress, P_MAX)
    elif num_comparisons > 0 and len(state["completed_comparison_tasks"]) > 0:
        comparison_progress = (len(state["completed_comparison_tasks"]) / max(1, num_comparisons)) * P_WORK
        current_progress = min(P_INIT + comparison_progress, P_MAX)
    elif num_variables > 0:
        completed_vars = len(set(t[0] for t in state["completed_eval_tasks"] if t[0]))
        variable_progress = (completed_vars / max(1, num_variables)) * P_WORK
        current_progress = min(P_INIT + variable_progress, P_MAX)
    else:
        if stage == "Comparison":
            current_progress = min(current_progress + P_INC, P_MAX)
        elif task_completed or stage or "complete" in line_lower or "done" in line_lower:
            current_progress = min(current_progress + P_INC * 2, P_MAX)

    # Progress emitted while a run is active should never move backwards:
    # task-count information can arrive after optimistic increments, and
    # malformed/noisy log lines should not reset the bar to PROGRESS_INIT.
    current_progress = max(start_progress, current_progress)
    return current_progress, var, stage

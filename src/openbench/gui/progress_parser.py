# -*- coding: utf-8 -*-
"""Shared log-line progress parser for local and remote runners."""

import json
import re

from openbench.runner.progress_events import GUI_PROGRESS_PREFIX


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

    state.setdefault("started_preprocess_tasks", set())
    state.setdefault("completed_preprocess_tasks", set())
    state.setdefault("completed_eval_tasks", set())
    state.setdefault("completed_groupby_tasks", set())
    state.setdefault("completed_comparison_tasks", set())
    state.setdefault("completed_statistics_tasks", set())

    line_lower = line.lower()

    protocol_line = GUI_PROGRESS_PREFIX in line
    if protocol_line:
        try:
            event = json.loads(line.split(GUI_PROGRESS_PREFIX, 1)[1])
        except (json.JSONDecodeError, TypeError):
            event = {}
        if event.get("event") in {"preprocessing_started", "preprocessing_completed", "evaluation_completed"}:
            state["current_variable"] = str(event.get("variable", ""))
            state["current_sim"] = str(event.get("sim", ""))
            state["current_ref"] = str(event.get("ref", ""))
            var = state["current_variable"]
            task_key = (state["current_variable"], state["current_ref"], state["current_sim"])
            if event.get("event") in {"preprocessing_started", "preprocessing_completed"}:
                if event.get("event") == "preprocessing_started":
                    state.setdefault("started_preprocess_tasks", set()).add(task_key)
                else:
                    state.setdefault("completed_preprocess_tasks", set()).add(task_key)
                stage = "Preprocessing"
            else:
                stage = "Evaluation"

    natural_line = "" if protocol_line else line
    natural_line_lower = natural_line.lower()
    evaluation_failure_summary = (
        "evaluation completed with errors" in natural_line_lower
        or "evaluation failed" in natural_line_lower
        or "failed evaluation" in natural_line_lower
    )

    # Detect variable being processed
    if "processing" in natural_line_lower or "evaluating" in natural_line_lower:
        for keyword in ["Processing", "Evaluating", "processing", "evaluating"]:
            if keyword in natural_line:
                parts = natural_line.split(keyword)
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
        natural_line,
        re.IGNORECASE,
    )
    if ref_match:
        state["current_ref"] = next(group for group in ref_match.groups() if group).strip(":,")

    sim_match = re.search(
        r"(?:^|[-\s])sim:\s*(\S+)|\bsim_source\b\s*[:=]\s*(\S+)|\bsimulation(?:\s+source)?\b\s*[:=]\s*(\S+)",
        natural_line,
        re.IGNORECASE,
    )
    if sim_match:
        state["current_sim"] = next(group for group in sim_match.groups() if group).strip(":,")

    completed_eval_match = re.search(
        r"\bcompleted\s+([^:]+):.*?\bsim\s*[=:]\s*([^\s,;]+).*?\bref\s*[=:]\s*([^\s,;]+)",
        natural_line,
        re.IGNORECASE,
    )
    if completed_eval_match and not stage:
        state["current_variable"] = completed_eval_match.group(1).strip()
        state["current_sim"] = completed_eval_match.group(2).strip(":,")
        state["current_ref"] = completed_eval_match.group(3).strip(":,")
        var = state["current_variable"]
        stage = "Evaluation"

    statistics_done = re.search(r"\bcompleted\s+([^:]+?)\s+analysis\b", natural_line, re.IGNORECASE)
    statistics_running = re.search(r"\brunning\s+([^:]+?)\s+analysis\b", natural_line, re.IGNORECASE)
    if statistics_done or statistics_running:
        state["current_statistic"] = (statistics_done or statistics_running).group(1).strip()
        stage = "Statistics"

    # Detect stage. Report filenames such as evaluation_report.html/pdf are
    # report artifacts, not a new Evaluation phase.
    report_artifact = re.search(r"\b[\w.-]*report\.(?:html|pdf)\b", natural_line_lower)
    if not stage and ("report" in natural_line_lower or report_artifact):
        stage = "Report"
    elif not stage and "evaluation" in natural_line_lower and "item" not in natural_line_lower:
        stage = "Evaluation"
    elif not stage and ("comparison" in natural_line_lower or "groupby" in natural_line_lower):
        stage = "Comparison"
        comparison_done = re.search(r"(?:done running|completed)\s+([\w-]+)\s+comparison", natural_line_lower)
        if comparison_done:
            state["completed_comparison_tasks"].add(comparison_done.group(1))
    elif not stage and "statistic" in natural_line_lower:
        stage = "Statistics"

    # Detect task completions
    task_completed = False

    if (
        stage == "Evaluation"
        and not evaluation_failure_summary
        and ("completed" in line_lower or "finished" in line_lower or "done" in line_lower)
    ):
        task_key = (state.get("current_variable", ""), state.get("current_ref", ""), state.get("current_sim", ""))
        if task_key not in state["completed_eval_tasks"] and state.get("current_variable"):
            state["completed_eval_tasks"].add(task_key)
            task_completed = True

    for groupby_type in ["igbp", "pft", "climate", "landcover"]:
        if groupby_type in natural_line_lower and (
            "complete" in natural_line_lower or "finished" in natural_line_lower or "done" in natural_line_lower
        ):
            task_key = groupby_type
            if task_key not in state["completed_groupby_tasks"]:
                state["completed_groupby_tasks"].add(task_key)
                task_completed = True

    if stage == "Statistics" and ("completed" in line_lower or "finished" in line_lower):
        stat_name = state.get("current_statistic") or "statistics"
        if stat_name not in state["completed_statistics_tasks"]:
            state["completed_statistics_tasks"].add(stat_name)
            task_completed = True

    # Calculate progress
    total_tasks = state.get("total_tasks", 0)
    num_comparisons = state.get("num_comparisons", 0)
    num_statistics = state.get("num_statistics", 0)
    num_variables = state.get("num_variables", 0)

    P_INIT = constants["PROGRESS_INIT"]
    P_WORK = constants["PROGRESS_WORK"]
    P_MAX = constants["PROGRESS_MAX"]
    P_INC = constants["PROGRESS_INCREMENT"]

    if total_tasks > 0:
        completed_eval_tasks = state["completed_eval_tasks"]
        preprocess_only = state.get("completed_preprocess_tasks", set()) - completed_eval_tasks
        preprocess_started_only = (
            state.get("started_preprocess_tasks", set())
            - state.get("completed_preprocess_tasks", set())
            - completed_eval_tasks
        )
        total_completed = (
            len(completed_eval_tasks)
            + len(state["completed_groupby_tasks"])
            + len(state["completed_comparison_tasks"])
            + len(state["completed_statistics_tasks"])
            + 0.4 * len(preprocess_only)
            + 0.05 * len(preprocess_started_only)
        )
        task_progress = (total_completed / max(1, total_tasks)) * P_WORK
        current_progress = min(P_INIT + task_progress, P_MAX)
    elif (num_comparisons + num_statistics) > 0 and (
        len(state["completed_comparison_tasks"]) + len(state["completed_statistics_tasks"])
    ) > 0:
        post_completed = len(state["completed_comparison_tasks"]) + len(state["completed_statistics_tasks"])
        post_total = num_comparisons + num_statistics
        comparison_progress = (post_completed / max(1, post_total)) * P_WORK
        current_progress = min(P_INIT + comparison_progress, P_MAX)
    elif num_variables > 0:
        completed_vars = len(set(t[0] for t in state["completed_eval_tasks"] if t[0]))
        variable_progress = (completed_vars / max(1, num_variables)) * P_WORK
        current_progress = min(P_INIT + variable_progress, P_MAX)
    else:
        if stage == "Comparison":
            current_progress = min(current_progress + P_INC, P_MAX)
        elif stage == "Report":
            report_inc = P_INC * (4 if "completed" in line_lower or "success" in line_lower else 2)
            current_progress = min(current_progress + report_inc, P_MAX)
        elif not evaluation_failure_summary and (task_completed or stage or "complete" in line_lower or "done" in line_lower):
            current_progress = min(current_progress + P_INC * 2, P_MAX)

    if stage == "Report":
        report_inc = P_INC * (4 if "completed" in line_lower or "success" in line_lower else 2)
        current_progress = max(current_progress, min(start_progress + report_inc, P_MAX))

    # Progress emitted while a run is active should never move backwards:
    # task-count information can arrive after optimistic increments, and
    # malformed/noisy log lines should not reset the bar to PROGRESS_INIT.
    current_progress = max(start_progress, current_progress)
    return current_progress, var, stage

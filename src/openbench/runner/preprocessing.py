"""Preprocessing orchestration for local runner tasks."""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any

from openbench.util.netcdf import write_file_atomic

logger = logging.getLogger(__name__)


def preprocess_variable(
    var_name: str,
    vtasks: list[dict],
    *,
    unified_mask: bool,
    time_alignment: str,
    build_bridge_runtime_info_fn,
    make_phase_error_fn,
    clone_or_link_ref_for_pair_fn,
    apply_unified_mask_fn,
) -> list[dict[str, Any]]:
    """Preprocess all tasks for one variable (serial within variable).

    Multi-reference + mixed data-type support: ref preprocessing dedupes
    on (ref_source, output_signature). The signature distinguishes pure
    grid×grid (which produces a shareable flat NC) from stn-involved
    runs (which write per-pair ``stn_<ref>_<sim>/`` and DELETE the flat
    as part of station extraction). Without this distinction, a sequence
    like grid×grid → stn×ref-grid → grid×grid would skip the third
    task's prep but find the flat NC already gone.

    For intersection/strict: unified_mask accumulates across sims for
    that ref_source (tracked separately).
    For per_pair: each sim gets its own ref copy (no cross-contamination).
    """
    from openbench.data.processing import DatasetProcessing

    # Dedupe by (ref_source, signature):
    #   - "_grid": pure grid×grid prep produces a flat NC reusable across
    #     multiple grid sims with the same ref
    #   - sim_source string: stn-involved prep writes a per-pair output
    #     dir (stn_<ref>_<sim>) and deletes the flat NC; cannot be shared
    preproc_done: set[tuple[str, str]] = set()
    # Track first-time-seen per ref_source for unified_mask accumulation
    # For stn×stn symlink optimization: first stn dir per ref_source
    ref_stn_data_dirs: dict[str, str] = {}
    # ref_source -> flat ref NC path (only valid while a grid-only path lives there)
    ref_flat_paths: dict[str, str] = {}
    sim_flat_paths: dict[str, str] = {}
    # ref_source -> temporary backup of the current flat ref NC. This is
    # refreshed whenever shared unified_mask mutates the flat file, so a
    # later station extraction can delete the flat without losing masks.
    ref_flat_backups: dict[str, str] = {}
    sim_flat_backups: dict[str, str] = {}
    temp_backup_paths: set[str] = set()
    # Track refs whose flat NC may need restoration because a stn-involved
    # prep deleted it but the variable also has grid×grid tasks pending.
    # extract_station_data_if_needed (processing.py) deletes the shared flat
    # at data/<var>_ref_<ref>_<varname>.nc as part of its cleanup; subsequent
    # grid evaluation reads exactly that path and would fail mid-run.
    refs_with_stn_prep: set[str] = set()
    refs_with_grid_tasks: set[str] = set()
    sims_with_stn_prep: set[str] = set()
    sims_with_grid_tasks: set[str] = set()
    # Remember a representative grid task per ref for end-of-loop restoration
    first_grid_task_per_ref: dict[str, dict[str, Any]] = {}
    first_grid_task_per_sim: dict[str, dict[str, Any]] = {}
    phase_errors: list[dict[str, Any]] = []

    def _backup_flat_ref(ref_source: str) -> None:
        flat_path = ref_flat_paths.get(ref_source)
        if not flat_path or not os.path.isfile(flat_path):
            return
        backup_path = f"{flat_path}.unifiedmask.bak"
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        write_file_atomic(backup_path, lambda tmp_path: shutil.copy2(flat_path, tmp_path), suffix=".tmp.bak")
        ref_flat_backups[ref_source] = backup_path
        temp_backup_paths.add(backup_path)

    def _restore_flat_ref_if_missing(ref_source: str) -> bool:
        flat_path = ref_flat_paths.get(ref_source)
        if not flat_path:
            return False
        if os.path.exists(flat_path):
            return True
        backup_path = ref_flat_backups.get(ref_source)
        if not backup_path or not os.path.isfile(backup_path):
            return False
        os.makedirs(os.path.dirname(flat_path), exist_ok=True)
        write_file_atomic(flat_path, lambda tmp_path: shutil.copy2(backup_path, tmp_path), suffix=".tmp.nc")
        return True

    def _backup_flat_sim(sim_source: str) -> None:
        flat_path = sim_flat_paths.get(sim_source)
        if not flat_path or not os.path.isfile(flat_path):
            return
        backup_path = f"{flat_path}.unifiedmask.bak"
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        write_file_atomic(backup_path, lambda tmp_path: shutil.copy2(flat_path, tmp_path), suffix=".tmp.bak")
        sim_flat_backups[sim_source] = backup_path
        temp_backup_paths.add(backup_path)

    def _restore_flat_sim_if_missing(sim_source: str) -> bool:
        flat_path = sim_flat_paths.get(sim_source)
        if not flat_path:
            return False
        if os.path.exists(flat_path):
            return True
        backup_path = sim_flat_backups.get(sim_source)
        if not backup_path or not os.path.isfile(backup_path):
            return False
        os.makedirs(os.path.dirname(flat_path), exist_ok=True)
        write_file_atomic(flat_path, lambda tmp_path: shutil.copy2(backup_path, tmp_path), suffix=".tmp.nc")
        return True

    def _cleanup_flat_backups() -> None:
        for backup_path in temp_backup_paths:
            try:
                if os.path.isfile(backup_path):
                    os.remove(backup_path)
            except OSError:
                logger.debug("Could not remove temporary flat-ref backup: %s", backup_path)

    try:
        for task in vtasks:
            if task.get("cache_skipped"):
                continue
            ref_source = task["ref_source"]
            sim_source = task["sim_source"]
            try:
                info = build_bridge_runtime_info_fn(task)
                ref_dtype = info.get("ref_data_type", "grid")
                sim_dtype = info.get("sim_data_type", "grid")
                task["ref_data_type"] = ref_dtype
                task["sim_data_type"] = sim_dtype
                is_stn_path = (ref_dtype == "stn") or (sim_dtype == "stn")
                prep_key = (ref_source, sim_source if is_stn_path else "_grid")

                # Track whether this ref ever sees stn-involved or grid-only paths
                # for end-of-loop flat-NC restoration when both kinds coexist.
                if is_stn_path:
                    refs_with_stn_prep.add(ref_source)
                    if sim_dtype != "stn":
                        sims_with_stn_prep.add(sim_source)
                else:
                    refs_with_grid_tasks.add(ref_source)
                    first_grid_task_per_ref.setdefault(ref_source, task)
                    if sim_dtype != "stn":
                        sims_with_grid_tasks.add(sim_source)
                        first_grid_task_per_sim.setdefault(sim_source, task)

                processor = DatasetProcessing(info)

                # Ref: dedupe by (ref_source, signature). Mixed-type configs
                # (same ref reused across grid and stn sims) correctly run
                # separate prep for the stn-side without skipping.
                if prep_key not in preproc_done:
                    logger.info(
                        "Preprocessing ref: %s (%s) [%s]",
                        var_name,
                        ref_source,
                        "stn-pair" if is_stn_path else "grid",
                    )
                    if is_stn_path:
                        _backup_flat_ref(ref_source)
                    processor.prepare_source("ref")
                    preproc_done.add(prep_key)
                    if is_stn_path:
                        # Remember first stn dir per ref for stn×stn symlink branch below
                        ref_stn_data_dirs.setdefault(
                            ref_source,
                            os.path.join(
                                info["casedir"],
                                "data",
                                f"stn_{ref_source}_{sim_source}",
                            ),
                        )
                    else:
                        # Grid prep produced a flat NC; remember its path
                        ref_varname = info.get("ref_varname", "")
                        ref_flat_paths[ref_source] = os.path.join(
                            info["casedir"],
                            "data",
                            f"{var_name}_ref_{ref_source}_{ref_varname}.nc",
                        )
                        _backup_flat_ref(ref_source)
                elif is_stn_path and ref_source in ref_stn_data_dirs:
                    # stn×stn (same ref, second sim): symlink ref files from
                    # the first stn dir to this sim's dir so we don't re-extract
                    # ref stations. Same-ref-same-stn-different-sim case.
                    this_data_dir = os.path.join(
                        info["casedir"],
                        "data",
                        f"stn_{ref_source}_{sim_source}",
                    )
                    src_data_dir = ref_stn_data_dirs[ref_source]
                    if src_data_dir and os.path.isdir(src_data_dir) and this_data_dir != src_data_dir:
                        os.makedirs(this_data_dir, exist_ok=True)
                        for ref_file in os.listdir(src_data_dir):
                            if "_ref_" in ref_file and ref_file.endswith(".nc"):
                                src = os.path.abspath(os.path.join(src_data_dir, ref_file))
                                dst = os.path.join(this_data_dir, ref_file)
                                if not os.path.exists(dst):
                                    os.symlink(src, dst)
                elif not is_stn_path:
                    _restore_flat_ref_if_missing(ref_source)

                # Sim: each task
                logger.info("Preprocessing sim: %s (%s)", var_name, sim_source)
                if sim_dtype != "stn":
                    sim_varname = info.get("sim_varname", "")
                    sim_flat_paths[sim_source] = os.path.join(
                        info["casedir"],
                        "data",
                        f"{var_name}_sim_{sim_source}_{sim_varname}.nc",
                    )
                    if is_stn_path:
                        _backup_flat_sim(sim_source)
                processor.prepare_source("sim")
                if sim_dtype != "stn" and not is_stn_path:
                    _backup_flat_sim(sim_source)

                # Unified mask: ensure evaluation only covers cells where both ref and sim are valid.
                if unified_mask:
                    if ref_dtype != "stn" and sim_dtype != "stn":
                        ref_file_path_for_pair = ref_flat_paths.get(ref_source)
                        if time_alignment == "per_pair":
                            if not ref_file_path_for_pair:
                                raise RuntimeError(
                                    "per_pair time alignment requires a preprocessed flat reference file "
                                    f"for variable={var_name}, ref={ref_source}, sim={sim_source}; "
                                    "refusing to fall back to shared intersection masking"
                                )
                            # per_pair: each sim gets its own ref copy — no cross-contamination
                            if not _restore_flat_ref_if_missing(ref_source):
                                raise RuntimeError(
                                    "per_pair time alignment could not restore the flat reference file "
                                    f"for variable={var_name}, ref={ref_source}, sim={sim_source}"
                                )
                            ref_varname_m = info.get("ref_varname", "")
                            pair_ref = os.path.join(
                                info["casedir"],
                                "data",
                                f"{var_name}_ref_{ref_source}_{sim_source}_{ref_varname_m}.nc",
                            )
                            strategy = clone_or_link_ref_for_pair_fn(ref_file_path_for_pair, pair_ref)
                            logger.debug(
                                "Prepared per-pair ref via %s: %s -> %s",
                                strategy,
                                ref_file_path_for_pair,
                                pair_ref,
                            )
                            apply_unified_mask_fn(info, var_name, ref_source, sim_source, ref_override=pair_ref)
                            _backup_flat_ref(ref_source)
                            # Record per-pair ref path so evaluation uses this copy
                            task["ref_file_override"] = pair_ref
                        else:
                            # intersection/strict: mask accumulates across sims onto shared ref
                            _restore_flat_ref_if_missing(ref_source)
                            apply_unified_mask_fn(info, var_name, ref_source, sim_source)
                            _backup_flat_ref(ref_source)

                task["ref_preprocessed"] = True

            except Exception as exc:
                task["preprocess_failed"] = True
                phase_errors.append(
                    make_phase_error_fn(
                        "preprocess",
                        f"preprocessing failed: {exc}",
                        variable=var_name,
                        sim=sim_source,
                        ref=ref_source,
                    )
                )
                if isinstance(exc, (FileNotFoundError, ValueError)):
                    # Expected errors: show concise message without full traceback
                    logger.error(
                        "Preprocessing failed: %s (sim=%s, ref=%s): %s",
                        var_name,
                        sim_source,
                        ref_source,
                        exc,
                    )
                else:
                    logger.exception("Preprocessing failed: %s (sim=%s, ref=%s)", var_name, sim_source, ref_source)

        # End-of-loop flat-NC restoration:
        # If a ref had any stn-involved prep AND any grid×grid task in this
        # variable, the stn prep's extract_station_data_if_needed deleted
        # the shared flat NC at data/<var>_ref_<ref>_<varname>.nc. The
        # downstream grid evaluation reads that exact path and would crash
        # with FileNotFoundError. Prefer restoring the last backed-up flat
        # ref, which preserves accumulated unified_mask edits; fall back to
        # re-running grid prep only if no backup is available.
        refs_needing_restore = refs_with_stn_prep & refs_with_grid_tasks
        for ref_to_restore in sorted(refs_needing_restore):
            if _restore_flat_ref_if_missing(ref_to_restore):
                continue
            grid_task = first_grid_task_per_ref.get(ref_to_restore)
            if grid_task is None:
                continue
            try:
                logger.info(
                    "Restoring flat ref NC for %s (%s) after stn-involved prep "
                    "deleted it (variable also has grid×grid tasks)",
                    var_name,
                    ref_to_restore,
                )
                info = build_bridge_runtime_info_fn(grid_task)
                DatasetProcessing(info).prepare_source("ref")
            except Exception as exc:
                phase_errors.append(
                    make_phase_error_fn(
                        "preprocess",
                        f"flat-ref restoration failed: {exc}",
                        variable=var_name,
                        ref=ref_to_restore,
                    )
                )
                logger.exception(
                    "Failed to restore flat ref NC for %s (%s)",
                    var_name,
                    ref_to_restore,
                )

        # Same restoration rule for grid simulation files. A grid sim can be
        # paired with both grid refs and station refs in the same variable.
        # The station-ref path extracts the grid sim to station files and
        # consumes/deletes data/<var>_sim_<sim>_<varname>.nc, but later grid
        # evaluation still needs the flat sim NC.
        sims_needing_restore = sims_with_stn_prep & sims_with_grid_tasks
        for sim_to_restore in sorted(sims_needing_restore):
            if _restore_flat_sim_if_missing(sim_to_restore):
                continue
            grid_task = first_grid_task_per_sim.get(sim_to_restore)
            if grid_task is None:
                continue
            try:
                logger.info(
                    "Restoring flat sim NC for %s (%s) after stn-involved prep "
                    "deleted it (variable also has grid×grid tasks)",
                    var_name,
                    sim_to_restore,
                )
                info = build_bridge_runtime_info_fn(grid_task)
                DatasetProcessing(info).prepare_source("sim")
            except Exception as exc:
                phase_errors.append(
                    make_phase_error_fn(
                        "preprocess",
                        f"flat-sim restoration failed: {exc}",
                        variable=var_name,
                        sim=sim_to_restore,
                    )
                )
                logger.exception(
                    "Failed to restore flat sim NC for %s (%s)",
                    var_name,
                    sim_to_restore,
                )

    finally:
        _cleanup_flat_backups()
    return phase_errors

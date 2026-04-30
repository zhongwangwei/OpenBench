"""Adapter to convert new OpenBenchConfig to legacy dict format.

The evaluation engine expects config as nested dicts with specific key
patterns. This adapter translates the new dataclass-based config.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any

from openbench.config.schema import OpenBenchConfig

logger = logging.getLogger(__name__)


def _resolve_root_relative_path(path_value: str, root_dir: str | None) -> str:
    """Resolve relative catalog paths against their dataset root_dir."""
    expanded = os.path.expanduser(os.path.expandvars(str(path_value)))
    if os.path.isabs(expanded) or not root_dir:
        return expanded
    return os.path.join(root_dir, expanded)


LEGACY_GENERAL_KEYS = {
    "basename",
    "basedir",
    "syear",
    "eyear",
    "min_year",
    "min_lat",
    "max_lat",
    "min_lon",
    "max_lon",
    "num_cores",
    "evaluation",
    "comparison",
    "statistics",
    "debug_mode",
    "only_drawing",
    "IGBP_groupby",
    "PFT_groupby",
    "Climate_zone_groupby",
    "unified_mask",
    "time_alignment",
    "generate_report",
    "weight",
    "compare_tim_res",
    "compare_tzone",
    "compare_grid_res",
}

LEGACY_ROOT_SECTION_KEYS = {
    "general",
    "evaluation_items",
    "metrics",
    "scores",
    "comparisons",
    "statistics",
}


@dataclass(frozen=True)
class RunnerConfig:
    """Runner-facing config derived from OpenBenchConfig."""

    basename: str
    basedir: str
    evaluation_items: dict[str, bool]
    metrics: list[str]
    scores: list[str]
    comparisons: list[str]
    statistics: list[str]
    general: dict[str, Any]

    @property
    def compare_tim_res(self) -> str:
        return self.general["compare_tim_res"]

    @property
    def compare_tzone(self) -> int:
        return self.general["compare_tzone"]

    @property
    def compare_grid_res(self) -> float:
        return self.general["compare_grid_res"]


@dataclass(frozen=True)
class RunnerBindings:
    """Adapter-produced bindings for the local runner."""

    runner_cfg: RunnerConfig
    namelists: "LegacyNamelists"
    figures: "LegacyFigureConfig"

    def iter_task_sources(self, variables: list[str]) -> list["EvaluationSource"]:
        """Return evaluation task sources."""
        tasks: list[EvaluationSource] = []
        ref_general = self.namelists.reference.get("general", {})
        sim_general = self.namelists.simulation.get("general", {})

        for var_name in variables:
            ref_source = ref_general.get(f"{var_name}_ref_source")
            sim_sources = sim_general.get(f"{var_name}_sim_source", [])

            if not ref_source:
                logger.warning("Skipping %s: no reference source", var_name)
                continue

            # Both ref_source and sim_source can be str or list[str] (v2.x compat)
            ref_sources_list = [ref_source] if isinstance(ref_source, str) else list(ref_source)
            if isinstance(sim_sources, str):
                sim_sources = [sim_sources]

            # Cartesian product: every (sim, ref) pair becomes a task
            for ref_s in ref_sources_list:
                for sim_source in sim_sources:
                    tasks.append(
                        EvaluationSource(
                            var_name=var_name,
                            sim_source=sim_source,
                            ref_source=ref_s,
                        )
                    )

        return tasks

    def has_grid_evaluation(self, variables: list[str]) -> "GridEvaluationEvidence":
        """Return typed evidence about whether any evaluation uses gridded data."""
        ref_general = self.namelists.reference.get("general", {})
        sim_general = self.namelists.simulation.get("general", {})

        for var_name in variables:
            ref_source = ref_general.get(f"{var_name}_ref_source")
            if not ref_source:
                continue

            ref_sources_list = [ref_source] if isinstance(ref_source, str) else list(ref_source)
            sim_sources = sim_general.get(f"{var_name}_sim_source", [])
            if isinstance(sim_sources, str):
                sim_sources = [sim_sources]

            # Full Cartesian product: must inspect every (ref, sim) pair.
            # Earlier code checked only sim_sources[0], which incorrectly returned
            # has_grid=False for mixed sim types like [SimStn, SimGrid] paired with
            # a stn ref — SimGrid required grid evaluation but was ignored.
            for ref_s in ref_sources_list:
                ref_dtype = self.namelists.reference.get(var_name, {}).get(f"{ref_s}_data_type")
                if not sim_sources:
                    if ref_dtype != "stn":
                        return GridEvaluationEvidence(has_grid=True)
                    continue
                for sim_s in sim_sources:
                    sim_dtype = self.namelists.simulation.get(var_name, {}).get(
                        f"{sim_s}_data_type"
                    )
                    if ref_dtype != "stn" or sim_dtype != "stn":
                        return GridEvaluationEvidence(has_grid=True)

        return GridEvaluationEvidence(has_grid=False)

    def build_report_config(self) -> "ReportContext":
        """Build the report-generator config payload."""
        return ReportContext(
            runner_cfg=self.runner_cfg,
            namelists=self.namelists,
        )

    def build_statistics_context(self, statistic_vars: list[str]) -> "StatisticsContext":
        """Build the statistics-phase payload.

        Each statistic method needs per-variable data source entries that point
        to the preprocessed NC files produced by the evaluation phase.  The
        file naming convention is:
            data/{Variable}_sim_{SimSource}_{varname}.nc
            data/{Variable}_ref_{RefSource}_{varname}.nc

        For two-source methods (Hellinger_Distance, Correlation, …) we emit
        ``{source}1`` (sim) and ``{source}2`` (ref) sub-entries.  For
        single-source methods (Basic / Mean / …) we emit one entry per
        sim source.
        """
        main_nl = self.namelists.main
        basedir = os.path.join(main_nl["general"]["basedir"], main_nl["general"]["basename"])
        data_dir = os.path.join(basedir, "data")
        stats_dir = os.path.join(basedir, "statistics")

        ref_general = self.namelists.reference.get("general", {})
        sim_general = self.namelists.simulation.get("general", {})
        eval_vars = list(self.runner_cfg.evaluation_items.keys())

        tim_res = main_nl["general"].get("compare_tim_res", "Month")
        grid_res = main_nl["general"].get("compare_grid_res", 0.5)
        syear = main_nl["general"]["syear"]
        eyear = main_nl["general"]["eyear"]

        # Two-source statistics methods that compare sim vs ref
        TWO_SOURCE_METHODS = {
            "Hellinger_Distance", "Correlation", "Functional_Response",
            "ANOVA", "Partial_Least_Squares_Regression",
        }
        # Three-source method
        THREE_SOURCE_METHODS = {"Three_Cornered_Hat"}

        def _base_entry(prefix_val: str, varname: str, varunit: str,
                        data_type: str = "grid") -> dict[str, Any]:
            """Build a single data-source entry dict."""
            return {
                "dir": data_dir,
                "data_type": data_type,
                "data_groupby": "single",
                "tim_res": tim_res,
                "grid_res": grid_res,
                "syear": syear,
                "eyear": eyear,
                "timezone": 0,
                "varname": varname,
                "varunit": varunit,
                "prefix": prefix_val,
                "suffix": "",
                "fulllist": "",
            }

        stats_nml: dict[str, Any] = {"general": {}}

        for stat in statistic_vars:
            stat_section: dict[str, Any] = {}
            source_names: list[str] = []

            for var_name in eval_vars:
                ref_source_raw = ref_general.get(f"{var_name}_ref_source")
                sim_sources = sim_general.get(f"{var_name}_sim_source", [])
                if isinstance(sim_sources, str):
                    sim_sources = [sim_sources]
                if not ref_source_raw or not sim_sources:
                    continue

                # Normalize ref to list (multi-ref support)
                ref_sources_list = (
                    [ref_source_raw] if isinstance(ref_source_raw, str) else list(ref_source_raw)
                )

                ref_nml_var = self.namelists.reference.get(var_name, {})
                sim_nml_var = self.namelists.simulation.get(var_name, {})

                # Iterate every ref × sim pair (Cartesian product, matching v2.x)
                for ref_source in ref_sources_list:
                    ref_varname = ref_nml_var.get(f"{ref_source}_varname", var_name)
                    ref_varunit = ref_nml_var.get(f"{ref_source}_varunit", "")
                    ref_dtype = ref_nml_var.get(f"{ref_source}_data_type", "grid")

                    for sim_source in sim_sources:
                        sim_varname = sim_nml_var.get(f"{sim_source}_varname", var_name)
                        sim_varunit = sim_nml_var.get(f"{sim_source}_varunit", "")
                        sim_dtype = sim_nml_var.get(f"{sim_source}_data_type", "grid")

                        # File prefixes match evaluation output naming
                        sim_file_prefix = f"{var_name}_sim_{sim_source}_{sim_varname}"
                        ref_file_prefix = f"{var_name}_ref_{ref_source}_{ref_varname}"

                        # Sanitised label for this var+sim+ref triple (multi-ref:
                        # include ref to keep entries distinct across ref sources)
                        pair_label = (
                            f"{var_name}_{sim_source}"
                            if len(ref_sources_list) == 1
                            else f"{var_name}_{sim_source}_{ref_source}"
                        )

                        if stat in TWO_SOURCE_METHODS:
                            source_names.append(pair_label)
                            for key, val in _base_entry(sim_file_prefix, sim_varname, sim_varunit, sim_dtype).items():
                                stat_section[f"{pair_label}1_{key}"] = val
                            for key, val in _base_entry(ref_file_prefix, ref_varname, ref_varunit, ref_dtype).items():
                                stat_section[f"{pair_label}2_{key}"] = val
                        elif stat in THREE_SOURCE_METHODS:
                            source_names.append(pair_label)
                            for key, val in _base_entry(sim_file_prefix, sim_varname, sim_varunit, sim_dtype).items():
                                stat_section[f"{pair_label}_{key}"] = val
                        else:
                            source_names.append(pair_label)
                            for key, val in _base_entry(sim_file_prefix, sim_varname, sim_varunit, sim_dtype).items():
                                stat_section[f"{pair_label}_{key}"] = val

                    # For Three_Cornered_Hat, add this ref as an extra source per ref
                    if stat in THREE_SOURCE_METHODS:
                        ref_label = f"{var_name}_{ref_source}"
                        if ref_label not in source_names:
                            source_names.append(ref_label)
                            ref_file_prefix = f"{var_name}_ref_{ref_source}_{ref_varname}"
                            for key, val in _base_entry(ref_file_prefix, ref_varname, ref_varunit, ref_dtype).items():
                                stat_section[f"{ref_label}_{key}"] = val

            # Add method-specific default parameters
            _STAT_DEFAULTS: dict[str, dict[str, Any]] = {
                "Hellinger_Distance": {"nbins": 25},
                "Functional_Response": {"nbins": 25},
                "Mann_Kendall_Trend_Test": {"significance_level": 0.05},
                "ANOVA": {"n_jobs": -1, "analysis_type": "one-way"},
                "Partial_Least_Squares_Regression": {
                    "max_components": 10, "n_splits": 5, "n_jobs": -1,
                },
            }
            if stat in _STAT_DEFAULTS:
                stat_section.update(_STAT_DEFAULTS[stat])

            stats_nml["general"][f"{stat}_data_source"] = ",".join(source_names)
            stats_nml[stat] = stat_section

        return StatisticsContext(
            namelists=self.namelists,
            stats_dir=stats_dir,
            stats_nml=stats_nml,
            num_cores=main_nl["general"].get("num_cores", 1),
            statistic_fig=self.figures.statistics,
        )

    def build_comparison_context(self) -> "ComparisonContext":
        """Build the comparison-phase payload."""
        return ComparisonContext(
            namelists=self.namelists,
            evaluation_items=list(self.runner_cfg.evaluation_items.keys()),
            score_vars=list(self.runner_cfg.scores),
            metric_vars=list(self.runner_cfg.metrics),
            comparison_fig=self.figures.comparison,
        )

    def build_groupby_context(self) -> "GroupbyContext":
        """Build the groupby-phase payload."""
        validation_fig = self.figures.igbp_groupby
        return GroupbyContext(
            namelists=self.namelists,
            evaluation_items=list(self.runner_cfg.evaluation_items.keys()),
            score_vars=list(self.runner_cfg.scores),
            metric_vars=list(self.runner_cfg.metrics),
            validation_fig=validation_fig,
            climate_zone_fig=self.figures.climate_zone_groupby,
        )

    def build_evaluation_fig_nml(self) -> "EvaluationFigureContext":
        """Build evaluator figure config payload."""
        return EvaluationFigureContext(figures=self.figures)

    def build_runtime_info_for(self, var_name: str, sim_source: str, ref_source: str) -> "BridgeRuntimeInfo":
        """Build bridge-provided runtime info for one evaluation task."""
        from openbench.config.legacy_processors import GeneralInfoReader

        info_reader = GeneralInfoReader(
            main_nl=self.namelists.main,
            sim_nml=self.namelists.simulation,
            ref_nml=self.namelists.reference,
            metric_vars=list(self.runner_cfg.metrics),
            score_vars=list(self.runner_cfg.scores),
            comparison_vars=list(self.runner_cfg.comparisons),
            statistic_vars=list(self.runner_cfg.statistics),
            item=var_name,
            sim_source=sim_source,
            ref_source=ref_source,
        )
        return BridgeRuntimeInfo.from_reader(info_reader)


@dataclass(frozen=True)
class ComparisonContext:
    """Typed payload for the comparison phase."""

    namelists: "LegacyNamelists"
    evaluation_items: list[str]
    score_vars: list[str]
    metric_vars: list[str]
    comparison_fig: dict[str, Any]


@dataclass(frozen=True)
class LegacyNamelists:
    """Typed container for the remaining legacy namelist payloads."""

    main: dict[str, Any]
    reference: dict[str, Any]
    simulation: dict[str, Any]


@dataclass(frozen=True)
class LegacyFigureConfig:
    """Typed container for the remaining legacy figure config payload."""

    raw: dict[str, Any]

    @property
    def comparison(self) -> dict[str, Any]:
        return self.raw.get("Comparison", {})

    @property
    def validation(self) -> dict[str, Any]:
        return self.raw.get("Validation", {})

    @property
    def igbp_groupby(self) -> dict[str, Any]:
        return self.raw.get("IGBP_groupby", self.validation)

    @property
    def climate_zone_groupby(self) -> dict[str, Any]:
        return self.raw.get("Climate_zone_groupby", self.igbp_groupby)

    @property
    def statistics(self) -> dict[str, Any]:
        return self.raw.get("Statistic", {})


@dataclass(frozen=True)
class EvaluationSource:
    """Typed evaluation task source entry."""

    var_name: str
    sim_source: str
    ref_source: str


@dataclass(frozen=True)
class GridEvaluationEvidence:
    """Typed evidence for whether statistics can run on gridded outputs."""

    has_grid: bool


@dataclass(frozen=True)
class ReportContext:
    """Typed payload for report generation."""

    runner_cfg: RunnerConfig
    namelists: "LegacyNamelists"

    @property
    def evaluation_items(self) -> list[str]:
        return list(self.runner_cfg.evaluation_items.keys())

    def to_report_config(self) -> dict[str, Any]:
        """Convert to the legacy-shaped dict expected by ReportGenerator."""
        return {
            "evaluation_items": list(self.runner_cfg.evaluation_items.keys()),
            "metrics": {metric: True for metric in self.runner_cfg.metrics},
            "scores": {score: True for score in self.runner_cfg.scores},
            "comparisons": {item: True for item in self.runner_cfg.comparisons},
            "statistics": {item: True for item in self.runner_cfg.statistics},
            "general": dict(self.runner_cfg.general),
            "ref_nml": dict(self.namelists.reference) if self.namelists.reference else {},
            "sim_nml": dict(self.namelists.simulation) if self.namelists.simulation else {},
        }


@dataclass(frozen=True)
class EvaluationFigureContext:
    """Typed payload for evaluator figure config."""

    figures: "LegacyFigureConfig"

    def to_fig_nml(self) -> dict[str, Any]:
        """Convert to the legacy-shaped figure config expected by evaluators."""
        return dict(self.figures.raw)


@dataclass(frozen=True)
class BridgeRuntimeInfo:
    """Typed payload for bridge-provided runtime info."""

    payload: dict[str, Any]

    @classmethod
    def from_reader(cls, reader: Any) -> "BridgeRuntimeInfo":
        """Snapshot public reader attributes without depending on a legacy to_dict() helper."""
        return cls(payload={k: v for k, v in vars(reader).items() if not k.startswith("_")})

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", dict(self.payload))

    def to_info(self) -> dict[str, Any]:
        """Convert to the legacy-shaped runtime info expected by runner/evaluator code."""
        return dict(self.payload)


@dataclass(frozen=True)
class GroupbyContext:
    """Typed payload for the groupby phase."""

    namelists: "LegacyNamelists"
    evaluation_items: list[str]
    score_vars: list[str]
    metric_vars: list[str]
    validation_fig: dict[str, Any]
    climate_zone_fig: dict[str, Any]


@dataclass(frozen=True)
class StatisticsContext:
    """Typed payload for the statistics phase."""

    namelists: "LegacyNamelists"
    stats_dir: str
    stats_nml: dict[str, Any]
    num_cores: int
    statistic_fig: dict[str, Any]


def _resolve_varname(profile_var, root_dir: str | None = None) -> tuple[str, str, str]:
    """Resolve a variable name from profile, handling fallbacks.

    Checks the data files to determine which variable actually exists.
    Uses the primary varname and any normalized fallback descriptors.

    Args:
        profile_var: A VariableMapping object or raw varname.
        root_dir: Path to simulation data directory to check actual files.

    Returns:
        (resolved_varname, resolved_varunit, convert_expr) tuple.
        convert_expr is a Python expression string (e.g. "value * 12.011")
        to apply after reading the data, or "" if no conversion needed.
    """
    # Extract primary varname and fallbacks from profile_var
    if hasattr(profile_var, "varname"):
        # It's a VariableMapping object
        primary = profile_var.varname
        primary_unit = profile_var.varunit
        fallbacks = profile_var.fallbacks or []
    elif isinstance(profile_var, str):
        return profile_var, "", ""
    else:
        return str(profile_var), "", ""

    all_names = [primary] + [fb.varname for fb in fallbacks]

    # If no root_dir, return primary
    if not root_dir or not all_names:
        return primary, primary_unit, ""

    # Check data files to find which variable exists
    from openbench.data.coordinates import glob_nc

    nc_files = glob_nc(root_dir)
    if not nc_files:
        return primary, primary_unit, ""

    try:
        import xarray as xr

        # Use a context manager so the file handle is released even if
        # data_vars probing or any later step inside this try block raises.
        with xr.open_dataset(nc_files[0]) as ds:
            available = set(ds.data_vars)

        # Try primary first
        if primary in available:
            return primary, primary_unit, ""

        # Try each fallback
        for fb in fallbacks:
            if fb.varname in available:
                convert_expr = fb.convert or ""
                if convert_expr:
                    logger.info(
                        "Varname fallback: %s not found → using %s (convert: %s)",
                        primary, fb.varname, convert_expr,
                    )
                else:
                    logger.info(
                        "Varname fallback: %s not found → using %s (%s)",
                        primary, fb.varname, fb.varunit or "no unit",
                    )
                # When convert is present, the expression transforms data to
                # the primary variable's unit, so report primary_unit.
                # When no convert, use the fallback's own unit.
                effective_unit = primary_unit if convert_expr else (fb.varunit or primary_unit)
                return fb.varname, effective_unit, convert_expr

        logger.warning("None of %s found in data, using primary: %s", all_names, primary)
    except Exception as exc:
        logger.debug("Data file inspection failed, using primary varname %s: %s", primary, exc)

    return primary, primary_unit, ""


def _find_nc_dir(ref_dir: str, data_root: str, sub_dir: str | None) -> str:
    """Find directory containing NC files, with two fallback strategies.

    1. If ref_dir has no NC files, check one level of subdirectories.
    2. If still nothing and data_root is a resolution directory (MidRes/HigRes),
       try the equivalent LowRes path.

    Returns the best directory found, or the original ref_dir if nothing better.
    """
    from openbench.data.coordinates import glob_nc

    if os.path.isdir(ref_dir) and glob_nc(ref_dir):
        return ref_dir

    # Strategy 1: check subdirectories (e.g., 0p25deg-daily/)
    if os.path.isdir(ref_dir):
        for child in sorted(os.listdir(ref_dir)):
            child_path = os.path.join(ref_dir, child)
            if os.path.isdir(child_path) and glob_nc(child_path):
                logger.info("NC files found in subdirectory: %s", child_path)
                return child_path

    # Strategy 2: fall back from MidRes/HigRes to LowRes
    if data_root and sub_dir:
        for res in ("MidRes", "HigRes"):
            if res in data_root:
                lowres_root = data_root.replace(res, "LowRes")
                lowres_dir = os.path.join(lowres_root, sub_dir)
                if os.path.isdir(lowres_dir) and glob_nc(lowres_dir):
                    logger.info("Falling back to LowRes: %s", lowres_dir)
                    return lowres_dir
                # Also check subdirectories of LowRes
                if os.path.isdir(lowres_dir):
                    for child in sorted(os.listdir(lowres_dir)):
                        child_path = os.path.join(lowres_dir, child)
                        if os.path.isdir(child_path) and glob_nc(child_path):
                            logger.info("Falling back to LowRes subdirectory: %s", child_path)
                            return child_path
                break

    return ref_dir


def build_runner_config(cfg: OpenBenchConfig) -> RunnerConfig:
    """Build runner-facing config from OpenBenchConfig."""
    from openbench.config.resolver import derive_target_resolution_context

    target_ctx = derive_target_resolution_context(cfg)

    general = {
        "basename": cfg.project.name,
        "basedir": cfg.project.output_dir,
        "syear": cfg.project.years[0],
        "eyear": cfg.project.years[1],
        "min_year": cfg.project.min_year_threshold,
        "min_lat": cfg.project.lat_range[0],
        "max_lat": cfg.project.lat_range[1],
        "min_lon": cfg.project.lon_range[0],
        "max_lon": cfg.project.lon_range[1],
        "num_cores": cfg.project.num_cores or max(1, os.cpu_count() or 1),
        "evaluation": True,
        "comparison": cfg.comparison.enabled,
        "statistics": cfg.statistics.enabled,
        "debug_mode": cfg.project.debug_mode,
        "only_drawing": cfg.project.only_drawing,
        "IGBP_groupby": cfg.project.IGBP_groupby,
        "PFT_groupby": cfg.project.PFT_groupby,
        "Climate_zone_groupby": cfg.project.climate_zone_groupby,
        "unified_mask": cfg.project.unified_mask,
        "time_alignment": cfg.project.time_alignment,
        "generate_report": cfg.project.generate_report,
        "weight": cfg.project.weight or "area",
        "compare_tim_res": target_ctx.tim_res or "Month",
        "compare_tzone": cfg.project.timezone or 0,
        "compare_grid_res": target_ctx.grid_res if target_ctx.grid_res is not None else 0.5,
    }

    evaluation_items = {var: True for var in cfg.evaluation.variables}

    if cfg.metrics:
        metrics_dict = {m: True for m in cfg.metrics}
    else:
        metrics_dict = {"bias": True, "RMSE": True, "correlation": True}

    if cfg.scores:
        scores_dict = {s: True for s in cfg.scores}
    else:
        scores_dict = {"Overall_Score": True}

    if cfg.comparison.items:
        comparisons_dict = {c: True for c in cfg.comparison.items}
    else:
        comparisons_dict = {"Taylor_Diagram": True, "HeatMap": True}

    if cfg.statistics.items:
        statistics_dict = {s: True for s in cfg.statistics.items}
    else:
        statistics_dict = {}

    return RunnerConfig(
        basename=general["basename"],
        basedir=general["basedir"],
        evaluation_items=evaluation_items,
        metrics=list(metrics_dict.keys()),
        scores=list(scores_dict.keys()),
        comparisons=list(comparisons_dict.keys()),
        statistics=list(statistics_dict.keys()),
        general=general,
    )

def build_runner_bindings(cfg: OpenBenchConfig) -> RunnerBindings:
    """Build the runner-facing config plus legacy-shaped bindings still needed downstream."""
    runner_cfg = build_runner_config(cfg)
    main_nl, ref_nml, sim_nml = build_legacy_namelists(cfg)
    fig_nml = build_fig_nml()
    return RunnerBindings(
        runner_cfg=runner_cfg,
        namelists=LegacyNamelists(
            main=main_nl,
            reference=ref_nml,
            simulation=sim_nml,
        ),
        figures=LegacyFigureConfig(raw=fig_nml),
    )


def to_legacy_config(cfg: OpenBenchConfig) -> dict[str, Any]:
    """Convert OpenBenchConfig to the legacy dict format."""
    runner_cfg = build_runner_config(cfg)

    return {
        "general": runner_cfg.general,
        "evaluation_items": runner_cfg.evaluation_items,
        "metrics": {m: True for m in runner_cfg.metrics},
        "scores": {s: True for s in runner_cfg.scores},
        "comparisons": {c: True for c in runner_cfg.comparisons},
        "statistics": {s: True for s in runner_cfg.statistics},
    }


def build_fig_nml() -> dict[str, Any]:
    """Build the figure namelist from bundled figure config files.

    Reads all figure config YAML files from the package's data/fignml/ directory
    and organizes them into the structure expected by the evaluation code:
        fig_nml["make_geo_plot_index"] = {...}   (validation configs, flattened)
        fig_nml["Comparison"]["Taylor_Diagram"] = {...}
        fig_nml["Statistic"]["Basic"] = {...}

    Returns:
        Processed fig_nml dict.
    """
    from pathlib import Path

    import yaml

    fignml_dir = Path(__file__).parent.parent / "data" / "fignml"
    figlib_path = fignml_dir / "figlib.yaml"

    if not figlib_path.exists():
        logger.warning("figlib.yaml not found at %s, visualization will be skipped", figlib_path)
        return {}

    with open(figlib_path) as f:
        figlib = yaml.safe_load(f)

    fig_nml: dict[str, Any] = {}

    # Process validation configs — keys go directly into fig_nml (flattened)
    for key, rel_path in figlib.get("validation_nml", {}).items():
        config_name = key.replace("_source", "")
        filename = Path(rel_path).name
        config_path = fignml_dir / filename
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f)
            fig_nml[config_name] = data.get("general", data)
        else:
            logger.debug("Figure config not found: %s", config_path)

    # Process comparison configs — nested under fig_nml["Comparison"]
    comparison = {}
    for key, rel_path in figlib.get("comparison_nml", {}).items():
        config_name = key.replace("_source", "")
        filename = Path(rel_path).name
        config_path = fignml_dir / filename
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f)
            comparison[config_name] = data.get("general", data)
    fig_nml["Comparison"] = comparison

    # Process statistic configs — nested under fig_nml["Statistic"]
    statistic = {}
    for key, rel_path in figlib.get("statistic_nml", {}).items():
        config_name = key.replace("_source", "")
        filename = Path(rel_path).name
        config_path = fignml_dir / filename
        if config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f)
            statistic[config_name] = data.get("general", data)
    fig_nml["Statistic"] = statistic

    # Keep raw registry sections for legacy figure-updater compatibility
    fig_nml["validation_nml"] = figlib.get("validation_nml", {})
    fig_nml["comparison_nml"] = figlib.get("comparison_nml", {})
    fig_nml["statistic_nml"] = figlib.get("statistic_nml", {})

    return fig_nml


def build_legacy_namelists(cfg: OpenBenchConfig) -> tuple[dict, dict, dict]:
    """Build legacy ref_nml, sim_nml, and main_nl from new config.

    Uses the registry to resolve reference dataset variable mappings
    and model profiles to resolve simulation variable mappings.

    Returns:
        (main_nl, ref_nml, sim_nml) tuple of legacy-format dicts.
    """
    from openbench.data.registry.manager import get_registry

    registry = get_registry()
    legacy = to_legacy_config(cfg)

    # --- main_nl: general settings + evaluation_items ---
    main_nl = {
        "general": legacy["general"],
        "evaluation_items": legacy["evaluation_items"],
        "metrics": legacy["metrics"],
        "scores": legacy["scores"],
        "comparisons": legacy["comparisons"],
        "statistics": legacy["statistics"],
    }

    # --- ref_nml ---
    from openbench.config.resolver import resolve_all_references

    ref_general: dict[str, Any] = {}
    ref_sections: dict[str, dict[str, Any]] = {}

    # strict mode: unresolved references raise ConfigError here
    resolved_refs = resolve_all_references(cfg, registry, strict=cfg.project.strict_reference)

    # Track which variables have already had a section initialized so we
    # can append additional refs from the same variable without overwriting.
    # When a variable has multiple resolved refs, ref_general[<var>_ref_source]
    # becomes a list; sections accumulate per-source keys.
    _ref_source_lists: dict[str, list[str]] = {}
    _ref_original_lists: dict[str, list[str]] = {}

    for r in resolved_refs:
        var_name = r.var_name

        if r.status == "not_found" and not r.source_name:
            logger.warning("No reference source configured for variable %s, skipping", var_name)
            continue

        resolved_name = r.resolved_name
        # Accumulate ref source names per variable (str when single, list when multi)
        _ref_source_lists.setdefault(var_name, []).append(resolved_name)
        if resolved_name != r.source_name:
            _ref_original_lists.setdefault(var_name, []).append(r.source_name)
        section: dict[str, Any] = ref_sections.get(var_name, {})
        prefix = resolved_name

        ref_ds = r.ref_ds
        var_map = r.var_map

        if r.status == "ok":

            # Construct directory: station data uses its own root_dir;
            # grid data prefers data_root (shared grid directory) over registry root_dir
            if ref_ds.data_type == "stn":
                data_root = ref_ds.root_dir or cfg.reference.data_root or ""
            else:
                data_root = cfg.reference.data_root or ref_ds.root_dir or ""
            if not data_root:
                logger.warning(
                    "No data_root or root_dir for reference %s variable %s. "
                    "Set reference.data_root in config or register with --root-dir.",
                    resolved_name, var_name,
                )
            ref_dir = data_root
            if var_map.sub_dir:
                ref_dir = os.path.join(ref_dir, var_map.sub_dir) if ref_dir else var_map.sub_dir

            # If ref_dir has no NC files, search one level deeper (e.g., 0p25deg-daily/)
            if ref_dir and ref_ds.data_type != "stn":
                ref_dir = _find_nc_dir(ref_dir, data_root, var_map.sub_dir)

            section[f"{prefix}_data_type"] = ref_ds.data_type
            section[f"{prefix}_varname"] = var_map.varname
            section[f"{prefix}_varunit"] = var_map.varunit
            section[f"{prefix}_data_groupby"] = ref_ds.data_groupby
            section[f"{prefix}_tim_res"] = ref_ds.tim_res
            section[f"{prefix}_grid_res"] = ref_ds.grid_res
            # Both lines must guard against years=None (not just falsy).
            # The previous eyear line called len(ref_ds.years) without a
            # None check, raising TypeError when years was None.
            ref_years = ref_ds.years or []
            section[f"{prefix}_syear"] = ref_years[0] if ref_years else cfg.project.years[0]
            section[f"{prefix}_eyear"] = ref_years[1] if len(ref_years) > 1 else cfg.project.years[1]
            section[f"{prefix}_dir"] = ref_dir
            section[f"{prefix}_prefix"] = var_map.prefix
            section[f"{prefix}_suffix"] = var_map.suffix
            section[f"{prefix}_timezone"] = ref_ds.timezone

            # Optional station-related fields
            if var_map.fulllist:
                section[f"{prefix}_fulllist"] = _resolve_root_relative_path(
                    var_map.fulllist,
                    ref_ds.root_dir or data_root,
                )
            elif ref_ds.fulllist:
                section[f"{prefix}_fulllist"] = _resolve_root_relative_path(
                    ref_ds.fulllist,
                    ref_ds.root_dir or data_root,
                )
            if var_map.max_uparea is not None:
                section[f"{prefix}_max_uparea"] = var_map.max_uparea
            if var_map.min_uparea is not None:
                section[f"{prefix}_min_uparea"] = var_map.min_uparea
        else:
            if r.message:
                logger.warning("Reference resolution: %s", r.message)
            else:
                logger.warning(
                    "Reference %s not found; variable %s using minimal defaults.",
                    r.source_name, var_name,
                )
            # Provide minimal defaults so processing doesn't crash on missing keys
            section[f"{prefix}_data_type"] = "grid"
            section[f"{prefix}_varname"] = var_name
            section[f"{prefix}_varunit"] = ""
            section[f"{prefix}_data_groupby"] = "Year"
            section[f"{prefix}_tim_res"] = "Month"
            section[f"{prefix}_grid_res"] = None
            section[f"{prefix}_syear"] = cfg.project.years[0]
            section[f"{prefix}_eyear"] = cfg.project.years[1]
            section[f"{prefix}_dir"] = cfg.reference.data_root or ""
            section[f"{prefix}_prefix"] = ""
            section[f"{prefix}_suffix"] = ""
            section[f"{prefix}_timezone"] = 0

        ref_sections[var_name] = section

    # Collapse single-source lists to plain strings; keep lists when multi-source.
    # Matches v2.x convention: evaluator wraps str -> [str] internally so either
    # form is acceptable, but storing single-string is more idiomatic for the
    # 99% case and easier on diff inspection.
    for var_name, names in _ref_source_lists.items():
        ref_general[f"{var_name}_ref_source"] = names[0] if len(names) == 1 else names
    for var_name, originals in _ref_original_lists.items():
        if originals:
            ref_general[f"{var_name}_ref_source_original"] = (
                originals[0] if len(originals) == 1 else originals
            )

    ref_nml = {"general": ref_general, **ref_sections}

    # --- sim_nml ---
    sim_general: dict[str, Any] = {}
    sim_sections: dict[str, dict[str, Any]] = {}

    for var_name in cfg.evaluation.variables:
        sim_sources: list[str] = []
        var_section: dict[str, Any] = {}

        for sim_label, sim_entry in cfg.simulation.items():
            model_name = sim_entry.model
            sim_sources.append(sim_label)

            model_profile = registry.get_model(model_name)

            # Determine variable mapping: inline overrides > model profile > fallback
            inline_vars = (sim_entry.variables or {}).get(var_name, {})

            # Entry-level prefix/suffix (shared across all variables for this sim)
            entry_prefix = sim_entry.prefix or ""
            entry_suffix = sim_entry.suffix or ""

            if model_profile and var_name in model_profile.variables:
                profile_var = model_profile.variables[var_name]
                if "varname" in inline_vars:
                    # Inline override — no fallback resolution
                    varname = inline_vars["varname"]
                    varunit = inline_vars.get("varunit", profile_var.varunit)
                    convert_expr = ""
                else:
                    # Resolve from profile with fallback chain
                    varname, varunit, convert_expr = _resolve_varname(profile_var, sim_entry.root_dir)
                    varunit = inline_vars.get("varunit", varunit)
                var_prefix = inline_vars.get("prefix", entry_prefix or profile_var.prefix)
                var_suffix = inline_vars.get("suffix", entry_suffix or profile_var.suffix)
            elif inline_vars:
                varname = inline_vars.get("varname", var_name)
                varunit = inline_vars.get("varunit", "")
                convert_expr = ""
                var_prefix = inline_vars.get("prefix", entry_prefix)
                var_suffix = inline_vars.get("suffix", entry_suffix)
            else:
                logger.warning(
                    "No variable mapping for %s in model %s (label %s); using variable name as varname",
                    var_name,
                    model_name,
                    sim_label,
                )
                varname = var_name
                varunit = ""
                convert_expr = ""
                var_prefix = entry_prefix
                var_suffix = entry_suffix

            # Data type / resolution: inline override > sim_entry override > model profile > defaults
            data_type = sim_entry.data_type or (model_profile.data_type if model_profile else "grid")
            grid_res = sim_entry.grid_res or (model_profile.grid_res if model_profile else None)
            tim_res = sim_entry.tim_res or (model_profile.tim_res if model_profile else "Month")

            # Construct sim directory: root_dir from sim_entry, optionally with sub_dir from profile
            sim_dir = sim_entry.root_dir
            if model_profile and var_name in model_profile.variables:
                profile_sub = model_profile.variables[var_name].sub_dir
                if profile_sub:
                    sim_dir = os.path.join(sim_dir, profile_sub)

            prefix = sim_label
            var_section[f"{prefix}_model"] = model_name
            var_section[f"{prefix}_data_type"] = data_type
            var_section[f"{prefix}_varname"] = varname
            var_section[f"{prefix}_varunit"] = varunit
            if convert_expr:
                var_section[f"{prefix}_convert"] = convert_expr
            var_section[f"{prefix}_data_groupby"] = (
                sim_entry.data_groupby or inline_vars.get("data_groupby", "Year")
            )
            var_section[f"{prefix}_tim_res"] = tim_res
            var_section[f"{prefix}_grid_res"] = grid_res
            var_section[f"{prefix}_syear"] = cfg.project.years[0]
            var_section[f"{prefix}_eyear"] = cfg.project.years[1]
            var_section[f"{prefix}_dir"] = sim_dir
            var_section[f"{prefix}_prefix"] = var_prefix
            var_section[f"{prefix}_suffix"] = var_suffix

            # Pass prefix_fallback if this variable may be in alternative files
            if model_profile and var_name in model_profile.variables:
                pf = model_profile.variables[var_name].prefix_fallback
                if pf:
                    var_section[f"{prefix}_prefix_fallback"] = pf
            var_section[f"{prefix}_timezone"] = inline_vars.get("timezone", 0)

            # Optional station-related fields: inline config > entry-level > model profile
            if "fulllist" in inline_vars:
                var_section[f"{prefix}_fulllist"] = inline_vars["fulllist"]
            elif sim_entry.fulllist:
                var_section[f"{prefix}_fulllist"] = sim_entry.fulllist
            if "max_uparea" in inline_vars:
                var_section[f"{prefix}_max_uparea"] = inline_vars["max_uparea"]
            if "min_uparea" in inline_vars:
                var_section[f"{prefix}_min_uparea"] = inline_vars["min_uparea"]

        sim_general[f"{var_name}_sim_source"] = sim_sources
        sim_sections[var_name] = var_section

    sim_nml = {"general": sim_general, **sim_sections}

    return main_nl, ref_nml, sim_nml

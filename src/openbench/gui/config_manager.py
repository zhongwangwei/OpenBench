# -*- coding: utf-8 -*-
"""
Configuration manager for loading, saving, and validating NML configs.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from pathlib import Path

import yaml

from openbench.gui.path_utils import convert_paths_in_dict, get_openbench_root, to_absolute_path

_BUILTIN_MODEL_KEYS: Optional[Set[str]] = None


def registry_model_profile(model_name: str):
    """Return the registry ModelProfile for a bare model name, else None.

    The scan-based Simulation Data page stores registry model names (e.g.
    ``CoLM2024``) in ``model_namelist``, not file paths. Anything that looks
    like a path or a definition file is not resolved here.
    """
    name = str(model_name or "")
    if not name or "/" in name or "\\" in name or name.endswith((".yaml", ".nml")):
        return None
    try:
        from openbench.data.registry.manager import get_registry

        return get_registry().get_model(name)
    except Exception:
        return None


def model_definition_from_registry(model_name: str, selected_items: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    """Build a model definition dict from the registry for export.

    Returns the legacy-shaped mapping (``general`` + one section per
    evaluation item) used by the exported ``nml/sim/models/*.yaml`` files,
    or None when ``model_name`` is not a bare registry model name.
    """
    profile = registry_model_profile(model_name)
    if profile is None:
        return None
    content: Dict[str, Any] = {"general": {"model": getattr(profile, "name", model_name)}}
    variables = getattr(profile, "variables", {}) or {}
    for item in selected_items or sorted(variables):
        mapping = variables.get(item)
        if mapping is not None:
            content[item] = mapping.to_dict()
    return content


def is_builtin_model(model_name: str) -> bool:
    """True when the model ships with the OpenBench package registry.

    Built-in models exist on every install (including remote servers), so
    they never need their profile uploaded; user-registered models do.
    """
    global _BUILTIN_MODEL_KEYS
    if _BUILTIN_MODEL_KEYS is None:
        names: Set[str] = set()
        try:
            from openbench.data.registry.manager import REGISTRY_DIR
            from openbench.util.names import normalize_name

            catalog = REGISTRY_DIR / "model_catalog.yaml"
            if catalog.is_file():
                data = yaml.safe_load(catalog.read_text(encoding="utf-8")) or {}
                names.update(normalize_name(key) for key in data)
            models_dir = REGISTRY_DIR / "models"
            if models_dir.is_dir():
                for entry in models_dir.iterdir():
                    if entry.name.endswith(".yaml"):
                        names.add(normalize_name(entry.name[: -len(".yaml")]))
        except Exception:
            return False
        _BUILTIN_MODEL_KEYS = names
    try:
        from openbench.util.names import normalize_name

        return normalize_name(str(model_name or "")) in _BUILTIN_MODEL_KEYS
    except Exception:
        return False


class ConfigManager:
    """Manages NML configuration loading, saving, and validation."""

    def __init__(self):
        self._last_dir = os.path.expanduser("~")

    def load_from_yaml(self, path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the YAML file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self._last_dir = os.path.dirname(path)
        return config or {}

    def save_to_yaml(self, config: Dict[str, Any], path: str):
        """
        Save configuration to a YAML file.

        Args:
            config: Configuration dictionary
            path: Output file path
        """
        # Ensure directory exists
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Atomic write: dump to a sibling temp file then os.replace onto the
        # target so a Ctrl+C / power loss / disk-full mid-write cannot leave
        # a truncated YAML clobbering a previously-working config.
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def is_unified_config(config: Dict[str, Any]) -> bool:
        """Return True when *config* looks like a v3 unified openbench.yaml."""
        return any(key in config for key in ("project", "evaluation", "reference", "simulation"))

    def unified_to_gui_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a v3 unified openbench.yaml dict to the GUI's internal shape.

        The GUI pages still read the legacy-shaped in-memory sections
        (``general``, ``ref_data``, ``sim_data``).  Loading a v3
        ``openbench.yaml`` must therefore reverse the export mapping enough
        for pages to show the user's existing values instead of defaults.
        """
        project = config.get("project", {}) or {}
        evaluation = config.get("evaluation", {}) or {}
        reference = config.get("reference", {}) or {}
        simulation = config.get("simulation", {}) or {}

        years = project.get("years") or [2000, 2020]
        lat_range = project.get("lat_range") or [-90.0, 90.0]
        lon_range = project.get("lon_range") or [-180.0, 180.0]

        general: Dict[str, Any] = {
            "basename": project.get("name", ""),
            "basedir": project.get("output_dir", "./output"),
            "compare_tim_res": project.get("tim_res", "month"),
            "compare_tzone": project.get("timezone", 0.0),
            "compare_grid_res": project.get("grid_res", 2.0),
            "syear": years[0] if len(years) > 0 else 2000,
            "eyear": years[1] if len(years) > 1 else 2020,
            "min_year": project.get("min_year_threshold", 1.0),
            "min_lat": lat_range[0] if len(lat_range) > 0 else -90.0,
            "max_lat": lat_range[1] if len(lat_range) > 1 else 90.0,
            "min_lon": lon_range[0] if len(lon_range) > 0 else -180.0,
            "max_lon": lon_range[1] if len(lon_range) > 1 else 180.0,
            "time_alignment": project.get("time_alignment", "intersection"),
            "num_cores": project.get("num_cores", 4),
            "evaluation": True,
            "comparison": bool((config.get("comparison") or {}).get("enabled", False)),
            "statistics": bool((config.get("statistics") or {}).get("enabled", False)),
            "debug_mode": project.get("debug_mode", False),
            "only_drawing": project.get("only_drawing", False),
            "unified_mask": project.get("unified_mask", True),
            "generate_report": project.get("generate_report", True),
            "dask": project.get("dask", {}) or {},
            "io": project.get("io", {}) or {},
            "IGBP_groupby": project.get("IGBP_groupby", False),
            "PFT_groupby": project.get("PFT_groupby", False),
            "Climate_zone_groupby": project.get(
                "Climate_zone_groupby",
                project.get("climate_zone_groupby", False),
            ),
            "weight": "none" if project.get("weight") is None else project.get("weight"),
            "execution_mode": "local",
        }

        variables = list(evaluation.get("variables") or [])
        evaluation_items = {var: True for var in variables}
        metrics = {name: True for name in (config.get("metrics") or [])}
        scores = {name: True for name in (config.get("scores") or [])}
        comparisons = {name: True for name in ((config.get("comparison") or {}).get("items") or [])}
        statistics = {name: True for name in ((config.get("statistics") or {}).get("items") or [])}

        ref_general: Dict[str, Any] = {}
        if reference.get("data_root"):
            ref_general["data_root"] = reference["data_root"]
        ref_sources = reference.get("sources") if isinstance(reference.get("sources"), dict) else reference
        for var in variables:
            if var in ref_sources:
                ref_general[f"{var}_ref_source"] = ref_sources[var]

        sim_defaults = simulation.get("_defaults", {}) if isinstance(simulation, dict) else {}
        sim_entries = {k: v for k, v in simulation.items() if k != "_defaults"} if isinstance(simulation, dict) else {}
        sim_general = {f"{var}_sim_source": list(sim_entries.keys()) for var in variables}
        sim_source_configs: Dict[str, Dict[str, Any]] = {}
        for label, raw_entry in sim_entries.items():
            if not isinstance(raw_entry, dict):
                continue
            entry = {**sim_defaults, **raw_entry}
            variables_override = {}
            if isinstance(sim_defaults.get("variables"), dict) or isinstance(raw_entry.get("variables"), dict):
                variables_override = {
                    **(sim_defaults.get("variables") or {}),
                    **(raw_entry.get("variables") or {}),
                }
            source_general: Dict[str, Any] = {
                "model": entry.get("model", ""),
                "model_namelist": entry.get("model", ""),
                "root_dir": entry.get("root_dir", ""),
            }
            for key in ("data_type", "grid_res", "tim_res", "data_groupby", "prefix", "suffix", "fulllist"):
                if entry.get(key) is not None:
                    source_general[key] = entry[key]
            sim_source_configs[label] = {
                "general": source_general,
                "variables": variables_override,
            }

        # CLI configs have no scan-root concept (only per-case root_dir), but
        # the GUI's Simulation Data page needs one to restore its scan field.
        from openbench.gui.path_utils import infer_common_scan_root

        sim_scan_root = infer_common_scan_root(
            [cfg.get("general", {}).get("root_dir", "") for cfg in sim_source_configs.values()]
        )

        return {
            "general": general,
            "evaluation_items": evaluation_items,
            "metrics": metrics,
            "scores": scores,
            "comparisons": comparisons,
            "statistics": statistics,
            "ref_data": {"general": ref_general, "def_nml": {}},
            "sim_data": {
                "general": sim_general,
                "def_nml": {},
                "source_configs": sim_source_configs,
                **({"_scan_root": sim_scan_root} if sim_scan_root else {}),
            },
        }

    def generate_main_nml(
        self,
        config: Dict[str, Any],
        openbench_root: Optional[str] = None,
        output_dir: Optional[str] = None,
        remote_openbench_path: Optional[str] = None,
    ) -> str:
        """
        Generate main NML YAML content.

        Args:
            config: Full configuration dictionary
            openbench_root: OpenBench root directory for generating absolute paths
            output_dir: Output directory path (for nml paths)
            remote_openbench_path: Deprecated. Kept for API compatibility;
                v3 support namelists are generated into the case output
                ``nml/`` directory instead of being read from an OpenBench
                installation tree.

        Returns:
            YAML string
        """
        main_config = {}

        # General section
        general = config.get("general", {})
        basename = general.get("basename", "config")

        # Check if in remote mode
        is_remote = general.get("execution_mode") == "remote"

        # Use provided output_dir, or compute from config
        if output_dir is None:
            basedir = general.get("basedir", "")
            if basedir and (os.path.isabs(basedir) or basedir.startswith("/")):
                # Normalize path to handle trailing slashes and multiple separators
                if is_remote:
                    # Use forward slashes for remote paths
                    normalized_basedir = basedir.rstrip("/").replace("\\", "/")
                    basedir_basename = normalized_basedir.split("/")[-1]
                    if basedir_basename == basename:
                        output_dir = normalized_basedir
                    else:
                        output_dir = f"{normalized_basedir}/{basename}"
                else:
                    normalized_basedir = os.path.normpath(basedir)
                    # Check if basedir already ends with basename to avoid duplication
                    if os.path.basename(normalized_basedir) == basename:
                        output_dir = normalized_basedir
                    else:
                        output_dir = os.path.normpath(os.path.join(normalized_basedir, basename))
            elif openbench_root:
                output_dir = os.path.normpath(os.path.join(openbench_root, "output", basename))
            else:
                output_dir = os.path.normpath(general.get("basedir", "./output"))

        # Generate absolute paths - ref and sim are in nml folder
        if is_remote:
            # Use forward slashes for remote paths
            output_dir = output_dir.replace("\\", "/")
            nml_dir = f"{output_dir.rstrip('/')}/nml"
            ref_nml_path = f"{nml_dir}/ref-{basename}.yaml"
            sim_nml_path = f"{nml_dir}/sim-{basename}.yaml"
        else:
            nml_dir = os.path.normpath(os.path.join(output_dir, "nml"))
            ref_nml_path = os.path.normpath(os.path.join(nml_dir, f"ref-{basename}.yaml"))
            sim_nml_path = os.path.normpath(os.path.join(nml_dir, f"sim-{basename}.yaml"))

        # Legacy split configs are emitted next to the generated main/ref/sim
        # files in the case output tree. Earlier v3 GUI code tried to resolve
        # these from an OpenBench install root (`nml/nml-yaml/stats.yaml` and
        # `figlib.yaml`), but those v2 files do not exist in wheel installs or
        # v3 source checkouts. Keeping them case-local makes the legacy split
        # layout self-contained for `openbench migrate` and old tooling.
        if is_remote:
            stats_nml_path = f"{nml_dir}/stats-{basename}.yaml"
            figure_nml_path = f"{nml_dir}/fig-{basename}.yaml"
        else:
            stats_nml_path = os.path.normpath(os.path.join(nml_dir, f"stats-{basename}.yaml"))
            figure_nml_path = os.path.normpath(os.path.join(nml_dir, f"fig-{basename}.yaml"))

        # For OpenBench, basedir should be the PARENT directory, not including basename
        # Because OpenBench computes output path as: basedir/basename
        if is_remote:
            parent_dir = "/".join(output_dir.rstrip("/").split("/")[:-1])
        else:
            parent_dir = os.path.dirname(output_dir.rstrip(os.sep))

        main_config["general"] = {
            "basename": basename,
            "basedir": parent_dir,
            "compare_tim_res": general.get("compare_tim_res", "month"),
            "compare_tzone": general.get("compare_tzone", 0.0),
            "compare_grid_res": general.get("compare_grid_res", 2.0),
            "syear": general.get("syear", 2000),
            "eyear": general.get("eyear", 2020),
            "min_year": general.get("min_year", 1.0),
            "max_lat": general.get("max_lat", 90.0),
            "min_lat": general.get("min_lat", -90.0),
            "max_lon": general.get("max_lon", 180.0),
            "min_lon": general.get("min_lon", -180.0),
            "reference_nml": ref_nml_path,
            "simulation_nml": sim_nml_path,
            "statistics_nml": stats_nml_path,
            "figure_nml": figure_nml_path,
            "num_cores": general.get("num_cores", 4),
            "evaluation": general.get("evaluation", True),
            "comparison": general.get("comparison", False),
            "statistics": general.get("statistics", False),
            "debug_mode": general.get("debug_mode", False),
            "only_drawing": general.get("only_drawing", False),
            "weight": "None" if str(general.get("weight", "none")).lower() == "none" else general.get("weight"),
            "IGBP_groupby": general.get("IGBP_groupby", True),
            "PFT_groupby": general.get("PFT_groupby", True),
            "Climate_zone_groupby": general.get("Climate_zone_groupby", True),
            "unified_mask": general.get("unified_mask", True),
            "generate_report": general.get("generate_report", True),
        }

        # Evaluation items
        main_config["evaluation_items"] = config.get("evaluation_items", {})

        # Metrics
        main_config["metrics"] = config.get("metrics", {})

        # Scores
        main_config["scores"] = config.get("scores", {})

        # Comparisons
        main_config["comparisons"] = config.get("comparisons", {})

        # Statistics
        main_config["statistics"] = config.get("statistics", {})

        return yaml.dump(main_config, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)

    def generate_stats_nml(self, config: Dict[str, Any]) -> str:
        """Generate the legacy statistics support YAML.

        The v3 runner builds its real statistics namelist from the unified
        ``openbench.yaml``. This file is only for the legacy split
        main/ref/sim layout that the GUI still exports for migration and
        third-party tooling. `openbench migrate` only needs the top-level
        method sections to preserve the selected statistics.
        """
        selected = [name for name, enabled in (config.get("statistics", {}) or {}).items() if bool(enabled)]
        stats_config: Dict[str, Any] = {"general": {}}
        for name in selected:
            stats_config["general"][f"{name}_data_source"] = ""
            stats_config[name] = {}
        return yaml.dump(stats_config, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)

    def generate_figure_nml(self) -> str:
        """Generate the legacy figure support YAML from bundled v3 resources."""
        from openbench.config.adapter import build_fig_nml

        return yaml.dump(build_fig_nml(), default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)

    def write_legacy_support_namelists(self, config: Dict[str, Any], nml_dir: str) -> Dict[str, str]:
        """Write legacy stats/figure support files into a case ``nml/`` dir.

        Returns:
            Mapping with ``statistics`` and ``figure`` file paths.
        """
        basename = config.get("general", {}).get("basename", "config")
        os.makedirs(nml_dir, exist_ok=True)

        stats_path = os.path.normpath(os.path.join(nml_dir, f"stats-{basename}.yaml"))
        figure_path = os.path.normpath(os.path.join(nml_dir, f"fig-{basename}.yaml"))

        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(self.generate_stats_nml(config))
        with open(figure_path, "w", encoding="utf-8") as f:
            f.write(self.generate_figure_nml())

        return {"statistics": stats_path, "figure": figure_path}

    def generate_ref_nml(
        self, config: Dict[str, Any], openbench_root: Optional[str] = None, output_dir: Optional[str] = None
    ) -> str:
        """
        Generate reference NML YAML content.

        Args:
            config: Full configuration dictionary
            openbench_root: OpenBench root directory for generating absolute paths
            output_dir: Output directory for local nml paths

        Returns:
            YAML string
        """
        import copy
        from openbench.gui.path_utils import remote_join

        ref_data = copy.deepcopy(config.get("ref_data", {}))

        # Check if in remote mode
        general = config.get("general", {})
        is_remote = general.get("execution_mode") == "remote"

        # Convert all paths to absolute (only in local mode)
        # In remote mode, paths are already remote paths and should not be converted
        if openbench_root is None:
            openbench_root = get_openbench_root()
        if not is_remote:
            ref_data = convert_paths_in_dict(ref_data, openbench_root)

        # Update def_nml paths to point to local copies
        if output_dir:
            if is_remote:
                # Use forward slashes for remote paths
                nml_dir = remote_join(output_dir, "nml", "ref")
                def_nml = ref_data.get("def_nml", {})
                for source_name in def_nml:
                    def_nml[source_name] = remote_join(nml_dir, f"{source_name}.yaml")
            else:
                nml_dir = os.path.join(output_dir, "nml", "ref")
                def_nml = ref_data.get("def_nml", {})
                for source_name in def_nml:
                    def_nml[source_name] = os.path.join(nml_dir, f"{source_name}.yaml")

        return yaml.dump(ref_data, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)

    def generate_sim_nml(
        self, config: Dict[str, Any], openbench_root: Optional[str] = None, output_dir: Optional[str] = None
    ) -> str:
        """
        Generate simulation NML YAML content.

        Args:
            config: Full configuration dictionary
            openbench_root: OpenBench root directory for generating absolute paths
            output_dir: Output directory for local nml paths

        Returns:
            YAML string
        """
        import copy
        from openbench.gui.path_utils import remote_join

        sim_data = copy.deepcopy(config.get("sim_data", {}))

        # Check if in remote mode
        general = config.get("general", {})
        is_remote = general.get("execution_mode") == "remote"

        # Convert all paths to absolute (only in local mode)
        # In remote mode, paths are already remote paths and should not be converted
        if openbench_root is None:
            openbench_root = get_openbench_root()
        if not is_remote:
            sim_data = convert_paths_in_dict(sim_data, openbench_root)

        # Update def_nml paths to point to local copies
        if output_dir:
            if is_remote:
                # Use forward slashes for remote paths
                nml_dir = remote_join(output_dir, "nml", "sim")
                def_nml = sim_data.get("def_nml", {})
                for source_name in def_nml:
                    def_nml[source_name] = remote_join(nml_dir, f"{source_name}.yaml")
            else:
                nml_dir = os.path.join(output_dir, "nml", "sim")
                def_nml = sim_data.get("def_nml", {})
                for source_name in def_nml:
                    def_nml[source_name] = os.path.join(nml_dir, f"{source_name}.yaml")

        return yaml.dump(sim_data, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)

    def validate(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration completeness.

        Args:
            config: Configuration dictionary

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        general = config.get("general", {})

        # Check required fields
        if not general.get("basename"):
            errors.append("Project name is required")

        if not general.get("basedir"):
            errors.append("Output directory is required")

        # Check year range
        syear = general.get("syear", 0)
        eyear = general.get("eyear", 0)
        if syear > eyear:
            errors.append("Start year must be less than or equal to end year")

        # Check evaluation items
        eval_items = config.get("evaluation_items", {})
        selected_items = [k for k, v in eval_items.items() if v]
        if not selected_items:
            errors.append("At least one evaluation item must be selected")

        # Check metrics
        metrics = config.get("metrics", {})
        selected_metrics = [k for k, v in metrics.items() if v]
        if not selected_metrics:
            errors.append("At least one metric must be selected")

        # Check ref data if any items selected
        if selected_items:
            ref_data = config.get("ref_data", {}).get("general", {})
            for item in selected_items:
                key = f"{item}_ref_source"
                if not ref_data.get(key):
                    errors.append(f"Reference data source required for {item}")

        # Check sim data if any items selected. The v3 CLI requires at
        # least one simulation entry with a non-empty root_dir/model. Catch
        # that at GUI validation time instead of exporting YAML that
        # `openbench run/check` will reject.
        sim_data = config.get("sim_data", {})
        sim_general = sim_data.get("general", {})
        sim_source_configs = sim_data.get("source_configs", {})
        seen_sim_sources = set()
        for item in selected_items:
            key = f"{item}_sim_source"
            sources = sim_general.get(key, [])
            if isinstance(sources, str):
                sources = [sources]
            sources = [s for s in sources if s]
            if not sources:
                errors.append(f"Simulation data source required for {item}")
                continue
            for source_name in sources:
                if source_name in seen_sim_sources:
                    continue
                seen_sim_sources.add(source_name)
                source_config = sim_source_configs.get(source_name)
                if not source_config:
                    errors.append(f"Simulation source '{source_name}' is missing configuration")
                    continue
                source_general = source_config.get("general", {}) if isinstance(source_config, dict) else {}
                if not (source_general.get("root_dir") or source_general.get("dir")):
                    errors.append(f"Simulation source '{source_name}' root directory is required")
                if not (source_general.get("model_namelist") or source_general.get("model")):
                    errors.append(f"Simulation source '{source_name}' model is required")

        if general.get("comparison"):
            selected_comparisons = [k for k, v in config.get("comparisons", {}).items() if v]
            if not selected_comparisons:
                errors.append("At least one comparison item must be selected when comparison is enabled")

        if general.get("statistics"):
            selected_statistics = [k for k, v in config.get("statistics", {}).items() if v]
            if not selected_statistics:
                errors.append("At least one statistics item must be selected when statistics is enabled")

        return errors

    def generate_config_yaml(
        self,
        config: Dict[str, Any],
        *,
        case_output_dir: Optional[str] = None,
        path_transform: Optional[Callable[[str], str]] = None,
    ) -> str:
        """Generate unified openbench.yaml content from internal GUI dict.

        Maps the legacy internal dict format to the new schema:
          general.*        → project.*
          evaluation_items → evaluation.variables
          ref_data         → reference (data_root + source names)
          sim_data         → simulation (_defaults + entries)
          metrics/scores   → metrics/scores (lists)
          comparisons      → comparison.items
          statistics       → statistics.items

        Args:
            config: Internal GUI configuration dictionary.
            case_output_dir: Optional full case output directory. When
                provided, ``project.output_dir`` is set to its parent so
                v3's ``project.output_dir / project.name`` convention lands
                on the requested case directory.
            path_transform: Optional callback used by remote export to
                rewrite user-entered paths into remote-server paths.

        Returns:
            YAML string in the new unified format.
        """
        general = config.get("general", {})

        def _case_parent(path: str) -> str:
            cleaned = str(path).rstrip("/\\")
            if "/" in cleaned and "\\" not in cleaned:
                parent = cleaned.rsplit("/", 1)[0]
                return parent or "/"
            return os.path.dirname(os.path.normpath(cleaned))

        def _maybe_transform_path(path: str) -> str:
            if not path or path_transform is None:
                return path
            return path_transform(path)

        # --- project ---
        project: Dict[str, Any] = {
            "name": general.get("basename", "config"),
            "output_dir": _case_parent(case_output_dir) if case_output_dir else general.get("basedir", "./output"),
            "years": [int(general.get("syear", 2000)), int(general.get("eyear", 2020))],
        }

        # Target resolution
        tim_res = general.get("compare_tim_res", "month")
        if tim_res:
            project["tim_res"] = tim_res
        grid_res = general.get("compare_grid_res", 2.0)
        if grid_res:
            project["grid_res"] = grid_res
        timezone = general.get("compare_tzone", 0.0)
        if timezone is not None:
            project["timezone"] = timezone
        weight = general.get("weight", "none")
        if weight is not None and str(weight).lower() != "":
            project["weight"] = "none" if str(weight).lower() == "none" else weight

        # Runtime
        num_cores = general.get("num_cores", 4)
        if num_cores:
            project["num_cores"] = int(num_cores)
        if not general.get("unified_mask", True):
            project["unified_mask"] = False
        if not general.get("generate_report", True):
            project["generate_report"] = False
        dask_config = general.get("dask")
        if isinstance(dask_config, dict) and dask_config.get("enabled"):
            project["dask"] = {k: v for k, v in dask_config.items() if v is not None}
        io_config = general.get("io")
        if isinstance(io_config, dict) and io_config:
            project["io"] = {k: v for k, v in io_config.items() if v is not None}

        # Spatial-temporal bounds
        min_year = general.get("min_year", 1.0)
        if min_year and float(min_year) != 3.0:
            project["min_year_threshold"] = float(min_year)
        lat_range = [general.get("min_lat", -90.0), general.get("max_lat", 90.0)]
        lon_range = [general.get("min_lon", -180.0), general.get("max_lon", 180.0)]
        if lat_range != [-90.0, 90.0]:
            project["lat_range"] = lat_range
        if lon_range != [-180.0, 180.0]:
            project["lon_range"] = lon_range

        # Groupby
        if general.get("IGBP_groupby", True):
            project["IGBP_groupby"] = True
        if general.get("PFT_groupby", True):
            project["PFT_groupby"] = True
        if general.get("Climate_zone_groupby", True):
            project["climate_zone_groupby"] = True

        # Time alignment strategy: forward the General Settings combo so the
        # exported YAML actually reflects per_pair / strict choices instead
        # of silently defaulting to intersection downstream.
        time_alignment = general.get("time_alignment")
        if time_alignment and time_alignment != "intersection":
            project["time_alignment"] = time_alignment

        # --- evaluation ---
        eval_items = config.get("evaluation_items", {})
        variables = [k for k, v in eval_items.items() if v]

        # --- reference ---
        reference: Dict[str, Any] = {}
        ref_data = config.get("ref_data", {})
        ref_general = ref_data.get("general", {})

        # data_root
        data_root = ref_general.get("data_root", "")
        if data_root:
            reference["data_root"] = _maybe_transform_path(data_root)

        for var in variables:
            source_key = f"{var}_ref_source"
            source = ref_general.get(source_key, "")
            if isinstance(source, list):
                cleaned = [s for s in source if s]
                if not cleaned:
                    continue
                # Single-element list → write a scalar so simple configs
                # stay simple; otherwise emit the list verbatim.
                reference[var] = cleaned[0] if len(cleaned) == 1 else cleaned
            elif source:
                reference[var] = source

        # --- simulation ---
        sim_data = config.get("sim_data", {})
        sim_general = sim_data.get("general", {})
        sim_def_nml = sim_data.get("def_nml", {})
        sim_source_configs = sim_data.get("source_configs", {})

        # Collect all sim source names
        all_sources: List[str] = []
        for var in variables:
            source_key = f"{var}_sim_source"
            sources = sim_general.get(source_key, [])
            if isinstance(sources, str):
                sources = [sources]
            for s in sources:
                if s and s not in all_sources:
                    all_sources.append(s)

        # Build simulation entries from source configs or def_nml
        sim_entries: Dict[str, Dict[str, Any]] = {}
        for source_name in all_sources:
            src_cfg = sim_source_configs.get(source_name, {})
            src_general = src_cfg.get("general", {}) if src_cfg else {}

            # Try to read from def_nml file if no source_config
            if not src_general and source_name in sim_def_nml:
                def_path = sim_def_nml[source_name]
                if os.path.exists(def_path):
                    try:
                        with open(def_path) as f:
                            def_data = yaml.safe_load(f) or {}
                        src_general = def_data.get("general", {})
                    except Exception:
                        pass

            # Detect model
            model_nml = src_general.get("model_namelist", "") or src_general.get("model", "")
            model = self._detect_model_from_path(model_nml) if model_nml else "unknown"

            entry: Dict[str, Any] = {
                "model": model,
                "root_dir": _maybe_transform_path(src_general.get("root_dir", src_general.get("dir", ""))),
            }
            if src_general.get("data_type"):
                entry["data_type"] = src_general["data_type"]
            if src_general.get("grid_res"):
                entry["grid_res"] = src_general["grid_res"]
            if src_general.get("tim_res"):
                entry["tim_res"] = src_general["tim_res"]
            if src_general.get("data_groupby"):
                entry["data_groupby"] = src_general["data_groupby"]
            if src_general.get("prefix"):
                entry["prefix"] = src_general["prefix"]
            if src_general.get("suffix"):
                entry["suffix"] = src_general["suffix"]
            if src_general.get("fulllist"):
                entry["fulllist"] = _maybe_transform_path(src_general["fulllist"])
            if isinstance(src_cfg.get("variables"), dict) and src_cfg["variables"]:
                entry["variables"] = src_cfg["variables"]

            # Use source_name as label, or derive a cleaner one
            label = self._derive_label(source_name, entry.get("root_dir", ""))
            sim_entries[label] = entry

        simulation = self._extract_sim_defaults(sim_entries)

        # --- metrics / scores ---
        metrics_dict = config.get("metrics", {})
        metrics_list = [k for k, v in metrics_dict.items() if v]

        scores_dict = config.get("scores", {})
        scores_list = [k for k, v in scores_dict.items() if v]

        # --- comparison ---
        comparison: Optional[Dict[str, Any]] = None
        if general.get("comparison"):
            comp_items = [k for k, v in config.get("comparisons", {}).items() if v]
            if comp_items:
                comparison = {"enabled": True, "items": comp_items}

        # --- statistics ---
        stats_section: Optional[Dict[str, Any]] = None
        if general.get("statistics"):
            stat_items = [k for k, v in config.get("statistics", {}).items() if v]
            if stat_items:
                stats_section = {"enabled": True, "items": stat_items}

        # --- assemble ---
        output: Dict[str, Any] = {
            "project": project,
            "evaluation": {"variables": variables},
            "reference": reference,
            "simulation": simulation,
        }
        if metrics_list:
            output["metrics"] = metrics_list
        if scores_list:
            output["scores"] = scores_list
        if comparison:
            output["comparison"] = comparison
        if stats_section:
            output["statistics"] = stats_section

        return yaml.dump(output, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)

    def _detect_model_from_path(self, model_nml_path: str) -> str:
        """Detect model name from model_namelist path.

        Accepts both legacy paths (e.g. './nml/Mod_variables_definition/
        CoLM.nml') and bare canonical model names (e.g. 'CoLM2024')
        produced by the new scan-based PageSimData. The previous code
        always lowercased the stem and looked up an aliasing map; bare
        names not in the map were returned in lowercase form, which then
        failed to match the case-sensitive registry. Preserve the original
        stem case so v3 configs carry the canonical name unchanged.
        """
        if not model_nml_path:
            return "unknown"
        stem = Path(model_nml_path).stem
        model_map = {
            "colm": "CoLM2024",
            "clm": "CLM5",
            "clm5": "CLM5",
            "noah": "NOAH",
            "noahmp5": "NoahMP5",
            "cama": "CaMa",
            "camaflood": "CaMa",
            "cama-flood": "CaMa",
            "gldas": "GLDAS",
            "gldas2": "GLDAS",
            "era5land": "ERA5-Land",
            "era5-land": "ERA5-Land",
            "te": "TE",
            "jules7": "JULES7",
            "vic5": "VIC5",
        }
        return model_map.get(stem.lower(), stem)

    def _derive_label(self, source_name: str, root_dir: str) -> str:
        """Derive a clean case label from source name or path."""
        import re

        match = re.search(r"(Case\d+)", root_dir)
        if match:
            return match.group(1)
        match = re.search(r"(\d+)_case", source_name)
        if match:
            return f"Case{match.group(1).zfill(2)}"
        return source_name

    def _extract_sim_defaults(self, entries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract _defaults from simulation entries if fields are shared."""
        if len(entries) < 2:
            return entries

        common_keys = ["model", "data_type", "grid_res", "tim_res", "data_groupby"]
        defaults: Dict[str, Any] = {}
        first = next(iter(entries.values()))
        for key in common_keys:
            val = first.get(key)
            if val is not None and all(e.get(key) == val for e in entries.values()):
                defaults[key] = val

        if not defaults:
            return entries

        result: Dict[str, Any] = {"_defaults": defaults}
        for label, entry in entries.items():
            cleaned = {k: v for k, v in entry.items() if k not in defaults or defaults[k] != v}
            result[label] = cleaned
        return result

    def export_all(
        self,
        config: Dict[str, Any],
        output_dir: str,
        basename: Optional[str] = None,
        openbench_root: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Export all NML files to directory.

        Args:
            config: Configuration dictionary
            output_dir: Output directory path
            basename: Base name for files (defaults to config basename)
            openbench_root: OpenBench root directory for path conversion

        Returns:
            Dictionary of {file_type: file_path}
        """
        if basename is None:
            basename = config.get("general", {}).get("basename", "config")

        if openbench_root is None:
            openbench_root = get_openbench_root()

        os.makedirs(output_dir, exist_ok=True)

        files = {}

        # Generate unified openbench.yaml
        config_path = os.path.join(output_dir, "openbench.yaml")
        config_content = self.generate_config_yaml(config, case_output_dir=output_dir)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)
        files["config"] = config_path

        return files

    def sync_namelists(
        self, config: Dict[str, Any], output_dir: str, openbench_root: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Sync namelists to output directory's nml/ subdirectory.

        Args:
            config: Configuration dictionary
            output_dir: Output directory path
            openbench_root: OpenBench root directory

        Returns:
            Dictionary of {source_name: copied_path}
        """
        if openbench_root is None:
            openbench_root = get_openbench_root()

        # Create nml subdirectories
        nml_dir = os.path.join(output_dir, "nml")
        sim_nml_dir = os.path.join(nml_dir, "sim")
        ref_nml_dir = os.path.join(nml_dir, "ref")
        os.makedirs(sim_nml_dir, exist_ok=True)
        os.makedirs(ref_nml_dir, exist_ok=True)

        # Get selected evaluation items
        eval_items = config.get("evaluation_items", {})
        selected_items = [k for k, v in eval_items.items() if v]

        copied_files = {}

        # Process simulation data namelists
        sim_data = config.get("sim_data", {})
        sim_def_nml = sim_data.get("def_nml", {})
        sim_source_configs = sim_data.get("source_configs", {})  # Get edited configs
        sim_copied, sim_model_files = self._copy_data_namelists(
            sim_def_nml, sim_nml_dir, selected_items, openbench_root, "sim", sim_source_configs
        )
        copied_files.update(sim_copied)

        # Copy model definition files for sim (into models/ subdirectory)
        sim_models_dir = os.path.join(sim_nml_dir, "models")

        for model_path in sim_model_files:
            if model_path:
                # Try to find the actual file (handle .nml -> .yaml conversion)
                actual_path = self._resolve_model_path(model_path)
                if actual_path and os.path.exists(actual_path):
                    # Always use .yaml extension for output
                    model_name = os.path.splitext(os.path.basename(actual_path))[0] + ".yaml"
                    dest_path = os.path.join(sim_models_dir, model_name)
                    self._copy_model_definition(actual_path, dest_path, selected_items)
                    copied_files[f"model_{model_name}"] = dest_path
                    continue
                # Bare registry model names (scan-based assignment) have no
                # definition file on disk — generate one from the registry.
                registry_content = model_definition_from_registry(model_path, selected_items)
                if registry_content is not None:
                    model_name = f"{model_path}.yaml"
                    dest_path = os.path.join(sim_models_dir, model_name)
                    os.makedirs(sim_models_dir, exist_ok=True)
                    with open(dest_path, "w", encoding="utf-8") as f:
                        yaml.dump(
                            registry_content, f, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2
                        )
                    copied_files[f"model_{model_name}"] = dest_path

        # Process reference data namelists
        ref_data = config.get("ref_data", {})
        ref_def_nml = ref_data.get("def_nml", {})
        ref_source_configs = ref_data.get("source_configs", {})  # Get edited configs
        ref_copied, _ = self._copy_data_namelists(
            ref_def_nml, ref_nml_dir, selected_items, openbench_root, "ref", ref_source_configs
        )
        copied_files.update(ref_copied)

        # Note: Don't update config paths here - keep original paths in config
        # The exported YAML files handle path conversion separately

        return copied_files

    def _copy_data_namelists(
        self,
        def_nml: Dict[str, str],
        dest_dir: str,
        selected_items: List[str],
        openbench_root: str,
        data_type: str,
        source_configs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, str], Set[str]]:
        """
        Copy data source namelists with filtering.

        Args:
            def_nml: Dictionary of {source_name: nml_path}
            dest_dir: Destination directory
            selected_items: List of selected evaluation items
            openbench_root: OpenBench root directory
            data_type: "sim" or "ref"
            source_configs: Optional dictionary of edited source configurations
                           For ref data, keys are compound: "var_name::source_name"

        Returns:
            Tuple of (copied_files dict, model_definition_paths set)
        """
        copied_files = {}
        model_files = set()

        if source_configs is None:
            source_configs = {}

        # For ref data, reorganize source_configs by source_name with per-variable configs
        # source_configs uses compound key "var_name::source_name"
        organized_configs = {}  # {source_name: {"_general": {...}, var_name: {...}}}

        for key, config in source_configs.items():
            if "::" in key:
                # Compound key format: "var_name::source_name"
                var_name, source_name = key.split("::", 1)
                if source_name not in organized_configs:
                    organized_configs[source_name] = {}
                # Store general section (shared) - but use the first one, don't overwrite
                if "general" in config and "_general" not in organized_configs[source_name]:
                    organized_configs[source_name]["_general"] = config["general"].copy()
                # Store var-specific config - copy all fields except internal ones
                var_config = {}
                skip_fields = {"general", "_var_name", "def_nml_path"}
                for field, value in config.items():
                    if field not in skip_fields:
                        var_config[field] = value

                # Store per-variable time range settings for this specific variable
                general = config.get("general", {})
                if general.get("per_var_time_range"):
                    var_config["per_var_time_range"] = True
                    if "syear" in general and general["syear"] != "":
                        var_config["syear"] = general["syear"]
                    if "eyear" in general and general["eyear"] != "":
                        var_config["eyear"] = general["eyear"]

                if var_config:
                    organized_configs[source_name][var_name] = var_config
            else:
                # Legacy format - source_name directly
                organized_configs[key] = {"_legacy": config}

        for source_name, nml_path in def_nml.items():
            dest_path = os.path.join(dest_dir, f"{source_name}.yaml")

            # Check if we have edited source config - use it instead of copying from file
            if source_name in organized_configs:
                model_path = self._write_source_config_organized(
                    organized_configs[source_name], dest_path, selected_items, openbench_root, data_type
                )
                copied_files[source_name] = dest_path
                if model_path:
                    model_files.add(model_path)
                continue

            # No edited config - copy from original file
            if not nml_path:
                continue

            # Resolve to absolute path
            src_path = to_absolute_path(nml_path, openbench_root)

            # Try YAML path if .nml doesn't exist
            if not os.path.exists(src_path):
                yaml_path = src_path.replace("nml-Fortran", "nml-yaml").replace(".nml", ".yaml")
                if os.path.exists(yaml_path):
                    src_path = yaml_path

            if not os.path.exists(src_path):
                continue

            # Copy with filtering
            model_path = self._copy_namelist_filtered(src_path, dest_path, selected_items, openbench_root)
            copied_files[source_name] = dest_path

            if model_path:
                model_files.add(model_path)

        return copied_files, model_files

    def _write_source_config(
        self, source_data: Dict[str, Any], dest_path: str, selected_items: List[str], openbench_root: str
    ) -> Optional[str]:
        """
        Write edited source configuration to a namelist file.
        (Legacy method - kept for backward compatibility)

        Args:
            source_data: The edited source configuration data
            dest_path: Destination file path
            selected_items: List of selected evaluation items
            openbench_root: OpenBench root directory

        Returns:
            Model definition path if found, None otherwise
        """
        # Build the output structure
        filtered = {}
        model_path = None

        # Check for general section
        if "general" in source_data:
            general = source_data["general"].copy()

            # Convert paths to absolute
            if "root_dir" in general and general["root_dir"]:
                general["root_dir"] = to_absolute_path(general["root_dir"], openbench_root)
            if "dir" in general and general["dir"]:
                general["dir"] = to_absolute_path(general["dir"], openbench_root)
            if "fulllist" in general and general["fulllist"]:
                general["fulllist"] = to_absolute_path(general["fulllist"], openbench_root)
            if "model_namelist" in general and general["model_namelist"]:
                model_path = to_absolute_path(general["model_namelist"], openbench_root)
                # Update to point to models subdirectory to avoid conflicts with case files
                model_basename = os.path.splitext(os.path.basename(model_path))[0]
                dest_dir = os.path.dirname(dest_path)
                models_dir = os.path.join(dest_dir, "models")
                general["model_namelist"] = os.path.join(models_dir, model_basename + ".yaml")

            filtered["general"] = general

        # Extract legacy top-level variable mapping fields.
        var_mapping_keys = ["sub_dir", "varname", "varunit", "prefix", "suffix"]
        top_level_var_mapping = {}
        for key in var_mapping_keys:
            if key in source_data:
                top_level_var_mapping[key] = source_data[key]

        # Include selected evaluation items that exist in source_data (with type validation)
        # Don't create entries for variables that don't exist in the source
        for item in selected_items:
            if item in source_data:
                # Item exists in source data
                item_data = source_data[item]
                if isinstance(item_data, dict):
                    item_copy = item_data.copy()
                    # Add top-level var mapping fields if not already present
                    for key, value in top_level_var_mapping.items():
                        if key not in item_copy:
                            item_copy[key] = value
                    filtered[item] = item_copy
                elif item_data is not None:
                    filtered[item] = item_data

        # Write the file
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            yaml.dump(filtered, f, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)

        return model_path

    def _write_source_config_organized(
        self,
        organized_data: Dict[str, Any],
        dest_path: str,
        selected_items: List[str],
        openbench_root: str,
        data_type: str,
    ) -> Optional[str]:
        """
        Write organized source configuration to a namelist file.
        Handles per-variable configurations properly.

        Args:
            organized_data: Dictionary with structure:
                           {"_general": {...}, "var_name1": {...}, "var_name2": {...}}
                           or {"_legacy": {...}} for old format
            dest_path: Destination file path
            selected_items: List of selected evaluation items
            openbench_root: OpenBench root directory
            data_type: "sim" or "ref"

        Returns:
            Model definition path if found, None otherwise
        """
        # Handle legacy format
        if "_legacy" in organized_data:
            return self._write_source_config(organized_data["_legacy"], dest_path, selected_items, openbench_root)

        filtered = {}
        model_path = None

        # Process general section
        if "_general" in organized_data:
            general = organized_data["_general"].copy()

            # Convert paths to absolute
            if "root_dir" in general and general["root_dir"]:
                general["root_dir"] = to_absolute_path(general["root_dir"], openbench_root)
            if "dir" in general and general["dir"]:
                general["dir"] = to_absolute_path(general["dir"], openbench_root)
            if "fulllist" in general and general["fulllist"]:
                general["fulllist"] = to_absolute_path(general["fulllist"], openbench_root)
            if "model_namelist" in general and general["model_namelist"]:
                model_path = to_absolute_path(general["model_namelist"], openbench_root)
                # Update to point to models subdirectory to avoid conflicts with case files
                model_basename = os.path.splitext(os.path.basename(model_path))[0]
                dest_dir = os.path.dirname(dest_path)
                models_dir = os.path.join(dest_dir, "models")
                general["model_namelist"] = os.path.join(models_dir, model_basename + ".yaml")

            # Remove UI-only field from output
            general.pop("per_var_time_range", None)

            # Check if any variable has per_var_time_range enabled
            any_per_var = any(organized_data.get(item, {}).get("per_var_time_range", False) for item in selected_items)

            # If any variable uses per-variable time range, remove from general
            if any_per_var:
                general.pop("syear", None)
                general.pop("eyear", None)

            filtered["general"] = general

        # Process per-variable configurations (with type validation)
        for item in selected_items:
            if item in organized_data:
                item_data = organized_data[item]
                if isinstance(item_data, dict):
                    # Use the specific config for this variable
                    var_config = item_data.copy()

                    # Remove per_var_time_range from output (it's only for UI control)
                    var_config.pop("per_var_time_range", None)

                    if var_config:  # Only add if there's data
                        filtered[item] = var_config
                elif item_data is not None:
                    filtered[item] = item_data

        # Write the file
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            yaml.dump(filtered, f, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)

        return model_path

    def _copy_namelist_filtered(
        self, src_path: str, dest_path: str, selected_items: List[str], openbench_root: str
    ) -> Optional[str]:
        """
        Copy a namelist file with smart filtering.

        Only keeps general section and selected evaluation items.
        Converts all paths to absolute.

        Args:
            src_path: Source file path
            dest_path: Destination file path
            selected_items: List of selected evaluation items
            openbench_root: OpenBench root directory

        Returns:
            Model definition path if found, None otherwise
        """
        try:
            with open(src_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f) or {}
        except Exception:
            return None

        # Build filtered content
        filtered = {}
        model_path = None

        # Always keep general section
        if "general" in content:
            general = content["general"].copy()

            # Convert paths to absolute
            if "root_dir" in general and general["root_dir"]:
                general["root_dir"] = to_absolute_path(general["root_dir"], openbench_root)
            if "dir" in general and general["dir"]:
                general["dir"] = to_absolute_path(general["dir"], openbench_root)
            if "fulllist" in general and general["fulllist"]:
                general["fulllist"] = to_absolute_path(general["fulllist"], openbench_root)
            if "model_namelist" in general and general["model_namelist"]:
                model_path = to_absolute_path(general["model_namelist"], openbench_root)
                # Update to point to models subdirectory to avoid conflicts with case files
                model_basename = os.path.splitext(os.path.basename(model_path))[0]
                dest_dir = os.path.dirname(dest_path)
                models_dir = os.path.join(dest_dir, "models")
                general["model_namelist"] = os.path.join(models_dir, model_basename + ".yaml")

            filtered["general"] = general

        # Keep only selected evaluation items (with type validation)
        for item in selected_items:
            if item in content:
                item_data = content[item]
                if isinstance(item_data, dict):
                    # sub_dir doesn't need path conversion (it's relative to root_dir)
                    filtered[item] = item_data.copy()
                elif item_data is not None:
                    filtered[item] = item_data

        # Write filtered content
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            yaml.dump(filtered, f, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)

        return model_path

    def _copy_model_definition(self, src_path: str, dest_path: str, selected_items: List[str]):
        """
        Copy a model definition file with filtering.

        Only keeps general section and selected evaluation items.

        Args:
            src_path: Source file path
            dest_path: Destination file path
            selected_items: List of selected evaluation items
        """
        try:
            with open(src_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f) or {}
        except Exception:
            return

        # Build filtered content
        filtered = {}

        # Always keep general section (with type validation)
        if "general" in content and isinstance(content["general"], dict):
            filtered["general"] = content["general"].copy()

        # Keep only selected evaluation items (with type validation)
        for item in selected_items:
            if item in content:
                item_data = content[item]
                if isinstance(item_data, dict):
                    filtered[item] = item_data.copy()
                elif item_data is not None:
                    filtered[item] = item_data

        # Write filtered content
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            yaml.dump(filtered, f, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)

    def _resolve_model_path(self, model_path: str) -> Optional[str]:
        """
        Resolve a model path, handling only explicit paths and adjacent
        .nml -> .yaml compatibility.

        Args:
            model_path: Path to model definition file (may be .nml or .yaml)

        Returns:
            Actual path to the file if found, None otherwise
        """
        if os.path.exists(model_path):
            return model_path

        # Keep the one compatibility case that is independent of install
        # layout: a legacy .nml reference may point at a side-by-side .yaml.
        if model_path.endswith(".nml"):
            yaml_path = model_path[:-4] + ".yaml"
            if os.path.exists(yaml_path):
                return yaml_path

        return None

    def _update_config_nml_paths(self, config: Dict[str, Any], output_dir: str):
        """
        Update config def_nml paths to point to local nml/ directory.

        Args:
            config: Configuration dictionary (modified in place)
            output_dir: Output directory path
        """
        nml_dir = os.path.join(output_dir, "nml")

        # Update sim_data def_nml paths
        sim_data = config.get("sim_data", {})
        sim_def_nml = sim_data.get("def_nml", {})
        for source_name in sim_def_nml:
            sim_def_nml[source_name] = os.path.join(nml_dir, "sim", f"{source_name}.yaml")

        # Update ref_data def_nml paths
        ref_data = config.get("ref_data", {})
        ref_def_nml = ref_data.get("def_nml", {})
        for source_name in ref_def_nml:
            ref_def_nml[source_name] = os.path.join(nml_dir, "ref", f"{source_name}.yaml")

    def _find_openbench_install_root(self, openbench_root: Optional[str] = None) -> Optional[str]:
        """
        Find the actual OpenBench installation directory.

        This is distinct from the project output directory. In v3 the
        installation directory contains pyproject.toml plus
        src/openbench/cli/main.py (editable layout). The pre-v3 markers
        (openbench/openbench.py, nml/nml-yaml/stats.yaml) no longer
        exist and are not checked.

        Args:
            openbench_root: A potential OpenBench root (may be output dir)

        Returns:
            Path to OpenBench installation directory, or None if not found
        """
        # Check if provided openbench_root is actually the installation
        if openbench_root and self._is_openbench_installation(openbench_root):
            return openbench_root

        env_root = os.environ.get("OPENBENCH_ROOT")
        if env_root and self._is_openbench_installation(env_root):
            return env_root

        detected_root = get_openbench_root()
        if detected_root and self._is_openbench_installation(detected_root):
            return detected_root

        # Search common locations
        search_paths = [
            os.path.expanduser("~/Desktop/OpenBench"),
            os.path.expanduser("~/Documents/OpenBench"),
            os.path.expanduser("~/OpenBench"),
            "/opt/OpenBench",
            "/usr/local/OpenBench",
        ]

        for path in search_paths:
            if self._is_openbench_installation(path):
                return path

        return None

    def _is_openbench_installation(self, path: str) -> bool:
        """
        Check if a path is a valid OpenBench v3 installation directory.

        Delegates to :func:`openbench.gui.path_utils.looks_like_openbench_root`
        so the GUI's three places that have to validate this (this
        method, page_runtime browse dialog, page_run_monitor manual
        select) all use one definition of "valid".
        """
        from openbench.gui.path_utils import looks_like_openbench_root

        return looks_like_openbench_root(path)

    def cleanup_unused_namelists(self, config: Dict[str, Any], output_dir: str):
        """
        Remove namelist files that are no longer used.

        Args:
            config: Configuration dictionary
            output_dir: Output directory path
        """
        nml_dir = os.path.join(output_dir, "nml")
        if not os.path.exists(nml_dir):
            return

        # Get currently used sources
        sim_sources = set(config.get("sim_data", {}).get("def_nml", {}).keys())
        ref_sources = set(config.get("ref_data", {}).get("def_nml", {}).keys())

        # Clean sim directory
        sim_nml_dir = os.path.join(nml_dir, "sim")
        if os.path.exists(sim_nml_dir):
            for filename in os.listdir(sim_nml_dir):
                if filename.endswith(".yaml"):
                    source_name = filename[:-5]  # Remove .yaml
                    if source_name not in sim_sources and not source_name.startswith("model_"):
                        os.remove(os.path.join(sim_nml_dir, filename))

        # Clean ref directory
        ref_nml_dir = os.path.join(nml_dir, "ref")
        if os.path.exists(ref_nml_dir):
            for filename in os.listdir(ref_nml_dir):
                if filename.endswith(".yaml"):
                    source_name = filename[:-5]  # Remove .yaml
                    if source_name not in ref_sources:
                        os.remove(os.path.join(ref_nml_dir, filename))

    def _has_per_var_time_range(self, config: Dict[str, Any]) -> bool:
        """
        Check if any source has per_var_time_range enabled.

        Args:
            config: Configuration dictionary

        Returns:
            True if any source has per_var_time_range enabled
        """
        # Check ref_data source_configs
        ref_source_configs = config.get("ref_data", {}).get("source_configs", {})
        for source_config in ref_source_configs.values():
            general = source_config.get("general", {})
            if general.get("per_var_time_range", False):
                return True

        # Check sim_data source_configs
        sim_source_configs = config.get("sim_data", {}).get("source_configs", {})
        for source_config in sim_source_configs.values():
            general = source_config.get("general", {})
            if general.get("per_var_time_range", False):
                return True

        return False

# -*- coding: utf-8 -*-
"""
Configuration manager for loading, saving, and validating NML configs.
"""

import os
import shutil
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path

import yaml

from openbench.gui.path_utils import convert_paths_in_dict, get_openbench_root, to_absolute_path


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

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)

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
            remote_openbench_path: Remote OpenBench installation path (for remote mode)

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

        # stats.yaml and figlib.yaml are always in the OpenBench installation directory
        # NOT in the project output directory - find them reliably
        if is_remote and remote_openbench_path:
            # Use remote OpenBench path with forward slashes
            remote_ob = remote_openbench_path.rstrip("/").replace("\\", "/")
            stats_nml_path = f"{remote_ob}/nml/nml-yaml/stats.yaml"
            figure_nml_path = f"{remote_ob}/nml/nml-yaml/figlib.yaml"
        else:
            install_root = self._find_openbench_install_root(openbench_root)
            if install_root:
                stats_nml_path = os.path.join(install_root, "nml", "nml-yaml", "stats.yaml")
                figure_nml_path = os.path.join(install_root, "nml", "nml-yaml", "figlib.yaml")
            else:
                # Fallback to relative paths if not found
                stats_nml_path = "./nml/nml-yaml/stats.yaml"
                figure_nml_path = "./nml/nml-yaml/figlib.yaml"

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
            "weight": "None" if general.get("weight", "none").lower() == "none" else general.get("weight"),
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

        return errors

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

        # Create output directory and nml subdirectory
        nml_dir = os.path.join(output_dir, "nml")
        os.makedirs(nml_dir, exist_ok=True)

        files = {}

        # Main NML - goes in nml folder
        main_path = os.path.join(nml_dir, f"main-{basename}.yaml")
        main_content = self.generate_main_nml(config, openbench_root, output_dir)
        with open(main_path, "w", encoding="utf-8") as f:
            f.write(main_content)
        files["main"] = main_path

        # Ref NML - goes in nml folder
        ref_path = os.path.join(nml_dir, f"ref-{basename}.yaml")
        ref_content = self.generate_ref_nml(config, openbench_root, output_dir)
        with open(ref_path, "w", encoding="utf-8") as f:
            f.write(ref_content)
        files["ref"] = ref_path

        # Sim NML - goes in nml folder
        sim_path = os.path.join(nml_dir, f"sim-{basename}.yaml")
        sim_content = self.generate_sim_nml(config, openbench_root, output_dir)
        with open(sim_path, "w", encoding="utf-8") as f:
            f.write(sim_content)
        files["sim"] = sim_path

        # Sync namelists to nml/sim and nml/ref subdirectories
        self.sync_namelists(config, output_dir, openbench_root)

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

        # Debug logging
        import tempfile

        debug_log = os.path.join(tempfile.gettempdir(), "openbench_wizard_debug.log")
        with open(debug_log, "a") as f:
            f.write(f"\n=== Local mode: Copy model files ===\n")
            f.write(f"sim_model_files: {sim_model_files}\n")
            f.write(f"sim_models_dir: {sim_models_dir}\n")

        for model_path in sim_model_files:
            if model_path:
                # Try to find the actual file (handle .nml -> .yaml conversion)
                actual_path = self._resolve_model_path(model_path)
                with open(debug_log, "a") as f:
                    f.write(
                        f"model_path={model_path}, actual_path={actual_path}, exists={os.path.exists(actual_path) if actual_path else False}\n"
                    )
                if actual_path and os.path.exists(actual_path):
                    # Always use .yaml extension for output
                    model_name = os.path.splitext(os.path.basename(actual_path))[0] + ".yaml"
                    dest_path = os.path.join(sim_models_dir, model_name)
                    self._copy_model_definition(actual_path, dest_path, selected_items)
                    copied_files[f"model_{model_name}"] = dest_path
                    with open(debug_log, "a") as f:
                        f.write(f"  copied to {dest_path}\n")

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

        # Extract variable mapping fields from top level (from DataSourceEditor)
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
        Resolve a model path, handling .nml -> .yaml conversion and path variations.

        Args:
            model_path: Path to model definition file (may be .nml or .yaml)

        Returns:
            Actual path to the file if found, None otherwise
        """
        if os.path.exists(model_path):
            return model_path

        # Try converting .nml to .yaml
        if model_path.endswith(".nml"):
            yaml_path = model_path[:-4] + ".yaml"
            if os.path.exists(yaml_path):
                return yaml_path

            # Try with nml-yaml directory structure
            # e.g., /path/nml/Mod_variables_definition/CoLM.nml
            # -> /path/nml/nml-yaml/Mod_variables_definition/CoLM.yaml
            yaml_path = model_path.replace("/nml/", "/nml/nml-yaml/").replace(".nml", ".yaml")
            if os.path.exists(yaml_path):
                return yaml_path

        # Try adding nml-yaml to path for yaml files too
        if "/nml/" in model_path and "/nml-yaml/" not in model_path:
            yaml_path = model_path.replace("/nml/", "/nml/nml-yaml/")
            if os.path.exists(yaml_path):
                return yaml_path

        # If path points to output directory but file doesn't exist,
        # search for the model file by name in standard locations
        model_name = os.path.basename(model_path)
        model_basename = os.path.splitext(model_name)[0]
        openbench_root = get_openbench_root()

        # Try common model definition locations
        search_paths = [
            os.path.join(openbench_root, "nml", "nml-yaml", "Mod_variables_definition", f"{model_basename}.yaml"),
            os.path.join(openbench_root, "nml", "nml-yaml", "Mod_variables_definition", f"{model_basename}.nml"),
            os.path.join(openbench_root, "nml", "Mod_variables_definition", f"{model_basename}.yaml"),
            os.path.join(openbench_root, "nml", "Mod_variables_definition", f"{model_basename}.nml"),
        ]

        for search_path in search_paths:
            if os.path.exists(search_path):
                return search_path

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

        This is distinct from the project output directory. The installation
        directory contains openbench/openbench.py and nml/nml-yaml/stats.yaml.

        Args:
            openbench_root: A potential OpenBench root (may be output dir)

        Returns:
            Path to OpenBench installation directory, or None if not found
        """
        # First, try using the wizard's own location
        # The wizard is at OpenBench/openbench_wizard/
        wizard_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        potential_root = os.path.dirname(wizard_dir)

        # Check if this looks like an OpenBench installation
        if self._is_openbench_installation(potential_root):
            return potential_root

        # Check if provided openbench_root is actually the installation
        if openbench_root and self._is_openbench_installation(openbench_root):
            return openbench_root

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

        # Try to find from environment variable
        env_root = os.environ.get("OPENBENCH_ROOT")
        if env_root and self._is_openbench_installation(env_root):
            return env_root

        return None

    def _is_openbench_installation(self, path: str) -> bool:
        """
        Check if a path is a valid OpenBench installation directory.

        Args:
            path: Path to check

        Returns:
            True if this is a valid OpenBench installation
        """
        if not path or not os.path.isdir(path):
            return False

        # Check for key files that indicate an OpenBench installation
        openbench_py = os.path.join(path, "openbench", "openbench.py")
        stats_yaml = os.path.join(path, "nml", "nml-yaml", "stats.yaml")

        return os.path.exists(openbench_py) or os.path.exists(stats_yaml)

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

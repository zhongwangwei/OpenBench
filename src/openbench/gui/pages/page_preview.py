# -*- coding: utf-8 -*-
"""
Preview and Export page.
"""

import logging
import os
import tempfile

from PySide6.QtWidgets import QVBoxLayout, QHBoxLayout, QTabWidget, QWidget, QPushButton, QLabel, QMessageBox
from PySide6.QtCore import Signal

from openbench.gui.pages.base_page import BasePage
from openbench.gui.widgets import YamlPreview
from openbench.gui.config_manager import ConfigManager
# get_openbench_root is now inherited from BasePage via _get_openbench_root()

logger = logging.getLogger(__name__)


def get_remote_ssh_manager(controller):
    """Get SSH manager from the controller if in remote mode.

    Args:
        controller: The WizardController instance

    Returns:
        SSHManager instance if in remote mode and connected, None otherwise
    """
    # Check storage type to determine if in remote mode
    from openbench.remote.storage import RemoteStorage

    if not isinstance(controller.storage, RemoteStorage):
        return None
    return controller.ssh_manager


class PagePreview(BasePage):
    """Preview and Export page."""

    PAGE_ID = "preview"
    PAGE_TITLE = "Preview & Export"
    PAGE_SUBTITLE = "Review generated configuration and export files"

    run_requested = Signal(str)  # Emits output directory

    def _setup_content(self):
        """Setup page content."""
        self.config_manager = ConfigManager()

        # Tab widget for different files
        self.tab_widget = QTabWidget()

        # Main NML preview
        self.main_preview = YamlPreview()
        self.tab_widget.addTab(self.main_preview, "main.yaml")

        # Ref NML preview
        self.ref_preview = YamlPreview()
        self.tab_widget.addTab(self.ref_preview, "ref.yaml")

        # Sim NML preview
        self.sim_preview = YamlPreview()
        self.tab_widget.addTab(self.sim_preview, "sim.yaml")

        self.content_layout.addWidget(self.tab_widget, 1)

        # Output directory info
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_label = QLabel("")
        self.output_dir_label.setStyleSheet("color: #666; font-style: italic;")
        info_layout.addWidget(self.output_dir_label, 1)
        self.content_layout.addLayout(info_layout)

    def load_from_config(self):
        """Load and generate previews."""
        import os

        config = self.controller.config

        # Update output directory display
        output_dir = self.controller.get_output_dir()
        self.output_dir_label.setText(output_dir)

        # Generate previews with absolute paths
        openbench_root = self._get_openbench_root()
        main_yaml = self.config_manager.generate_main_nml(config, openbench_root, output_dir)
        self.main_preview.set_content(main_yaml)

        ref_yaml = self.config_manager.generate_ref_nml(config, openbench_root, output_dir)
        self.ref_preview.set_content(ref_yaml)

        sim_yaml = self.config_manager.generate_sim_nml(config, openbench_root, output_dir)
        self.sim_preview.set_content(sim_yaml)

    def export_and_run(self) -> bool:
        """Export files and trigger run. Returns True if successful."""
        # Use the controller's output directory
        output_dir = self.controller.get_output_dir()

        # Validate first
        errors = self.config_manager.validate(self.controller.config)
        if errors:
            error_msg = "Cannot run with validation errors:\n\n" + "\n".join(f"• {e}" for e in errors)
            QMessageBox.warning(self, "Validation Failed", error_msg)
            return False

        # Check if in remote mode using storage type
        from openbench.remote.storage import RemoteStorage

        is_remote = isinstance(self.controller.storage, RemoteStorage)

        if is_remote:
            # TODO: Refactor to use ProjectStorage interface for unified export
            # Currently uses direct SSH/SFTP operations
            return self._export_and_run_remote(output_dir)
        else:
            return self._export_and_run_local(output_dir)

    def _export_and_run_local(self, output_dir: str) -> bool:
        """Export files locally and trigger run."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        try:
            openbench_root = self._get_openbench_root()
            files = self.config_manager.export_all(self.controller.config, output_dir, openbench_root=openbench_root)

            # Navigate to run page
            self.controller.go_to_page("run_monitor")

            # Emit signal with main config path
            self.run_requested.emit(files["main"])
            return True

        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
            return False

    def _export_and_run_remote(self, output_dir: str) -> bool:
        """Export files to remote server and trigger run."""
        ssh_manager = get_remote_ssh_manager(self.controller)
        if not ssh_manager or not ssh_manager.is_connected:
            QMessageBox.warning(
                self, "Not Connected", "Please connect to the remote server first in the Runtime Environment page."
            )
            return False

        try:
            # Create output directory on remote server
            nml_dir = f"{output_dir}/nml"
            sim_nml_dir = f"{nml_dir}/sim"
            ref_nml_dir = f"{nml_dir}/ref"

            stdout, stderr, exit_code = ssh_manager.execute(
                f"mkdir -p '{nml_dir}' '{sim_nml_dir}' '{ref_nml_dir}'", timeout=30
            )
            if exit_code != 0:
                QMessageBox.critical(self, "Error", f"Failed to create remote directories:\n{stderr}")
                return False

            # Export to local temp directory, but use remote paths in generated files
            with tempfile.TemporaryDirectory() as temp_dir:
                openbench_root = self._get_openbench_root()

                # Get remote OpenBench path from remote config
                remote_config = self.controller.config.get("general", {}).get("remote", {})
                remote_openbench_path = remote_config.get("openbench_path", "")

                # Generate config files with remote output_dir paths
                # This ensures paths like reference_nml point to remote locations
                files = self._export_for_remote(temp_dir, output_dir, openbench_root, remote_openbench_path)

                # Debug: list temp directory contents before upload
                debug_log = os.path.join(tempfile.gettempdir(), "openbench_wizard_debug.log")
                with open(debug_log, "a") as f:
                    f.write(f"\n=== Before upload ===\n")
                    f.write(f"temp_dir: {temp_dir}\n")
                    local_nml_dir = os.path.join(temp_dir, "nml")
                    f.write(f"local_nml_dir: {local_nml_dir}\n")
                    # List all files recursively
                    for root, dirs, files_list in os.walk(local_nml_dir):
                        rel_root = os.path.relpath(root, local_nml_dir)
                        f.write(f"  dir: {rel_root}/\n")
                        for fname in files_list:
                            f.write(f"    file: {fname}\n")

                # Upload files to remote server
                sftp = ssh_manager._client.open_sftp()
                try:
                    # Upload all files in nml directory
                    self._upload_directory(sftp, local_nml_dir, nml_dir)
                finally:
                    sftp.close()

            # Navigate to run page
            self.controller.go_to_page("run_monitor")

            # Emit signal with remote main config path
            basename = self.controller.config.get("general", {}).get("basename", "config")
            remote_main_path = f"{nml_dir}/main-{basename}.yaml"
            self.run_requested.emit(remote_main_path)
            return True

        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
            return False

    def _export_for_remote(
        self, local_dir: str, remote_dir: str, openbench_root: str, remote_openbench_path: str = ""
    ) -> dict:
        """Export config files locally but with remote paths inside.

        Args:
            local_dir: Local directory to write files to
            remote_dir: Remote directory that paths should point to
            openbench_root: OpenBench root directory (local, for fallback)
            remote_openbench_path: OpenBench installation path on remote server

        Returns:
            Dictionary of {file_type: local_file_path}
        """
        import yaml

        config = self.controller.config
        basename = config.get("general", {}).get("basename", "config")

        # Create local nml directories
        nml_dir = os.path.join(local_dir, "nml")
        sim_nml_dir = os.path.join(nml_dir, "sim")
        ref_nml_dir = os.path.join(nml_dir, "ref")
        os.makedirs(sim_nml_dir, exist_ok=True)
        os.makedirs(ref_nml_dir, exist_ok=True)

        files = {}

        # Generate main config with remote paths
        remote_nml_dir = f"{remote_dir}/nml"
        main_content = self.config_manager.generate_main_nml(config, openbench_root, remote_dir, remote_openbench_path)
        main_path = os.path.join(nml_dir, f"main-{basename}.yaml")
        with open(main_path, "w", encoding="utf-8") as f:
            f.write(main_content)
        files["main"] = main_path

        # Generate ref config with remote paths
        ref_content = self.config_manager.generate_ref_nml(config, openbench_root, remote_dir)
        ref_path = os.path.join(nml_dir, f"ref-{basename}.yaml")
        with open(ref_path, "w", encoding="utf-8") as f:
            f.write(ref_content)
        files["ref"] = ref_path

        # Generate sim config with remote paths
        sim_content = self.config_manager.generate_sim_nml(config, openbench_root, remote_dir)
        sim_path = os.path.join(nml_dir, f"sim-{basename}.yaml")
        with open(sim_path, "w", encoding="utf-8") as f:
            f.write(sim_content)
        files["sim"] = sim_path

        # Sync namelists (source definition files) with remote paths
        self._sync_namelists_for_remote(config, local_dir, remote_dir, openbench_root)

        return files

    def _sync_namelists_for_remote(self, config: dict, local_dir: str, remote_dir: str, openbench_root: str):
        """Sync namelist files with remote paths."""
        import yaml
        import shutil
        import tempfile
        from openbench.gui.path_utils import remote_join

        # Debug logging at function start
        debug_log = os.path.join(tempfile.gettempdir(), "openbench_wizard_debug.log")
        with open(debug_log, "a") as f:
            f.write(f"\n=== _sync_namelists_for_remote called ===\n")
            f.write(f"local_dir: {local_dir}\n")
            f.write(f"remote_dir: {remote_dir}\n")
            f.write(f"openbench_root: {openbench_root}\n")

        # Local directories for writing files
        nml_dir = os.path.join(local_dir, "nml")
        sim_nml_dir = os.path.join(nml_dir, "sim")
        ref_nml_dir = os.path.join(nml_dir, "ref")
        sim_models_dir = os.path.join(sim_nml_dir, "models")

        # Remote directories (paths that will be embedded in config files)
        remote_nml_dir = remote_join(remote_dir, "nml")
        remote_sim_nml_dir = remote_join(remote_nml_dir, "sim")
        remote_ref_nml_dir = remote_join(remote_nml_dir, "ref")

        os.makedirs(sim_nml_dir, exist_ok=True)
        os.makedirs(ref_nml_dir, exist_ok=True)
        os.makedirs(sim_models_dir, exist_ok=True)

        # Get SSH manager for remote file reading
        ssh_manager = get_remote_ssh_manager(self.controller)

        eval_items = config.get("evaluation_items", {})
        selected_items = [k for k, v in eval_items.items() if v]

        # Process simulation data namelists - group by source_name
        sim_data = config.get("sim_data", {})
        sim_source_configs = sim_data.get("source_configs", {})

        # Debug: log sim_source_configs structure
        with open(debug_log, "a") as f:
            f.write(f"sim_source_configs keys: {list(sim_source_configs.keys())}\n")
            for key, sc in sim_source_configs.items():
                general = sc.get("general", {})
                model_path = general.get("model_namelist", "")
                f.write(f"  {key}: model_namelist={model_path}\n")

        # Group configs by source_name
        sim_grouped = {}
        for key, source_config in sim_source_configs.items():
            if "::" in key:
                var_name, source_name = key.split("::", 1)
            else:
                source_name = key
                var_name = None

            if source_name not in sim_grouped:
                sim_grouped[source_name] = {"configs": [], "var_names": []}
            sim_grouped[source_name]["configs"].append(source_config)
            if var_name:
                sim_grouped[source_name]["var_names"].append(var_name)

        # Write grouped sim configs
        for source_name, group_data in sim_grouped.items():
            # Merge configs for the same source
            merged_config = self._merge_source_configs(group_data["configs"], group_data["var_names"])
            dest_path = os.path.join(sim_nml_dir, f"{source_name}.yaml")
            remote_dest_path = remote_join(remote_sim_nml_dir, f"{source_name}.yaml")
            self._write_source_config_remote(merged_config, dest_path, selected_items, openbench_root, remote_dest_path)

        # Copy model definition files for sim (read from remote server)
        # Extract model_namelist paths from source configs
        copied_models = set()

        # Debug logging for model copy phase
        with open(debug_log, "a") as f:
            f.write(f"\n=== Model copy phase ===\n")
            f.write(f"sim_models_dir: {sim_models_dir}\n")

        for key, source_config in sim_source_configs.items():
            general = source_config.get("general", {})
            model_path = general.get("model_namelist", "")
            with open(debug_log, "a") as f:
                f.write(f"key={key}, model_path={model_path}\n")
            if model_path and model_path not in copied_models:
                copied_models.add(model_path)
                actual_path = self._resolve_model_path(
                    model_path, openbench_root, is_remote=True, ssh_manager=ssh_manager
                )
                with open(debug_log, "a") as f:
                    f.write(f"resolved actual_path={actual_path}\n")
                if actual_path:
                    # Extract model name from path
                    model_basename = actual_path.rstrip("/").split("/")[-1]
                    model_name = os.path.splitext(model_basename)[0] + ".yaml"
                    dest_path = os.path.join(sim_models_dir, model_name)
                    with open(debug_log, "a") as f:
                        f.write(f"copying model from {actual_path} to {dest_path}\n")
                    self._copy_model_definition_filtered(
                        actual_path, dest_path, selected_items, is_remote=True, ssh_manager=ssh_manager
                    )
                    with open(debug_log, "a") as f:
                        f.write(f"model copy done, file exists: {os.path.exists(dest_path)}\n")

        # Process reference data namelists - group by source_name
        ref_data = config.get("ref_data", {})
        ref_source_configs = ref_data.get("source_configs", {})

        # Group configs by source_name
        ref_grouped = {}
        for key, source_config in ref_source_configs.items():
            if "::" in key:
                var_name, source_name = key.split("::", 1)
            else:
                source_name = key
                var_name = None

            if source_name not in ref_grouped:
                ref_grouped[source_name] = {"configs": [], "var_names": []}
            ref_grouped[source_name]["configs"].append(source_config)
            if var_name:
                ref_grouped[source_name]["var_names"].append(var_name)

        # Write grouped ref configs
        for source_name, group_data in ref_grouped.items():
            # Merge configs for the same source
            merged_config = self._merge_source_configs(group_data["configs"], group_data["var_names"])
            dest_path = os.path.join(ref_nml_dir, f"{source_name}.yaml")
            remote_dest_path = remote_join(remote_ref_nml_dir, f"{source_name}.yaml")
            self._write_source_config_remote(merged_config, dest_path, selected_items, openbench_root, remote_dest_path)

    def _merge_source_configs(self, configs: list, var_names: list) -> dict:
        """Merge multiple source configs into one.

        Creates a structure like:
        {
            "general": {...},
            "Latent_Heat": {"sub_dir": ..., "varname": ..., ...},
            "Sensible_Heat": {"sub_dir": ..., "varname": ..., ...},
        }

        Consistent with local mode _copy_data_namelists organization.
        """
        if not configs:
            return {}

        merged = {}
        var_mapping_keys = ["sub_dir", "varname", "varunit", "prefix", "suffix"]

        # Use the first config's general section
        for cfg in configs:
            if "general" in cfg and isinstance(cfg["general"], dict):
                merged["general"] = cfg["general"].copy()
                break

        # Create per-variable entries from each config
        for cfg, var_name in zip(configs, var_names or [None] * len(configs)):
            if not var_name:
                continue

            # Build variable-specific config from top-level fields
            var_config = {}
            for key in var_mapping_keys:
                if key in cfg:
                    var_config[key] = cfg[key]

            # Also check if there's already a variable entry in the config
            if var_name in cfg and isinstance(cfg[var_name], dict):
                # Merge with existing variable config
                for k, v in cfg[var_name].items():
                    var_config[k] = v

            # Handle per-variable time range settings (consistent with local mode)
            general = cfg.get("general", {})
            if general.get("per_var_time_range"):
                var_config["per_var_time_range"] = True
                if "syear" in general and general["syear"] != "":
                    var_config["syear"] = general["syear"]
                if "eyear" in general and general["eyear"] != "":
                    var_config["eyear"] = general["eyear"]

            if var_config:
                merged[var_name] = var_config

        return merged

    def _resolve_model_path(
        self, model_path: str, openbench_root: str, is_remote: bool = False, ssh_manager=None
    ) -> str:
        """Resolve model definition path, handling .nml to .yaml conversion.

        For remote mode, works the same as local mode:
        - Extract relative path from model_path (using /nml/ marker)
        - Join with openbench_root (remote or local)
        - Check file existence and try .yaml extension
        """
        from openbench.gui.path_utils import to_absolute_path, to_posix_path
        import shlex

        if not model_path:
            return ""

        if is_remote:
            # Normalize path separators
            path = to_posix_path(model_path)

            # If it's already an absolute path on the remote server, try it directly first
            if path.startswith("/") and ssh_manager:
                paths_to_try = [path]
                if path.endswith(".nml"):
                    paths_to_try.insert(0, path[:-4] + ".yaml")
                elif not path.endswith(".yaml"):
                    paths_to_try.append(path + ".yaml")

                for try_path in paths_to_try:
                    try:
                        quoted_path = shlex.quote(try_path)
                        stdout, stderr, exit_code = ssh_manager.execute(
                            f"test -f {quoted_path} && echo 'exists'", timeout=10
                        )
                        if exit_code == 0 and "exists" in stdout:
                            return try_path
                    except Exception:
                        pass

            # Handle Windows absolute paths (e.g., C:/Users/...)
            if len(path) >= 2 and path[1] == ":":
                # This is a Windows local path - extract relative part if contains /nml/
                if "/nml/" in path:
                    relative_path = "nml/" + path.split("/nml/", 1)[1]
                else:
                    # Unknown Windows path
                    relative_path = path
            # Extract relative path - look for /nml/ marker (handles local temp paths)
            elif "/nml/" in path:
                relative_path = "nml/" + path.split("/nml/", 1)[1]
            elif path.startswith("./"):
                relative_path = path[2:]
            elif path.startswith("/"):
                # Already absolute but file doesn't exist, try to resolve
                if "/nml/" in path:
                    relative_path = "nml/" + path.split("/nml/", 1)[1]
                else:
                    relative_path = None
            else:
                # Relative path
                relative_path = path

            # Build absolute path on remote server
            if relative_path and openbench_root:
                path = f"{openbench_root.rstrip('/')}/{relative_path}"

            # Try different extensions on remote
            if ssh_manager:
                paths_to_try = [path]
                if path.endswith(".nml"):
                    paths_to_try.insert(0, path[:-4] + ".yaml")
                elif not path.endswith(".yaml"):
                    paths_to_try.append(path + ".yaml")

                for try_path in paths_to_try:
                    try:
                        quoted_path = shlex.quote(try_path)
                        stdout, stderr, exit_code = ssh_manager.execute(
                            f"test -f {quoted_path} && echo 'exists'", timeout=10
                        )
                        if exit_code == 0 and "exists" in stdout:
                            return try_path
                    except Exception:
                        pass

            return path

        # Local mode: Convert to absolute path
        abs_path = to_absolute_path(model_path, openbench_root)

        # If file exists, return it
        if os.path.exists(abs_path):
            return abs_path

        # Try .yaml extension if .nml was specified
        if abs_path.endswith(".nml"):
            yaml_path = abs_path[:-4] + ".yaml"
            if os.path.exists(yaml_path):
                return yaml_path

        return ""

    def _copy_model_definition_filtered(
        self, src_path: str, dest_path: str, selected_items: list, is_remote: bool = False, ssh_manager=None
    ):
        """Copy model definition file with filtering for selected items."""
        import yaml

        content = {}

        if is_remote and ssh_manager:
            # Read from remote server via SSH
            try:
                # Try .yaml first, then .nml
                paths_to_try = [src_path]
                if src_path.endswith(".nml"):
                    paths_to_try.insert(0, src_path[:-4] + ".yaml")

                for path in paths_to_try:
                    stdout, stderr, exit_code = ssh_manager.execute(f"cat '{path}' 2>/dev/null", timeout=30)
                    if exit_code == 0 and stdout.strip():
                        content = yaml.safe_load(stdout) or {}
                        break
            except Exception:
                return
        else:
            # Read from local file
            try:
                with open(src_path, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f) or {}
            except Exception:
                return

        if not content:
            return

        # Filter to only include selected items
        filtered = {}

        # Always include general section (contains model name, etc.)
        if "general" in content and isinstance(content["general"], dict):
            filtered["general"] = content["general"].copy()

        # Include selected evaluation items
        for item in selected_items:
            if item in content and isinstance(content[item], dict):
                filtered[item] = content[item].copy()

        if filtered:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "w", encoding="utf-8") as f:
                yaml.dump(filtered, f, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)

    def _resolve_path_for_remote(self, path: str, openbench_root: str) -> str:
        """Convert a path to absolute remote path.

        Works the same as local mode's to_absolute_path() - but for remote paths.
        Handles both Unix and Windows local paths.
        """
        from openbench.gui.path_utils import to_posix_path

        if not path:
            return ""

        # Convert to POSIX format (forward slashes)
        path = to_posix_path(path)

        # Handle Windows absolute paths (e.g., C:/Users/... or D:/...)
        # After to_posix_path, Windows paths become like "C:/Users/..."
        if len(path) >= 2 and path[1] == ":":
            # This is a Windows local path - extract relative part if contains /nml/
            if "/nml/" in path:
                relative_path = "nml/" + path.split("/nml/", 1)[1]
                return f"{openbench_root.rstrip('/')}/{relative_path}"
            # Unknown Windows path - cannot convert to remote
            return path

        # If path is already an absolute Unix path
        if path.startswith("/"):
            # Check if it's a valid remote server path
            if any(
                path.startswith(p) for p in ["/home/", "/share/", "/data/", "/work/", "/scratch/", "/opt/", "/usr/"]
            ):
                return path
            # Handle local temp paths (like /var/folders/... or /tmp/...) - extract relative part
            if "/nml/" in path:
                relative_path = "nml/" + path.split("/nml/", 1)[1]
                return f"{openbench_root.rstrip('/')}/{relative_path}"
            # Unknown absolute path - return as-is (might be a valid remote path)
            return path

        # Handle relative paths starting with ./
        if path.startswith("./"):
            path = path[2:]

        # Join relative path with openbench_root
        if openbench_root:
            return f"{openbench_root.rstrip('/')}/{path}"

        return path

    def _write_source_config_remote(
        self, source_data: dict, dest_path: str, selected_items: list, openbench_root: str, remote_dest_path: str = ""
    ):
        """Write source config file for remote execution.

        Expects source_data to have structure:
        {
            "general": {...},
            "Latent_Heat": {"sub_dir": ..., "varname": ..., ...},
            "Sensible_Heat": {"sub_dir": ..., "varname": ..., ...},
        }

        Output is consistent with local mode _write_source_config_organized.

        Args:
            source_data: Source configuration data
            dest_path: Local path to write the file
            selected_items: List of selected evaluation items
            openbench_root: OpenBench root path (remote)
            remote_dest_path: Remote path that will be used in the config (for model_namelist paths)
        """
        import yaml
        from openbench.gui.path_utils import remote_join

        if not source_data:
            return

        filtered = {}

        # Path fields that need to be converted to absolute remote paths
        path_fields = ["root_dir", "basedir", "fulllist", "data_path", "file_path"]

        # Internal fields that should NOT be written to output (UI-only or internal)
        internal_fields = ["def_nml_path", "per_var_time_range", "_var_name", "source_configs"]

        # Process general section
        if "general" in source_data and isinstance(source_data["general"], dict):
            general = source_data["general"].copy()

            # Remove internal fields from general
            for field in internal_fields:
                general.pop(field, None)

            # Convert path fields to absolute remote paths
            for field in path_fields:
                if field in general and general[field]:
                    general[field] = self._resolve_path_for_remote(general[field], openbench_root)

            # Update model_namelist path to point to models subdirectory (consistent with local mode)
            if "model_namelist" in general and general["model_namelist"]:
                model_path = general["model_namelist"]
                model_basename = model_path.rstrip("/").split("/")[-1]
                model_basename = os.path.splitext(model_basename)[0]
                # Use remote path for model_namelist, not local path
                if remote_dest_path:
                    dest_dir = remote_dest_path.rsplit("/", 1)[0] if "/" in remote_dest_path else remote_dest_path
                else:
                    dest_dir = dest_path.rsplit("/", 1)[0] if "/" in dest_path else os.path.dirname(dest_path)
                models_dir = remote_join(dest_dir, "models")
                general["model_namelist"] = remote_join(models_dir, model_basename + ".yaml")

            # Check if any variable has per_var_time_range enabled
            any_per_var = any(
                source_data.get(item, {}).get("per_var_time_range", False)
                for item in selected_items
                if isinstance(source_data.get(item), dict)
            )

            # If any variable uses per-variable time range, remove from general
            if any_per_var:
                general.pop("syear", None)
                general.pop("eyear", None)

            filtered["general"] = general

        # Include selected evaluation items that exist in source_data
        for item in selected_items:
            if item in source_data:
                item_data = source_data[item]
                if isinstance(item_data, dict):
                    var_config = item_data.copy()
                    # Remove internal fields from variable config
                    for field in internal_fields:
                        var_config.pop(field, None)
                    # Convert path fields in variable config to absolute remote paths
                    for field in path_fields:
                        if field in var_config and var_config[field]:
                            var_config[field] = self._resolve_path_for_remote(var_config[field], openbench_root)
                    if var_config:  # Only add if there's data
                        filtered[item] = var_config
                elif item_data is not None:
                    filtered[item] = item_data

        # Don't write empty files (only general, no variables)
        if len(filtered) <= 1:
            return

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            yaml.dump(filtered, f, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)

    def _upload_directory(self, sftp, local_dir: str, remote_dir: str):
        """Recursively upload a directory to remote server."""
        import tempfile

        debug_log = os.path.join(tempfile.gettempdir(), "openbench_wizard_debug.log")

        if not os.path.exists(local_dir):
            with open(debug_log, "a") as f:
                f.write(f"_upload_directory: local_dir does not exist: {local_dir}\n")
            return

        with open(debug_log, "a") as f:
            f.write(f"\n=== Uploading directory ===\n")
            f.write(f"local_dir: {local_dir}\n")
            f.write(f"remote_dir: {remote_dir}\n")
            f.write(f"contents: {os.listdir(local_dir)}\n")

        for item in os.listdir(local_dir):
            local_path = os.path.join(local_dir, item)
            remote_path = f"{remote_dir}/{item}"

            if os.path.isfile(local_path):
                with open(debug_log, "a") as f:
                    f.write(f"  uploading file: {local_path} -> {remote_path}\n")
                sftp.put(local_path, remote_path)
            elif os.path.isdir(local_path):
                # Create remote directory
                with open(debug_log, "a") as f:
                    f.write(f"  creating remote dir: {remote_path}\n")
                try:
                    sftp.mkdir(remote_path)
                except IOError as e:
                    with open(debug_log, "a") as f:
                        f.write(f"    mkdir error (may already exist): {e}\n")
                self._upload_directory(sftp, local_path, remote_path)

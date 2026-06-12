# -*- coding: utf-8 -*-
"""
Preview and Export page.
"""

import logging
import os
import posixpath
import tempfile

from PySide6.QtWidgets import QHBoxLayout, QLabel, QMessageBox
from PySide6.QtCore import Signal

from openbench.gui.remote_python import quote_remote_path
from openbench.gui.widgets._ssh_worker import execute_responsive
from openbench.gui.pages.base_page import BasePage
from openbench.gui.widgets import YamlPreview
from openbench.gui.config_manager import ConfigManager
# get_openbench_root is now inherited from BasePage via _get_openbench_root()

logger = logging.getLogger(__name__)


class RemoteNamelistSyncError(RuntimeError):
    """Raised when remote namelist/model synchronization cannot proceed safely."""


from openbench.gui.path_utils import get_remote_ssh_manager


class PagePreview(BasePage):
    """Preview and Export page."""

    PAGE_ID = "preview"
    PAGE_TITLE = "Preview & Export"
    PAGE_SUBTITLE = "Review generated configuration and export files"

    run_requested = Signal(str)  # Emits output directory

    def _setup_content(self):
        """Setup page content."""
        self.config_manager = ConfigManager()

        # Single YAML preview
        self.config_preview = YamlPreview()
        self.content_layout.addWidget(self.config_preview, 1)

        # Output directory info
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_label = QLabel("")
        self.output_dir_label.setStyleSheet("color: #666; font-style: italic;")
        info_layout.addWidget(self.output_dir_label, 1)
        self.content_layout.addLayout(info_layout)

    def load_from_config(self):
        """Load and generate previews."""

        config = self.controller.config

        # Update output directory display
        output_dir = self.controller.get_output_dir()
        self.output_dir_label.setText(output_dir)

        # Generate unified config preview
        generate_kwargs = {"case_output_dir": output_dir}
        from openbench.remote.storage import RemoteStorage

        if isinstance(self.controller.storage, RemoteStorage):
            remote_path_base = self.controller.remote_settings().get("openbench_path", "") or output_dir
            generate_kwargs["path_transform"] = lambda path: self._resolve_path_for_remote(path, remote_path_base)

        config_yaml = self.config_manager.generate_config_yaml(config, **generate_kwargs)
        self.config_preview.set_content(config_yaml)

    def export_and_run(self) -> bool:
        """Export files and trigger run. Returns True if successful."""
        if getattr(self, "_export_in_progress", False):
            logger.warning("Ignoring duplicate Run request while export is still in progress")
            return False

        self._export_in_progress = True
        try:
            return self._export_and_run_once()
        finally:
            self._export_in_progress = False

    def _export_and_run_once(self) -> bool:
        """Export files and trigger one run attempt."""
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

            # Emit signal with config path
            self.run_requested.emit(files["config"])
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

            stdout, stderr, exit_code = execute_responsive(
                ssh_manager,
                f"mkdir -p {quote_remote_path(nml_dir)} {quote_remote_path(sim_nml_dir)} {quote_remote_path(ref_nml_dir)}",
                timeout=30,
            )
            if exit_code != 0:
                QMessageBox.critical(self, "Error", f"Failed to create remote directories:\n{stderr}")
                return False

            # Export to local temp directory, but use remote paths in generated files
            with tempfile.TemporaryDirectory() as temp_dir:
                openbench_root = self._get_openbench_root()

                # Get remote OpenBench path from remote config
                remote_openbench_path = self.controller.remote_settings().get("openbench_path", "")

                # Generate config files with remote output_dir paths
                # This ensures paths like reference_nml point to remote locations
                files = self._export_for_remote(temp_dir, output_dir, openbench_root, remote_openbench_path)

                local_nml_dir = os.path.join(temp_dir, "nml")
                local_config_path = files.get("config")
                if not local_config_path or not os.path.exists(local_config_path):
                    raise RemoteNamelistSyncError("Remote export did not create openbench.yaml")
                if not os.path.isdir(local_nml_dir):
                    raise RemoteNamelistSyncError("Remote export did not create the nml directory")

                # Upload files to remote server. The sftp client is owned
                # by SSHManager (cached, reused), so we must NOT close it
                # here — SSHManager.disconnect() handles its lifecycle.
                sftp = ssh_manager.open_sftp()
                # Upload all files in nml directory
                uploaded_files = self._upload_directory(sftp, local_nml_dir, nml_dir)
                for local_path, remote_path in uploaded_files:
                    self._mark_remote_upload_synced(local_path, remote_path)
                # Upload the v3 unified config alongside the nml/
                # tree so the runner can read it directly.
                sftp.put(local_config_path, f"{output_dir}/openbench.yaml")
                self._mark_remote_upload_synced(local_config_path, f"{output_dir}/openbench.yaml")

            # Navigate to run page
            self.controller.go_to_page("run_monitor")

            # Emit the v3 unified config path (uploaded alongside the
            # legacy main-/ref-/sim- triple). The previous code emitted
            # the legacy main-{basename}.yaml, which the v3 `openbench
            # run` entry point cannot read directly — it would refuse
            # the file and instruct the user to run `openbench migrate`.
            remote_config_path = f"{output_dir}/openbench.yaml"
            self.run_requested.emit(remote_config_path)
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

        support_files = self.config_manager.write_legacy_support_namelists(config, nml_dir)
        files["statistics"] = support_files["statistics"]
        files["figure"] = support_files["figure"]

        # Also emit the v3 unified config. The v3 `openbench run` entry
        # point reads this unified format directly; the legacy main-/ref-/
        # sim-{basename}.yaml triple above is kept for back-compat with
        # `openbench migrate` and any tooling that still consumes the
        # split layout. The remote run path emits the unified config so
        # the runner does not have to migrate at run time.
        remote_path_base = remote_openbench_path or remote_dir
        config_content = self.config_manager.generate_config_yaml(
            config,
            case_output_dir=remote_dir,
            path_transform=lambda path: self._resolve_path_for_remote(path, remote_path_base),
        )
        config_path = os.path.join(local_dir, "openbench.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)
        files["config"] = config_path

        # Sync namelists (source definition files) with remote paths
        self._sync_namelists_for_remote(config, local_dir, remote_dir, openbench_root)

        return files

    def _sync_namelists_for_remote(self, config: dict, local_dir: str, remote_dir: str, openbench_root: str):
        """Sync namelist files with remote paths."""
        from openbench.gui.path_utils import remote_join

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
        # Extract model_namelist paths from source configs. Previously this
        # block emitted progress to a `remote_sync_debug.log` written into
        # local_dir, which then got uploaded verbatim to the remote server
        # along with the rest of the directory — leaking local paths and
        # polluting the remote nml/ tree. Use the standard logger instead.
        copied_models = set()
        logger.debug("=== Model copy phase ===")
        logger.debug("sim_models_dir: %s", sim_models_dir)

        for key, source_config in sim_source_configs.items():
            general = source_config.get("general", {})
            model_path = general.get("model_namelist", "")
            logger.debug("key=%s, model_path=%s", key, model_path)
            if model_path and model_path not in copied_models:
                copied_models.add(model_path)

                # Scan-based assignments store bare registry model names
                # (e.g. "CoLM2024"), not file paths. Generate the definition
                # from the local registry instead of searching the remote
                # filesystem for a file that never existed there.
                from openbench.gui.config_manager import model_definition_from_registry

                registry_content = model_definition_from_registry(model_path, selected_items)
                if registry_content is not None:
                    import yaml as _yaml

                    dest_path = os.path.join(sim_models_dir, f"{model_path}.yaml")
                    with open(dest_path, "w", encoding="utf-8") as f:
                        _yaml.dump(
                            registry_content, f, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2
                        )
                    logger.debug("model %s generated from local registry → %s", model_path, dest_path)
                    self._stage_remote_registry_model(model_path, ssh_manager)
                    continue

                actual_path = self._resolve_model_path(
                    model_path, openbench_root, is_remote=True, ssh_manager=ssh_manager
                )
                logger.debug("resolved actual_path=%s", actual_path)
                if not actual_path:
                    raise RemoteNamelistSyncError(
                        f"Remote model definition not found: {model_path} "
                        "(not a registered model name and no matching file on the remote server)"
                    )

                # Extract model name from path
                model_basename = actual_path.rstrip("/").split("/")[-1]
                model_name = os.path.splitext(model_basename)[0] + ".yaml"
                dest_path = os.path.join(sim_models_dir, model_name)
                logger.debug("copying model from %s to %s", actual_path, dest_path)
                copied = self._copy_model_definition_filtered(
                    actual_path, dest_path, selected_items, is_remote=True, ssh_manager=ssh_manager
                )
                if not copied:
                    raise RemoteNamelistSyncError(f"Failed to copy remote model definition: {actual_path}")
                logger.debug("model copy done, file exists: %s", os.path.exists(dest_path))

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

    def _stage_remote_registry_model(self, model_name: str, ssh_manager) -> None:
        """Upload a user-registered model profile to the remote registry.

        The remote ``openbench run`` resolves ``simulation.<case>.model``
        through its own registry. Built-in models ship with every install,
        but models the user registered locally (e.g. via the Data Registry
        page) must be staged into the remote ``~/.openbench/models/`` overlay
        or the remote preflight rejects the model name.
        """
        from openbench.gui.config_manager import is_builtin_model, registry_model_profile
        from openbench.gui.path_utils import remote_home_dir

        if is_builtin_model(model_name):
            return
        profile = registry_model_profile(model_name)
        if profile is None or not ssh_manager or not ssh_manager.is_connected:
            return

        import tempfile

        import yaml as _yaml

        try:
            home = remote_home_dir(ssh_manager)
            if not home or home == "/":
                raise RemoteNamelistSyncError("could not determine remote home directory")
            remote_models_dir = f"{home.rstrip('/')}/.openbench/models"
            _stdout, stderr, exit_code = execute_responsive(
                ssh_manager, f"mkdir -p {quote_remote_path(remote_models_dir)}", timeout=15
            )
            if exit_code != 0:
                raise RemoteNamelistSyncError(stderr.strip() or f"mkdir exit code {exit_code}")

            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
                _yaml.dump(
                    profile.to_dict(), tmp, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2
                )
                tmp_path = tmp.name
            try:
                remote_path = f"{remote_models_dir}/{model_name}.yaml"
                ssh_manager.upload_file(tmp_path, remote_path)
                logger.info("Staged model profile %s to remote registry: %s", model_name, remote_path)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        except Exception as exc:
            raise RemoteNamelistSyncError(
                f"Could not stage model profile '{model_name}' on the remote server: {exc}"
            ) from exc

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
                        quoted_path = quote_remote_path(try_path)
                        stdout, stderr, exit_code = execute_responsive(
                            ssh_manager, f"test -f {quoted_path} && echo 'exists'", timeout=10
                        )
                        if exit_code == 0 and "exists" in stdout:
                            return try_path
                    except Exception as exc:
                        raise RemoteNamelistSyncError(f"Failed to check remote model path {try_path}: {exc}") from exc

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
                        quoted_path = quote_remote_path(try_path)
                        stdout, stderr, exit_code = execute_responsive(
                            ssh_manager, f"test -f {quoted_path} && echo 'exists'", timeout=10
                        )
                        if exit_code == 0 and "exists" in stdout:
                            return try_path
                    except Exception as exc:
                        raise RemoteNamelistSyncError(f"Failed to check remote model path {try_path}: {exc}") from exc

            return ""

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
    ) -> bool:
        """Copy model definition file with filtering for selected items.

        Returns:
            True when a filtered model file was written, False when the
            source was absent/empty or none of the selected items matched.

        Raises:
            RemoteNamelistSyncError: if SSH access or YAML parsing fails.
        """
        import yaml

        content = {}

        if is_remote and ssh_manager:
            # Read from remote server via SSH
            paths_to_try = [src_path]
            if src_path.endswith(".nml"):
                paths_to_try.insert(0, src_path[:-4] + ".yaml")

            last_error = ""
            for path in paths_to_try:
                try:
                    stdout, stderr, exit_code = execute_responsive(
                        ssh_manager, f"cat {quote_remote_path(path)} 2>/dev/null", timeout=30
                    )
                except Exception as exc:
                    raise RemoteNamelistSyncError(f"Failed to read remote model definition {path}: {exc}") from exc
                if exit_code == 0 and stdout.strip():
                    try:
                        content = yaml.safe_load(stdout) or {}
                    except yaml.YAMLError as exc:
                        raise RemoteNamelistSyncError(f"Invalid YAML in remote model definition {path}: {exc}") from exc
                    break
                last_error = stderr.strip() or stdout.strip() or f"exit code {exit_code}"
            if not content and last_error:
                logger.warning("Could not read remote model definition %s: %s", src_path, last_error)
        else:
            # Read from local file
            try:
                with open(src_path, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f) or {}
            except OSError as exc:
                logger.warning("Could not read local model definition %s: %s", src_path, exc)
                return False
            except yaml.YAMLError as exc:
                raise RemoteNamelistSyncError(f"Invalid YAML in model definition {src_path}: {exc}") from exc

        if not content:
            return False

        # Filter to only include selected items
        filtered = {}

        # Always include general section (contains model name, etc.)
        if "general" in content and isinstance(content["general"], dict):
            filtered["general"] = content["general"].copy()

        # Include selected evaluation items
        for item in selected_items:
            if item in content and isinstance(content[item], dict):
                filtered[item] = content[item].copy()

        if not filtered:
            return False

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "w", encoding="utf-8") as f:
            yaml.dump(filtered, f, default_flow_style=False, allow_unicode=True, sort_keys=False, indent=2)
        return True

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
            remote_root = to_posix_path(openbench_root).rstrip("/") if openbench_root else ""
            if remote_root and (path == remote_root or path.startswith(remote_root + "/")):
                return path
            # Handle local temp paths (like /var/folders/... or /tmp/...) - extract relative part
            if "/nml/" in path:
                relative_path = "nml/" + path.split("/nml/", 1)[1]
                return f"{openbench_root.rstrip('/')}/{relative_path}"
            if os.path.exists(path):
                raise RemoteNamelistSyncError(
                    "Ambiguous local absolute path cannot be converted to a remote path: "
                    f"{path}. Choose or enter the corresponding remote server path."
                )
            # Check if it's a plausible remote server path. Existing local
            # paths with the same prefix are rejected above so a local /data
            # tree cannot silently leak into remote YAML.
            if any(
                path.startswith(p) for p in ["/home/", "/share/", "/data/", "/work/", "/scratch/", "/opt/", "/usr/"]
            ):
                return path
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

    def _remote_storage_relpath(self, remote_path: str) -> str | None:
        """Return storage-relative path for a remote path, or None if outside storage root."""
        storage = getattr(self.controller, "storage", None)
        storage_root = getattr(storage, "project_dir", "")
        if not storage_root:
            return None
        root = posixpath.normpath(str(storage_root).rstrip("/").replace("\\", "/"))
        path = posixpath.normpath(str(remote_path).replace("\\", "/"))
        if path == root:
            return ""
        if not path.startswith(root + "/"):
            return None
        return posixpath.relpath(path, root)

    def _mark_remote_upload_synced(self, local_path: str, remote_path: str):
        """Update RemoteStorage cache after a direct SFTP upload."""
        storage = getattr(self.controller, "storage", None)
        mark_synced = getattr(storage, "mark_synced", None)
        if mark_synced is None:
            return
        rel_path = self._remote_storage_relpath(remote_path)
        if rel_path is None:
            return
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                mark_synced(rel_path, f.read())
        except Exception as exc:
            logger.warning("Failed to update remote storage cache for %s: %s", remote_path, exc)

    def _upload_directory(self, sftp, local_dir: str, remote_dir: str):
        """Recursively upload a directory to remote server.

        Returns a list of ``(local_path, remote_path)`` files uploaded, so
        callers that bypass ProjectStorage can still refresh its cache.
        """
        if not os.path.exists(local_dir):
            logger.warning("_upload_directory: local_dir does not exist: %s", local_dir)
            return []

        uploaded = []
        for item in os.listdir(local_dir):
            local_path = os.path.join(local_dir, item)
            remote_path = f"{remote_dir}/{item}"

            if os.path.isfile(local_path):
                logger.debug("Uploading file: %s -> %s", local_path, remote_path)
                sftp.put(local_path, remote_path)
                uploaded.append((local_path, remote_path))
            elif os.path.isdir(local_path):
                try:
                    sftp.mkdir(remote_path)
                except IOError:
                    pass  # May already exist
                uploaded.extend(self._upload_directory(sftp, local_path, remote_path))
        return uploaded

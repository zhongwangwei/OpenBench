# -*- coding: utf-8 -*-
"""
Wizard flow controller - manages page navigation and visibility.
"""

import os
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from PySide6.QtCore import QObject, Signal

from openbench.gui.config_manager import ConfigManager
from openbench.gui.path_utils import get_openbench_root

if TYPE_CHECKING:
    from openbench.remote.storage import ProjectStorage


class WizardController(QObject):
    """Controls wizard page flow and configuration state."""

    # Signals
    page_changed = Signal(str)  # Emitted when current page changes
    config_updated = Signal(dict)  # Emitted when config is modified
    pages_visibility_changed = Signal()  # Emitted when visible pages change

    # All possible pages in order
    ALL_PAGES = [
        "runtime",
        "general",
        "evaluation_items",
        "metrics",
        "scores",
        "comparisons",
        "statistics",
        "ref_data",
        "sim_data",
        "preview",
        "run_monitor",
    ]

    # Page display names
    PAGE_NAMES = {
        "runtime": "Runtime Environment",
        "general": "General",
        "evaluation_items": "Evaluation Items",
        "metrics": "Metrics",
        "scores": "Scores",
        "comparisons": "Comparisons",
        "statistics": "Statistics",
        "ref_data": "Reference Data",
        "sim_data": "Simulation Data",
        "preview": "Preview & Export",
        "run_monitor": "Run & Monitor",
    }

    # Conditional pages and their toggle keys
    CONDITIONAL_PAGES = {
        "comparisons": "comparison",
        "statistics": "statistics",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._config: Dict[str, Any] = self._default_config()
        self._current_page: str = "runtime"
        self._project_root: str = ""
        self._config_manager = ConfigManager()
        self._auto_sync_enabled = True
        self._ssh_manager = None  # SSH manager for remote mode
        self._storage: Optional['ProjectStorage'] = None

    @property
    def project_root(self) -> str:
        """Get project root directory."""
        return self._project_root

    @project_root.setter
    def project_root(self, value: str):
        """Set project root directory."""
        self._project_root = value

    @property
    def ssh_manager(self):
        """Get SSH manager for remote mode."""
        return self._ssh_manager

    @ssh_manager.setter
    def ssh_manager(self, value):
        """Set SSH manager for remote mode."""
        self._ssh_manager = value

    @property
    def storage(self) -> Optional['ProjectStorage']:
        """Get project storage instance."""
        return self._storage

    @storage.setter
    def storage(self, value: Optional['ProjectStorage']):
        """Set project storage instance."""
        self._storage = value

    def is_remote_mode(self) -> bool:
        """Check if using remote storage."""
        from openbench.remote.storage import RemoteStorage
        return isinstance(self._storage, RemoteStorage)

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration structure."""
        return {
            "general": {
                "basename": "",
                "basedir": "./output",
                "compare_tim_res": "month",
                "compare_tzone": 0.0,
                "compare_grid_res": 2.0,
                "syear": 2000,
                "eyear": 2020,
                "min_lat": -90.0,
                "max_lat": 90.0,
                "min_lon": -180.0,
                "max_lon": 180.0,
                "num_cores": 4,
                "evaluation": True,
                "comparison": True,
                "statistics": False,
                "debug_mode": False,
                "generate_report": True,
                "IGBP_groupby": True,
                "PFT_groupby": True,
                "Climate_zone_groupby": True,
            },
            "evaluation_items": {},
            "metrics": {},
            "scores": {},
            "comparisons": {},
            "statistics": {},
            "ref_data": {"general": {}, "def_nml": {}},
            "sim_data": {"general": {}, "def_nml": {}},
        }

    @property
    def config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config

    @config.setter
    def config(self, value: Dict[str, Any]):
        """Set configuration and emit signal."""
        self._config = value
        self.config_updated.emit(self._config)
        self.pages_visibility_changed.emit()

    def update_config(self, section: str, key: str, value: Any):
        """Update a specific config value."""
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value
        self.config_updated.emit(self._config)

        # Check if this affects page visibility
        if section == "general" and key in self.CONDITIONAL_PAGES.values():
            self.pages_visibility_changed.emit()

    def update_section(self, section: str, data: Dict[str, Any]):
        """Update entire config section."""
        self._config[section] = data
        self.config_updated.emit(self._config)

        # Check if this affects page visibility
        if section == "general":
            self.pages_visibility_changed.emit()

    def get_visible_pages(self) -> List[str]:
        """Return list of currently visible pages based on config."""
        visible = []
        general = self._config.get("general", {})

        for page in self.ALL_PAGES:
            if page in self.CONDITIONAL_PAGES:
                toggle_key = self.CONDITIONAL_PAGES[page]
                if general.get(toggle_key, False):
                    visible.append(page)
            else:
                visible.append(page)

        return visible

    def get_page_name(self, page_id: str) -> str:
        """Get display name for a page."""
        return self.PAGE_NAMES.get(page_id, page_id)

    def is_page_visible(self, page_id: str) -> bool:
        """Check if a page should be visible."""
        return page_id in self.get_visible_pages()

    @property
    def current_page(self) -> str:
        """Get current page ID."""
        return self._current_page

    @current_page.setter
    def current_page(self, page_id: str):
        """Set current page and emit signal."""
        if page_id in self.get_visible_pages():
            self._current_page = page_id
            self.page_changed.emit(page_id)

    def next_page(self) -> Optional[str]:
        """Get next visible page, or None if at end."""
        visible = self.get_visible_pages()
        try:
            idx = visible.index(self._current_page)
            if idx + 1 < len(visible):
                return visible[idx + 1]
        except ValueError:
            pass
        return None

    def prev_page(self) -> Optional[str]:
        """Get previous visible page, or None if at start."""
        visible = self.get_visible_pages()
        try:
            idx = visible.index(self._current_page)
            if idx > 0:
                return visible[idx - 1]
        except ValueError:
            pass
        return None

    def go_next(self) -> bool:
        """Navigate to next page. Returns True if successful."""
        next_p = self.next_page()
        if next_p:
            self.current_page = next_p
            return True
        return False

    def go_prev(self) -> bool:
        """Navigate to previous page. Returns True if successful."""
        prev_p = self.prev_page()
        if prev_p:
            self.current_page = prev_p
            return True
        return False

    def go_to_page(self, page_id: str) -> bool:
        """Navigate to specific page. Returns True if successful."""
        if self.is_page_visible(page_id):
            self.current_page = page_id
            return True
        return False

    def reset(self):
        """Reset to default state."""
        self._config = self._default_config()
        self._current_page = "runtime"
        self.config_updated.emit(self._config)
        self.pages_visibility_changed.emit()
        self.page_changed.emit(self._current_page)

    @property
    def config_manager(self) -> ConfigManager:
        """Get the config manager instance."""
        return self._config_manager

    @property
    def auto_sync_enabled(self) -> bool:
        """Check if auto sync is enabled."""
        return self._auto_sync_enabled

    @auto_sync_enabled.setter
    def auto_sync_enabled(self, value: bool):
        """Enable or disable auto sync."""
        self._auto_sync_enabled = value

    def get_output_dir(self) -> str:
        """Get the output directory path.

        Returns basedir/basename, avoiding duplication if basedir already ends with basename.
        In remote mode, always uses forward slashes (POSIX paths).
        """
        general = self._config.get("general", {})
        basedir = general.get("basedir", "")
        basename = general.get("basename", "config")

        # Check if in remote mode using storage type
        from openbench.remote.storage import RemoteStorage
        is_remote = isinstance(self.storage, RemoteStorage)

        if basedir and (os.path.isabs(basedir) or basedir.startswith('/')):
            # Check if basedir already ends with basename to avoid duplication
            # Use both separators for compatibility
            basedir_stripped = basedir.rstrip('/').rstrip('\\')
            basedir_basename = basedir_stripped.split('/')[-1].split('\\')[-1]
            if basedir_basename == basename:
                result = basedir
            else:
                # Append basename to basedir
                if is_remote:
                    # Remote mode: use forward slashes (POSIX)
                    result = f"{basedir.rstrip('/')}/{basename}"
                else:
                    result = os.path.join(basedir, basename)
        else:
            # Use project root to construct output path
            openbench_root = self._project_root or get_openbench_root()
            result = os.path.join(openbench_root, "output", basename)

        # In remote mode, ensure forward slashes
        if is_remote:
            result = result.replace('\\', '/')

        return result

    def sync_namelists(self):
        """
        Sync namelists to output directory.
        Called automatically when config changes if auto_sync_enabled is True.
        Also saves the main config file to the nml folder.
        """
        if not self._auto_sync_enabled:
            return

        # Check if we have enough config to sync
        general = self._config.get("general", {})
        basename = general.get("basename", "")
        if not basename:
            return  # No project name yet, skip sync

        # Use storage if available
        if self._storage:
            self._sync_namelists_with_storage()
        else:
            # Fallback to file-based sync
            output_dir = self.get_output_dir()
            openbench_root = self._project_root or get_openbench_root()

            try:
                # Sync data source namelists
                self._config_manager.sync_namelists(
                    self._config, output_dir, openbench_root
                )
                # Also cleanup unused files
                self._config_manager.cleanup_unused_namelists(self._config, output_dir)

                # Save main config file to nml folder
                self._save_main_config(output_dir, basename, openbench_root)
            except Exception as e:
                # Log error but don't crash
                print(f"Warning: Failed to sync namelists: {e}")

    def _sync_namelists_with_storage(self):
        """Sync namelists using ProjectStorage."""
        general = self._config.get("general", {})
        basename = general.get("basename", "config")
        output_dir = self.get_output_dir()
        openbench_root = self._project_root or get_openbench_root()

        # Get remote OpenBench path if in remote mode
        remote_openbench_path = None
        if self.is_remote_mode():
            remote_config = self._config.get("general", {}).get("remote", {})
            remote_openbench_path = remote_config.get("openbench_path")

        # Generate YAML content
        main_content = self._config_manager.generate_main_nml(
            self._config, openbench_root, output_dir, remote_openbench_path
        )
        ref_content = self._config_manager.generate_ref_nml(
            self._config, openbench_root, output_dir
        )
        sim_content = self._config_manager.generate_sim_nml(
            self._config, openbench_root, output_dir
        )

        # Calculate the relative path from storage root to output_dir
        # This ensures files are written to the case output directory, not OpenBench/nml
        storage_root = self._storage.project_dir
        if self.is_remote_mode():
            # Remote mode: use forward slashes
            storage_root = storage_root.rstrip('/').replace('\\', '/')
            output_dir_clean = output_dir.rstrip('/').replace('\\', '/')
            if output_dir_clean.startswith(storage_root):
                rel_path = output_dir_clean[len(storage_root):].lstrip('/')
            else:
                rel_path = output_dir_clean.lstrip('/')
            nml_path = f"{rel_path}/nml" if rel_path else "nml"
        else:
            # Local mode: use os.path
            try:
                rel_path = os.path.relpath(output_dir, storage_root)
            except ValueError:
                # Different drives on Windows
                rel_path = output_dir
            nml_path = os.path.join(rel_path, "nml")

        # Write via storage to the case output directory
        try:
            self._storage.mkdir(nml_path)
            if self.is_remote_mode():
                self._storage.write_file(f"{nml_path}/main-{basename}.yaml", main_content)
                self._storage.write_file(f"{nml_path}/ref-{basename}.yaml", ref_content)
                self._storage.write_file(f"{nml_path}/sim-{basename}.yaml", sim_content)
            else:
                self._storage.write_file(os.path.join(nml_path, f"main-{basename}.yaml"), main_content)
                self._storage.write_file(os.path.join(nml_path, f"ref-{basename}.yaml"), ref_content)
                self._storage.write_file(os.path.join(nml_path, f"sim-{basename}.yaml"), sim_content)
        except Exception as e:
            print(f"Warning: Failed to sync namelists: {e}")

    def _save_main_config(self, output_dir: str, basename: str, openbench_root: str):
        """Save the main config file to the nml folder."""
        import os

        nml_dir = os.path.join(output_dir, "nml")
        os.makedirs(nml_dir, exist_ok=True)

        main_path = os.path.join(nml_dir, f"main-{basename}.yaml")
        main_content = self._config_manager.generate_main_nml(
            self._config, openbench_root, output_dir
        )

        with open(main_path, 'w', encoding='utf-8') as f:
            f.write(main_content)

    def get_combined_metrics_scores_selection(self) -> Dict[str, bool]:
        """Get combined selection from metrics and scores pages."""
        metrics = self._config.get("metrics", {})
        scores = self._config.get("scores", {})
        return {**metrics, **scores}

"""OpenBench GUI application bootstrap."""

import os
import sys


def launch(config_path=None):
    """Launch the OpenBench GUI application.

    Args:
        config_path: Optional path to an openbench.yaml to load on startup.
    """
    from openbench.gui import _check_gui_deps

    _check_gui_deps()

    from PySide6.QtWidgets import QApplication

    # X11 forwarding support
    if sys.platform != "win32":
        os.environ.setdefault("LIBGL_ALWAYS_INDIRECT", "1")
        os.environ.setdefault("QT_QUICK_BACKEND", "software")

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("OpenBench")
    app.setOrganizationName("CoLM-SYSU")

    # Load stylesheet
    style_dir = os.path.join(os.path.dirname(__file__), "styles")
    theme_path = os.path.join(style_dir, "theme.qss")
    checkmark_path = os.path.join(style_dir, "checkmark.png")

    if os.path.exists(theme_path):
        with open(theme_path) as f:
            stylesheet = f.read()
        stylesheet = stylesheet.replace("CHECKMARK_PATH", checkmark_path.replace("\\", "/"))
        app.setStyleSheet(stylesheet)

    from openbench.gui.main_window import MainWindow

    window = MainWindow()
    if config_path:
        window._load_config_file(config_path)
    window.show()

    # No auto-scan on startup — user triggers scan via Reference Data page button

    sys.exit(app.exec())


def _auto_discover_datasets(window):
    """Scan for new reference datasets and prompt user to register them."""
    # Check common data locations
    data_roots = []

    # Check environment variable first
    env_root = os.environ.get("OPENBENCH_DATA_ROOT")
    if env_root:
        data_roots.append(env_root)

    # Common locations
    data_roots.extend([
        os.path.expanduser("~/data/Reference"),
        os.path.expanduser("~/Reference"),
        "/Volumes/work/Reference",
    ])

    # Also check config for data_root
    try:
        from platformdirs import user_config_dir
        from pathlib import Path
        import yaml

        config_path = Path(user_config_dir("openbench")) / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                user_cfg = yaml.safe_load(f) or {}
            if user_cfg.get("data_root"):
                data_roots.insert(0, user_cfg["data_root"])
    except Exception:
        pass

    for ref_root in data_roots:
        if not os.path.isdir(ref_root):
            continue

        try:
            from openbench.data.registry.scanner import find_new_datasets, register_scanned_dataset

            new_groups = find_new_datasets(ref_root)
            if not new_groups:
                continue

            from openbench.gui.dialogs.data_discovery import DataDiscoveryDialog

            dlg = DataDiscoveryDialog(new_groups, parent=window)
            if dlg.exec():
                selected = dlg.get_selected()
                registered = 0

                # Try to find existing descriptors to merge variable metadata
                from openbench.data.registry.manager import RegistryManager
                mgr = RegistryManager()

                for base_name, res_name, variant in selected:
                    existing = mgr.get_reference(base_name)
                    existing_dict = None
                    if existing:
                        # Convert to dict for merging
                        existing_dict = {
                            "variables": {
                                vn: {"varname": vm.varname, "varunit": vm.varunit,
                                     "prefix": vm.prefix, "suffix": vm.suffix}
                                for vn, vm in existing.variables.items()
                            }
                        }

                    register_scanned_dataset(variant, existing_descriptor=existing_dict)
                    registered += 1

                if registered > 0:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.information(
                        window,
                        "Datasets Registered",
                        f"Registered {registered} dataset(s) from {ref_root}.\n\n"
                        "They are now available in the Reference Data page.",
                    )

            break  # Only scan the first available data root
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Auto-discovery failed: %s", e)

# Sub-project 7: GUI Migration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the wizard GUI from openbench-wizard into `src/openbench/gui/` so `openbench gui` launches the graphical interface. Migrate as-is first (fix imports), page simplification (11→7) deferred.

**Architecture:** Copy all 30 UI files, fix imports from `core.*` → `openbench.remote.*` / `openbench.config.*`, copy styles/resources. The `openbench gui` command bootstraps the PySide6 app.

**Tech Stack:** PySide6, PyYAML

**Working Directory:** `/Volumes/Data01/Openbench`

---

### Task 1: Copy GUI Files and Fix Imports

**Files:**
- Copy: 30 .py files from openbench-wizard/ui/ → src/openbench/gui/
- Copy: styles directory
- Also need: path_utils, config_manager, wizard_config, data_validator, validation from wizard core

- [ ] **Step 1: Copy all GUI files preserving directory structure**

```bash
cd /Volumes/Data01/Openbench

# Main GUI files
cp openbench-wizard/ui/main_window.py src/openbench/gui/main_window.py
cp openbench-wizard/ui/wizard_controller.py src/openbench/gui/controller.py

# Pages
mkdir -p src/openbench/gui/pages
cp openbench-wizard/ui/pages/*.py src/openbench/gui/pages/

# Widgets
mkdir -p src/openbench/gui/widgets
cp openbench-wizard/ui/widgets/*.py src/openbench/gui/widgets/

# Dialogs
mkdir -p src/openbench/gui/dialogs
cp openbench-wizard/ui/dialogs/*.py src/openbench/gui/dialogs/

# Styles
mkdir -p src/openbench/gui/styles
cp openbench-wizard/ui/styles/* src/openbench/gui/styles/

# Supporting core modules needed by GUI
cp openbench-wizard/core/path_utils.py src/openbench/gui/path_utils.py
cp openbench-wizard/core/config_manager.py src/openbench/gui/config_manager.py
cp openbench-wizard/core/wizard_config.py src/openbench/gui/wizard_config.py
cp openbench-wizard/core/data_validator.py src/openbench/gui/data_validator.py
cp openbench-wizard/core/validation.py src/openbench/gui/validation.py
cp openbench-wizard/core/runner.py src/openbench/gui/runner.py
cp openbench-wizard/core/remote_runner.py src/openbench/gui/remote_runner.py
```

- [ ] **Step 2: Fix imports in ALL copied files**

In every .py file under `src/openbench/gui/`, apply these replacements:

Core module references → new locations:
- `from core.ssh_manager` → `from openbench.remote.ssh`
- `from core.credential_manager` → `from openbench.remote.credentials`
- `from core.connection_manager` → `from openbench.remote.connections`
- `from core.sync_engine` → `from openbench.remote.sync`
- `from core.storage` → `from openbench.remote.storage`
- `from core.path_utils` → `from openbench.gui.path_utils`
- `from core.config_manager` → `from openbench.gui.config_manager`
- `from core.wizard_config` → `from openbench.gui.wizard_config`
- `from core.data_validator` → `from openbench.gui.data_validator`
- `from core.validation` → `from openbench.gui.validation`
- `from core.runner` → `from openbench.gui.runner`
- `from core.remote_runner` → `from openbench.gui.remote_runner`

UI internal references:
- `from ui.wizard_controller` → `from openbench.gui.controller`
- `from ui.main_window` → `from openbench.gui.main_window`
- `from ui.pages.` → `from openbench.gui.pages.`
- `from ui.pages ` → `from openbench.gui.pages `
- `from ui.widgets.` → `from openbench.gui.widgets.`
- `from ui.widgets ` → `from openbench.gui.widgets `
- `from ui.dialogs.` → `from openbench.gui.dialogs.`
- `from ui.styles` → fix to use pkg_resources or __file__ for style paths

- [ ] **Step 3: Create `src/openbench/gui/app.py`**

```python
"""OpenBench GUI application bootstrap."""

import sys
import os


def launch(config_path=None):
    """Launch the OpenBench GUI application.

    Args:
        config_path: Optional path to an openbench.yaml to load on startup.
    """
    from openbench.gui import _check_gui_deps
    _check_gui_deps()

    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

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
        window.load_config_file(config_path)
    window.show()

    sys.exit(app.exec())
```

- [ ] **Step 4: Update `src/openbench/cli/gui.py`**

```python
"""openbench gui command."""

import click


@click.command()
@click.argument("config", type=click.Path(exists=True), required=False)
@click.option("--remote", is_flag=True, help="Start in remote mode.")
def gui(config, remote):
    """Launch the OpenBench graphical interface."""
    try:
        from openbench.gui import _check_gui_deps
        _check_gui_deps()
    except ImportError as e:
        raise click.ClickException(str(e))

    from openbench.gui.app import launch
    launch(config_path=config)
```

- [ ] **Step 5: Run existing tests (GUI code shouldn't break non-GUI tests)**

```bash
pytest tests/ -v --tb=short
```

- [ ] **Step 6: Commit**

```bash
git add src/openbench/gui/ src/openbench/cli/gui.py
git commit -m "feat(gui): migrate wizard GUI from openbench-wizard"
```

---

### Task 2: Final Verification

- [ ] **Step 1: Lint**

```bash
ruff check src/ tests/
```

Add per-file ignores for GUI if needed.

- [ ] **Step 2: Tests**

```bash
pytest tests/ -v
```

- [ ] **Step 3: Tag**

```bash
git add -A
git commit -m "chore: SP7 cleanup"
git tag -a v3.0.0a7 -m "Sub-project 7 complete: GUI migration"
```

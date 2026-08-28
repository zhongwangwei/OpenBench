"""Controller-aware registry access for local and remote GUI sessions."""

from __future__ import annotations

import json
from typing import Any

from openbench.data.registry.manager import _auto_resolve_variant, _build_model, _build_reference
from openbench.data.registry.schema import ModelProfile, ReferenceDataset
from openbench.util.names import normalize_name

_REMOTE_CACHE: dict[tuple, "RemoteRegistrySnapshot"] = {}


_SNAPSHOT_SCRIPT = """
import json
from openbench.data.registry.manager import get_registry

registry = get_registry()
print(json.dumps({
    "references": [ref.to_dict() for ref in registry.list_references()],
    "models": [model.to_dict() for model in registry.list_models()],
    "model_aliases": getattr(registry, "_model_aliases", {}),
}, ensure_ascii=False))
"""


def _remote_crud_script(action: str, name: str, payload: dict | None = None) -> str:
    return f"""
import json
from openbench.data.registry.manager import get_registry, _build_model, _build_reference

registry = get_registry()
action = {action!r}
name = {name!r}
payload = json.loads({json.dumps(payload or {})!r})
if action == "save_reference":
    registry.save_reference(name, _build_reference(payload))
elif action == "delete_reference":
    registry.delete_reference(name)
elif action == "save_model":
    registry.save_model(name, _build_model(payload))
elif action == "delete_model":
    registry.delete_model(name)
else:
    raise ValueError("unknown registry action: %s" % action)
print(json.dumps({{"ok": True}}, ensure_ascii=False))
"""


def _is_remote_controller(controller) -> bool:
    checker = getattr(controller, "is_remote_mode", None)
    if callable(checker):
        return bool(checker())
    return False


def _remote_settings(controller) -> dict[str, Any]:
    getter = getattr(controller, "remote_settings", None)
    if callable(getter):
        return getter() or {}
    return {}


def _ssh_manager(controller):
    if hasattr(controller, "storage"):
        from openbench.gui.path_utils import get_remote_ssh_manager

        ssh = get_remote_ssh_manager(controller)
    else:  # lightweight controller doubles used outside the GUI
        ssh = getattr(controller, "ssh_manager", None)
    if ssh is None:
        raise RuntimeError("Remote registry requires a connected SSH manager")
    return ssh


def _target_identity(ssh_manager) -> tuple:
    getter = getattr(ssh_manager, "get_active_target_identity", None)
    identity = getter() if callable(getter) else None
    if identity is None:
        raise RuntimeError("Remote registry requires an active SSH target")
    return tuple(identity)


def _remote_bootstrap(openbench_path: str) -> str:
    if not openbench_path:
        return ""
    root = openbench_path.rstrip("/")
    return (
        "import os\n"
        "import sys\n"
        f"for _path in ({json.dumps(root)}, {json.dumps(root + '/src')}):\n"
        "    _path = os.path.expanduser(_path)\n"
        "    if _path not in sys.path:\n"
        "        sys.path.insert(0, _path)\n"
    )


def _remote_json(controller, script: str):
    from openbench.gui import remote_python

    settings = _remote_settings(controller)
    script = _remote_bootstrap(settings.get("openbench_path", "")) + script
    return remote_python.run_remote_python_json(
        _ssh_manager(controller),
        script,
        python_path=settings.get("python_path", ""),
        conda_env=settings.get("conda_env", ""),
    )


def _load_remote_snapshot(controller, identity: tuple) -> "RemoteRegistrySnapshot":
    payload = _remote_json(controller, _SNAPSHOT_SCRIPT)
    return RemoteRegistrySnapshot(controller, identity, payload)


def _build_remote_reference(data: dict) -> ReferenceDataset:
    """Rehydrate remote data without resolving its paths against local settings."""
    payload = dict(data)
    root_dir = payload.pop("root_dir", None)
    fulllist = payload.pop("fulllist", None)
    reference = _build_reference(payload)
    reference.root_dir = root_dir
    reference.fulllist = fulllist
    return reference


class RemoteRegistrySnapshot:
    """Read-only in-memory registry mirror with remote write-through methods."""

    RESOLUTION_SUFFIXES = ("_LowRes", "_MidRes", "_HigRes")
    REFERENCE_ALIASES: dict[str, str] = {}

    def __init__(self, controller, identity: tuple, payload: dict[str, Any]):
        self._controller = controller
        self._identity = identity
        self.last_resolve_reason = ""
        self._references: dict[str, ReferenceDataset] = {}
        self._models: dict[str, ModelProfile] = {}
        self._model_aliases: dict[str, str] = {}
        self._var_index: dict[str, list[str]] = {}
        self._replace(payload)

    @property
    def target_identity(self) -> tuple:
        return self._identity

    def _replace(self, payload: dict[str, Any]) -> None:
        self._references.clear()
        self._models.clear()
        for item in payload.get("references", []) if isinstance(payload, dict) else []:
            ref = _build_remote_reference(item)
            self._references[normalize_name(ref.name)] = ref
        for item in payload.get("models", []) if isinstance(payload, dict) else []:
            model = _build_model(item)
            self._models[normalize_name(model.name)] = model
        self._model_aliases = {
            normalize_name(alias): normalize_name(target)
            for alias, target in (payload.get("model_aliases", {}) if isinstance(payload, dict) else {}).items()
        }
        self._build_var_index()

    def _build_var_index(self) -> None:
        self._var_index.clear()
        for key, ref in self._references.items():
            for var_name in ref.variables:
                self._var_index.setdefault(normalize_name(var_name), []).append(key)

    def _ensure_current_target(self) -> None:
        current = _target_identity(_ssh_manager(self._controller))
        if current != self._identity:
            raise RuntimeError("Remote registry target changed; refresh the controller registry")

    def refresh(self) -> "RemoteRegistrySnapshot":
        self._ensure_current_target()
        fresh = _load_remote_snapshot(self._controller, self._identity)
        self._replace(
            {
                "references": [ref.to_dict() for ref in fresh.list_references()],
                "models": [model.to_dict() for model in fresh.list_models()],
                "model_aliases": fresh._model_aliases,
            }
        )
        return self

    def list_references(self) -> list[ReferenceDataset]:
        return sorted(self._references.values(), key=lambda ref: ref.name)

    def get_reference(
        self,
        name: str,
        sim_tim_res: str | None = None,
        sim_grid_res: float | None = None,
    ) -> ReferenceDataset | None:
        key = normalize_name(name)
        key = self.REFERENCE_ALIASES.get(key, key)
        if key in self._references:
            self.last_resolve_reason = ""
            return self._references[key]

        variants = self.get_resolution_variants(key)
        if not variants:
            self.last_resolve_reason = ""
            return None
        if sim_tim_res is None and sim_grid_res is None:
            self.last_resolve_reason = ""
            return None
        ref, reason = _auto_resolve_variant(variants, sim_tim_res=sim_tim_res, sim_grid_res=sim_grid_res)
        self.last_resolve_reason = reason
        return ref

    def get_resolution_variants(self, base_name: str) -> dict[str, ReferenceDataset]:
        variants = {}
        base_key = normalize_name(base_name)
        base_key = self.REFERENCE_ALIASES.get(base_key, base_key)
        for suffix in self.RESOLUTION_SUFFIXES:
            full_key = f"{base_key}{normalize_name(suffix)}"
            if full_key in self._references:
                variants[suffix[1:]] = self._references[full_key]
        if base_key in self._references and not variants:
            variants["default"] = self._references[base_key]
        return variants

    def references_for_variable(self, variable: str) -> list[ReferenceDataset]:
        return [self._references[key] for key in self._var_index.get(normalize_name(variable), []) if key in self._references]

    def list_models(self) -> list[ModelProfile]:
        return sorted(self._models.values(), key=lambda model: model.name)

    def get_model(self, name: str) -> ModelProfile | None:
        key = normalize_name(name)
        result = self._models.get(key)
        if result is not None:
            return result
        alias = self._model_aliases.get(key)
        return self._models.get(alias) if alias else None

    def save_reference(self, name: str, dataset: ReferenceDataset) -> None:
        self._ensure_current_target()
        _remote_json(self._controller, _remote_crud_script("save_reference", name, dataset.to_dict()))
        self.refresh()

    def delete_reference(self, name: str) -> None:
        self._ensure_current_target()
        _remote_json(self._controller, _remote_crud_script("delete_reference", name))
        self.refresh()

    def save_model(self, name: str, profile: ModelProfile) -> None:
        self._ensure_current_target()
        _remote_json(self._controller, _remote_crud_script("save_model", name, profile.to_dict()))
        self.refresh()

    def delete_model(self, name: str) -> None:
        self._ensure_current_target()
        _remote_json(self._controller, _remote_crud_script("delete_model", name))
        self.refresh()


def get_registry(controller=None, *, refresh: bool = False):
    """Return the local RegistryManager or a cached remote snapshot for ``controller``."""
    if controller is None or not _is_remote_controller(controller):
        from openbench.data.registry.manager import get_registry as get_local_registry

        return get_local_registry()

    ssh = _ssh_manager(controller)
    identity = _target_identity(ssh)
    if refresh or identity not in _REMOTE_CACHE:
        _REMOTE_CACHE[identity] = _load_remote_snapshot(controller, identity)
    return _REMOTE_CACHE[identity]


def refresh_registry(controller=None):
    """Force reload the registry backing ``controller`` and return it."""
    if controller is None or not _is_remote_controller(controller):
        from openbench.data.registry.manager import clear_registry_cache, get_registry as get_local_registry

        clear_registry_cache()
        return get_local_registry()
    return get_registry(controller, refresh=True)


def clear_registry(controller=None) -> None:
    """Clear local or remote registry cache for ``controller``."""
    if controller is None or not _is_remote_controller(controller):
        from openbench.data.registry.manager import clear_registry_cache

        clear_registry_cache()
        return
    try:
        identity = _target_identity(_ssh_manager(controller))
    except RuntimeError:
        _REMOTE_CACHE.clear()
        return
    _REMOTE_CACHE.pop(identity, None)

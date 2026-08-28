import ast
import json
from types import SimpleNamespace

import pytest

from openbench.data.registry.schema import ModelProfile, ReferenceDataset, VariableMapping


def _ref(name="RemoteRef_LowRes", variable="Runoff"):
    return ReferenceDataset(
        name=name,
        description="remote ref",
        category="Water",
        data_type="grid",
        tim_res="Day",
        data_groupby="Year",
        timezone=0,
        years=[2000, 2001],
        variables={variable: VariableMapping(varname="ro", varunit="mm day-1")},
        grid_res=0.5,
        root_dir="/remote/ref",
    )


def _model(name="RemoteModel"):
    return ModelProfile(
        name=name,
        description="remote model",
        data_type="grid",
        tim_res="Day",
        variables={"Runoff": VariableMapping(varname="mrro", varunit="mm day-1")},
    )


class FakeSSH:
    is_connected = True

    def __init__(self, identity):
        self.identity = identity
        self.calls = []

    def get_active_target_identity(self):
        return self.identity


class FakeController:
    def __init__(self, ssh, settings=None, remote=True):
        self.ssh_manager = ssh
        self._settings = settings or {}
        self._remote = remote

    def is_remote_mode(self):
        return self._remote

    def remote_settings(self):
        return dict(self._settings)


def _snapshot(refs=None, models=None):
    if refs is None:
        refs = [_ref()]
    if models is None:
        models = [_model()]
    return {
        "references": [ref.to_dict() for ref in refs],
        "models": [model.to_dict() for model in models],
    }


def test_local_provider_returns_existing_local_registry(monkeypatch):
    from openbench.data.registry import manager as local_manager
    from openbench.gui import remote_registry

    sentinel = object()
    monkeypatch.setattr(local_manager, "get_registry", lambda: sentinel)

    assert remote_registry.get_registry(SimpleNamespace(is_remote_mode=lambda: False)) is sentinel


def test_remote_snapshot_does_not_call_local_registry_or_write_paths(monkeypatch):
    from openbench.data.registry import manager as local_manager
    from openbench.gui import remote_python, remote_registry

    remote_registry.clear_registry(FakeController(FakeSSH(("direct", "old"))))
    ssh = FakeSSH(("direct", "alice", "login", 22))
    controller = FakeController(
        ssh,
        {"python_path": "~/venv/bin/python", "conda_env": "ob", "openbench_path": "~/OpenBench"},
    )
    calls = []

    def fake_run(ssh_manager, script, **kwargs):
        calls.append((ssh_manager, script, kwargs))
        return _snapshot()

    monkeypatch.setattr(local_manager, "get_registry", lambda: (_ for _ in ()).throw(AssertionError("local read")))
    monkeypatch.setattr(
        local_manager,
        "get_writable_reference_catalog_path",
        lambda: (_ for _ in ()).throw(AssertionError("local write path")),
    )
    monkeypatch.setattr(remote_python, "run_remote_python_json", fake_run)

    registry = remote_registry.get_registry(controller)

    assert registry.list_references()[0].name == "RemoteRef_LowRes"
    assert registry.get_reference("RemoteRef_LowRes").root_dir == "/remote/ref"
    assert registry.references_for_variable("runoff")[0].name == "RemoteRef_LowRes"
    assert registry.get_resolution_variants("RemoteRef") == {"LowRes": registry.get_reference("RemoteRef_LowRes")}
    assert registry.get_model("RemoteModel").variables["Runoff"].varname == "mrro"
    assert calls[0][0] is ssh
    assert calls[0][2] == {"python_path": "~/venv/bin/python", "conda_env": "ob"}
    assert "sys.path.insert" in calls[0][1]
    assert "~/OpenBench/src" in calls[0][1]


def test_remote_snapshot_never_expands_paths_from_local_environment(monkeypatch):
    from openbench.gui.remote_registry import RemoteRegistrySnapshot

    monkeypatch.setenv("OPENBENCH_REF_ROOT", "/local/ref")
    reference = _ref().to_dict()
    reference["root_dir"] = "${OPENBENCH_REF_ROOT}/Grid/RemoteRef"
    reference["fulllist"] = "$OPENBENCH_REF_ROOT/stations.csv"

    registry = RemoteRegistrySnapshot(
        FakeController(FakeSSH(("direct", "alice", "login", 22))),
        ("direct", "alice", "login", 22),
        {"references": [reference], "models": []},
    )

    remote_ref = registry.get_reference("RemoteRef_LowRes")
    assert remote_ref.root_dir == "${OPENBENCH_REF_ROOT}/Grid/RemoteRef"
    assert remote_ref.fulllist == "$OPENBENCH_REF_ROOT/stations.csv"


def test_remote_cache_is_bound_to_active_target_identity(monkeypatch):
    from openbench.gui import remote_python, remote_registry

    remote_registry._REMOTE_CACHE.clear()
    ssh_a = FakeSSH(("direct", "alice", "login-a", 22))
    ssh_b = FakeSSH(("direct", "alice", "login-b", 22))
    controller_a = FakeController(ssh_a)
    controller_b = FakeController(ssh_b)
    calls = []

    def fake_run(ssh_manager, script, **kwargs):
        calls.append(ssh_manager.identity)
        name = "ARef_LowRes" if ssh_manager is ssh_a else "BRef_LowRes"
        return _snapshot(refs=[_ref(name)])

    monkeypatch.setattr(remote_python, "run_remote_python_json", fake_run)

    reg_a1 = remote_registry.get_registry(controller_a)
    reg_a2 = remote_registry.get_registry(controller_a)
    reg_b = remote_registry.get_registry(controller_b)

    assert reg_a1 is reg_a2
    assert reg_a1 is not reg_b
    assert [ref.name for ref in reg_a1.list_references()] == ["ARef_LowRes"]
    assert [ref.name for ref in reg_b.list_references()] == ["BRef_LowRes"]
    assert calls == [ssh_a.identity, ssh_b.identity]

def test_remote_cache_is_bound_to_execution_context(monkeypatch):
    from openbench.gui import remote_python, remote_registry

    remote_registry._REMOTE_CACHE.clear()
    ssh = FakeSSH(("direct", "alice", "login", 22))
    controller_a = FakeController(
        ssh,
        {"python_path": "/envs/a/bin/python", "conda_env": "a", "openbench_path": "/opt/OpenBenchA"},
    )
    controller_b = FakeController(
        ssh,
        {"python_path": "/envs/b/bin/python", "conda_env": "b", "openbench_path": "/opt/OpenBenchB"},
    )
    calls = []

    def fake_run(ssh_manager, script, **kwargs):
        calls.append((ssh_manager.identity, kwargs, script))
        name = "ARef_LowRes" if kwargs["python_path"] == "/envs/a/bin/python" else "BRef_LowRes"
        return _snapshot(refs=[_ref(name)])

    monkeypatch.setattr(remote_python, "run_remote_python_json", fake_run)

    reg_a1 = remote_registry.get_registry(controller_a)
    reg_a2 = remote_registry.get_registry(controller_a)
    reg_b = remote_registry.get_registry(controller_b)

    assert reg_a1 is reg_a2
    assert reg_a1 is not reg_b
    assert [ref.name for ref in reg_a1.list_references()] == ["ARef_LowRes"]
    assert [ref.name for ref in reg_b.list_references()] == ["BRef_LowRes"]
    assert [call[1] for call in calls] == [
        {"python_path": "/envs/a/bin/python", "conda_env": "a"},
        {"python_path": "/envs/b/bin/python", "conda_env": "b"},
    ]
    assert "/opt/OpenBenchA/src" in calls[0][2]
    assert "/opt/OpenBenchB/src" in calls[1][2]

def test_stale_remote_snapshot_refuses_write_after_execution_context_switch(monkeypatch):
    from openbench.gui import remote_python, remote_registry

    remote_registry._REMOTE_CACHE.clear()
    ssh = FakeSSH(("direct", "alice", "login", 22))
    settings = {"python_path": "/envs/a/bin/python", "conda_env": "a", "openbench_path": "/opt/OpenBenchA"}
    controller = FakeController(ssh, settings)
    calls = []

    def fake_run(ssh_manager, script, **kwargs):
        calls.append(kwargs)
        return _snapshot()

    monkeypatch.setattr(remote_python, "run_remote_python_json", fake_run)

    registry = remote_registry.get_registry(controller)
    controller._settings = {"python_path": "/envs/b/bin/python", "conda_env": "b", "openbench_path": "/opt/OpenBenchB"}

    with pytest.raises(RuntimeError, match="target or execution context changed"):
        registry.save_reference("RemoteRef_LowRes", _ref())

    assert calls == [{"python_path": "/envs/a/bin/python", "conda_env": "a"}]

def test_clear_remote_cache_for_target_removes_all_contexts(monkeypatch):
    from openbench.gui import remote_python, remote_registry

    remote_registry._REMOTE_CACHE.clear()
    ssh = FakeSSH(("direct", "alice", "login", 22))
    other_ssh = FakeSSH(("direct", "bob", "login", 22))
    calls = []

    def fake_run(ssh_manager, script, **kwargs):
        calls.append((ssh_manager.identity, kwargs))
        return _snapshot(refs=[_ref(f"Ref{len(calls)}_LowRes")])

    monkeypatch.setattr(remote_python, "run_remote_python_json", fake_run)
    remote_registry.get_registry(FakeController(ssh, {"python_path": "/a", "openbench_path": "/oa"}))
    remote_registry.get_registry(FakeController(ssh, {"python_path": "/b", "openbench_path": "/ob"}))
    other = remote_registry.get_registry(FakeController(other_ssh, {"python_path": "/c", "openbench_path": "/oc"}))

    remote_registry.clear_remote_cache_for_target(ssh)

    remote_registry.get_registry(FakeController(ssh, {"python_path": "/a", "openbench_path": "/oa"}))
    other_again = remote_registry.get_registry(
        FakeController(other_ssh, {"python_path": "/c", "openbench_path": "/oc"})
    )
    assert other_again is other
    assert len(calls) == 4

def test_remote_registry_rejects_manager_when_controller_is_not_remote_storage():
    from openbench.gui import remote_registry

    controller = FakeController(FakeSSH(("direct", "alice", "login", 22)))
    controller.storage = object()

    with pytest.raises(RuntimeError, match="connected SSH manager"):
        remote_registry.get_registry(controller, refresh=True)


def test_remote_save_delete_execute_on_remote_and_refresh_same_snapshot(monkeypatch):
    from openbench.gui import remote_python, remote_registry

    remote_registry._REMOTE_CACHE.clear()
    ssh = FakeSSH(("direct", "alice", "login", 22))
    controller = FakeController(ssh)
    scripts = []
    snapshots = [
        _snapshot(refs=[_ref("Before_LowRes")], models=[_model("BeforeModel")]),
        {"ok": True},
        _snapshot(refs=[_ref("AfterSave_LowRes")], models=[_model("BeforeModel")]),
        {"ok": True},
        _snapshot(refs=[], models=[_model("BeforeModel")]),
        {"ok": True},
        _snapshot(refs=[], models=[_model("AfterModel")]),
        {"ok": True},
        _snapshot(refs=[], models=[]),
    ]

    def fake_run(ssh_manager, script, **kwargs):
        scripts.append(script)
        return snapshots.pop(0)

    monkeypatch.setattr(remote_python, "run_remote_python_json", fake_run)

    registry = remote_registry.get_registry(controller)
    assert [ref.name for ref in registry.list_references()] == ["Before_LowRes"]

    registry.save_reference("AfterSave_LowRes", _ref("AfterSave_LowRes"))
    assert [ref.name for ref in registry.list_references()] == ["AfterSave_LowRes"]

    registry.delete_reference("AfterSave_LowRes")
    assert registry.list_references() == []

    registry.save_model("AfterModel", _model("AfterModel"))
    assert [model.name for model in registry.list_models()] == ["AfterModel"]

    registry.delete_model("AfterModel")
    assert registry.list_models() == []

    joined = "\n".join(scripts)
    assert "registry.save_reference" in joined
    assert "registry.delete_reference" in joined
    assert "registry.save_model" in joined
    assert "registry.delete_model" in joined
    assert remote_registry.get_registry(controller) is registry


def test_stale_remote_snapshot_refuses_write_after_target_switch(monkeypatch):
    from openbench.gui import remote_python, remote_registry

    remote_registry._REMOTE_CACHE.clear()
    ssh = FakeSSH(("direct", "alice", "login-a", 22))
    controller = FakeController(ssh)
    calls = []

    def fake_run(ssh_manager, script, **kwargs):
        calls.append(script)
        return _snapshot()

    monkeypatch.setattr(remote_python, "run_remote_python_json", fake_run)

    registry = remote_registry.get_registry(controller)
    ssh.identity = ("direct", "alice", "login-b", 22)

    with pytest.raises(RuntimeError, match="target or execution context changed"):
        registry.save_reference("RemoteRef_LowRes", _ref())

    assert len(calls) == 1

def test_stale_remote_snapshot_refuses_reads_after_target_switch(monkeypatch):
    from openbench.gui import remote_python, remote_registry

    remote_registry._REMOTE_CACHE.clear()
    ssh = FakeSSH(("direct", "alice", "login-a", 22))
    controller = FakeController(ssh)
    monkeypatch.setattr(remote_python, "run_remote_python_json", lambda *args, **kwargs: _snapshot())

    registry = remote_registry.get_registry(controller)
    ssh.identity = ("direct", "alice", "login-b", 22)

    readers = [
        lambda: registry.list_references(),
        lambda: registry.get_reference("RemoteRef_LowRes"),
        lambda: registry.get_resolution_variants("RemoteRef"),
        lambda: registry.references_for_variable("Runoff"),
        lambda: registry.list_models(),
        lambda: registry.get_model("RemoteModel"),
    ]
    for read in readers:
        with pytest.raises(RuntimeError, match="target or execution context changed"):
            read()

def test_remote_crud_payload_is_a_mapping():
    from openbench.gui.remote_registry import _remote_crud_script

    script = _remote_crud_script("save_model", "RemoteModel", _model().to_dict())
    tree = ast.parse(script)
    payload_assign = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "payload" for target in node.targets)
    )
    encoded = ast.literal_eval(payload_assign.value.args[0])

    assert isinstance(json.loads(encoded), dict)
    compile(script, "<remote-registry-crud>", "exec")


def test_remote_pages_select_remote_registry_without_local_fallback(monkeypatch):
    from openbench.gui import remote_registry
    from openbench.gui.pages import page_ref_data, page_registry, page_sim_data

    sentinel = SimpleNamespace(
        list_models=lambda: [_model()],
        list_references=lambda: [_ref()],
        get_model=lambda name: _model(name),
    )
    controller = SimpleNamespace(is_remote_mode=lambda: True)
    monkeypatch.setattr(remote_registry, "get_registry", lambda current: sentinel)
    monkeypatch.setattr(
        page_registry,
        "_get_registry",
        lambda: (_ for _ in ()).throw(AssertionError("local registry fallback")),
    )

    registry_page = SimpleNamespace(controller=controller)
    ref_page = SimpleNamespace(controller=controller)
    sim_page = SimpleNamespace(controller=controller)

    assert page_registry.PageRegistry._registry(registry_page) is sentinel
    assert page_ref_data.PageRefData._registry(ref_page) is sentinel
    assert page_sim_data.PageSimData._registry_model_names(sim_page) == ["RemoteModel"]
    assert page_sim_data.PageSimData._registry_model_variables(sim_page, "RemoteModel") == ["Runoff"]


def test_remote_ref_page_preserves_saved_config_when_registry_is_offline(monkeypatch):
    from openbench.gui.pages.page_ref_data import PageRefData
    from openbench.remote.storage import RemoteStorage
    from tests.gui_fakes import FakeLineEdit

    class Controller:
        def __init__(self):
            self.config = {
                "evaluation_items": {"Runoff": True},
                "ref_data": {
                    "general": {"Runoff_ref_source": "RemoteRef"},
                    "source_configs": {
                        "Runoff::RemoteRef": {
                            "general": {"root_dir": "/remote/ref"},
                            "varname": "remote_runoff",
                        }
                    },
                },
            }
            self.storage = RemoteStorage("/remote/project", SimpleNamespace())
            self.ssh_manager = None

        def is_remote_mode(self):
            return True

        def update_section(self, section, data):
            self.config[section] = data

        def sync_namelists(self):
            pass

    class Label:
        def setText(self, text):
            self.text = text

    page = PageRefData.__new__(PageRefData)
    page.controller = Controller()
    page.registry_label = Label()
    page.data_root_input = FakeLineEdit("")
    page._source_configs = {}
    page._var_combos = {}
    page._var_advanced_fields = {}
    page._rebuild_variable_groups = lambda: None
    monkeypatch.setattr(
        "openbench.data.registry.manager.get_registry",
        lambda: (_ for _ in ()).throw(AssertionError("local registry read")),
    )

    PageRefData.load_from_config(page)

    saved = page.controller.config["ref_data"]["source_configs"]["Runoff::RemoteRef"]
    assert saved["general"]["root_dir"] == "/remote/ref"
    assert saved["varname"] == "remote_runoff"
    assert "unavailable" in page.registry_label.text

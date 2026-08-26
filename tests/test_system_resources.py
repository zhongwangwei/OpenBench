import logging
from types import SimpleNamespace

import openbench.data._system_resources as resource_module


def test_get_system_resources_uses_defaults_quietly_without_psutil(monkeypatch, caplog):
    monkeypatch.setattr(resource_module, "psutil", None)
    monkeypatch.setattr(resource_module, "_get_macos_cpu_freq", lambda: 0)
    monkeypatch.setattr(resource_module, "_get_linux_cpu_freq", lambda: 0)
    monkeypatch.setattr(resource_module, "_get_windows_cpu_freq", lambda: 0)

    with caplog.at_level(logging.WARNING):
        resources = resource_module.get_system_resources()

    assert resources["total_memory_gb"] == 8
    assert resources["available_memory_gb"] == 4
    assert resources["cpu_count"] == 4
    assert "NoneType" not in caplog.text
    assert "Failed to get memory info" not in caplog.text
    assert "Failed to get CPU count" not in caplog.text


def test_get_system_resources_uses_psutil_when_available(monkeypatch):
    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(total=16 * 1024**3, available=10 * 1024**3),
        cpu_count=lambda logical=False: 6 if not logical else 12,
        cpu_freq=lambda: SimpleNamespace(max=3200, current=3100),
    )
    monkeypatch.setattr(resource_module, "psutil", fake_psutil)

    resources = resource_module.get_system_resources()

    assert resources["total_memory_gb"] == 16
    assert resources["available_memory_gb"] == 10
    assert resources["cpu_count"] == 6
    assert resources["cpu_freq_mhz"] == 3200


def test_effective_cpu_count_caps_host_count_to_affinity(monkeypatch):
    monkeypatch.setattr(resource_module.os, "sched_getaffinity", lambda _pid: {0, 1}, raising=False)
    monkeypatch.setattr(resource_module, "_cgroup_cpu_limit", lambda: None)

    assert resource_module.effective_cpu_count(16) == 2


def test_effective_cpu_count_caps_host_count_to_cgroup(monkeypatch):
    monkeypatch.delattr(resource_module.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(resource_module, "_cgroup_cpu_limit", lambda: 3)

    assert resource_module.effective_cpu_count(16) == 3


def test_worker_thread_limit_updates_loaded_blas_and_environment(monkeypatch):
    import threadpoolctl

    limiter = object()
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", lambda limits: limiter)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")

    resource_module.limit_native_threads()

    assert resource_module._NATIVE_THREAD_LIMITER is limiter
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        assert resource_module.os.environ[name] == "1"


def test_project_num_cores_respects_effective_cpu_limit(monkeypatch):
    import openbench.runner.dask_runtime as dask_runtime

    monkeypatch.setattr(dask_runtime.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(dask_runtime, "effective_cpu_count", lambda _count: 2)
    cfg = SimpleNamespace(project=SimpleNamespace(num_cores=8))

    assert dask_runtime.project_num_cores(cfg) == 2

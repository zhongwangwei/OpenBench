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

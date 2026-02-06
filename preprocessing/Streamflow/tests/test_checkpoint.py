"""Tests for checkpoint system with HMAC-signed JSON."""

from src.utils.checkpoint import CheckpointManager


def test_save_and_load_step(tmp_path):
    mgr = CheckpointManager(tmp_path)
    mgr.save_step("validate", {"stations_processed": 100})
    result = mgr.load_step("validate")
    assert result["stations_processed"] == 100


def test_get_completed_steps(tmp_path):
    mgr = CheckpointManager(tmp_path)
    mgr.save_step("download", {"ok": True})
    mgr.save_step("validate", {"ok": True})
    completed = mgr.get_completed_steps()
    assert "download" in completed
    assert "validate" in completed
    assert "cama" not in completed


def test_get_resume_point(tmp_path):
    mgr = CheckpointManager(tmp_path)
    mgr.save_step("download", {})
    mgr.save_step("validate", {})
    assert mgr.get_resume_step() == "cama"


def test_empty_checkpoint(tmp_path):
    mgr = CheckpointManager(tmp_path)
    assert mgr.get_completed_steps() == set()
    assert mgr.get_resume_step() == "download"

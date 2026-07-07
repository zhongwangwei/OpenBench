from pathlib import Path

import pytest

from openbench.remote.storage import LocalStorage


def test_local_storage_rejects_symlink_escape_on_write(tmp_path: Path):
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (project / "linked").symlink_to(outside, target_is_directory=True)

    storage = LocalStorage(str(project))

    with pytest.raises(ValueError, match="escapes project directory"):
        storage.write_file("linked/leak.txt", "secret")

    assert not (outside / "leak.txt").exists()


def test_local_storage_rejects_symlink_escape_on_read(tmp_path: Path):
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    (project / "linked").symlink_to(outside, target_is_directory=True)

    storage = LocalStorage(str(project))

    with pytest.raises(ValueError, match="escapes project directory"):
        storage.read_file("linked/secret.txt")

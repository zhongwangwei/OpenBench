"""Tests for docs.manual.scripts.lib.io."""
from pathlib import Path

from docs.manual.scripts.lib.io import write_generated


class TestWriteGenerated:
    def test_writes_with_warning_header(self, tmp_path: Path):
        target = tmp_path / "out.tex"
        body = r"\section{测试}"
        write_generated(target, body, source="reference_catalog.yaml")
        text = target.read_text()
        # 头注释告诉读者别手改
        assert "% AUTO-GENERATED" in text
        assert "reference_catalog.yaml" in text
        # body 必须完整出现
        assert r"\section{测试}" in text
        # 末尾留有换行
        assert text.endswith("\n")

    def test_overwrites_existing(self, tmp_path: Path):
        target = tmp_path / "out.tex"
        target.write_text("OLD CONTENT")
        write_generated(target, "NEW", source="x.py")
        assert "OLD CONTENT" not in target.read_text()
        assert "NEW" in target.read_text()

    def test_creates_parent_dir(self, tmp_path: Path):
        target = tmp_path / "deep" / "nested" / "out.tex"
        write_generated(target, "x", source="x.py")
        assert target.exists()

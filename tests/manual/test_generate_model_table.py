"""Tests for generate_model_table."""
from pathlib import Path

from docs.manual.scripts.generate_model_table import build_model_table


class TestBuildModelTable:
    def test_lists_all_models(self, minimal_model_catalog: Path):
        out = build_model_table(minimal_model_catalog)
        assert "CoLM2024" in out

    def test_includes_var_count(self, minimal_model_catalog: Path):
        out = build_model_table(minimal_model_catalog)
        # CoLM2024 fixture 有 2 个变量
        assert "2" in out

    def test_includes_data_type(self, minimal_model_catalog: Path):
        out = build_model_table(minimal_model_catalog)
        assert "grid" in out.lower()

    def test_empty_catalog(self, tmp_path: Path):
        catalog = tmp_path / "empty.yaml"
        catalog.write_text("{}\n")
        out = build_model_table(catalog)
        assert r"\begin{longtable}" in out or "无模型" in out

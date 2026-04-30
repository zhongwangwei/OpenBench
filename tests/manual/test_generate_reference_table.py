"""Tests for generate_reference_table."""
from pathlib import Path

from docs.manual.scripts.generate_reference_table import build_reference_table


class TestBuildReferenceTable:
    def test_groups_by_category(self, minimal_reference_catalog: Path):
        out = build_reference_table(minimal_reference_catalog)
        # 至少出现 "Water" 分类（minimal fixture 中两个都属 Water）
        assert "Water" in out
        # GLEAM 与 FLUXNET 都应被列出
        assert "GLEAM" in out
        assert "FLUXNET" in out

    def test_distinguishes_grid_vs_stn(self, minimal_reference_catalog: Path):
        out = build_reference_table(minimal_reference_catalog)
        # GLEAM 是 grid，FLUXNET 是 stn —— 输出中能找到 "grid" 与 "stn" 字样
        assert "grid" in out.lower() or "Grid" in out
        assert "stn" in out.lower() or "Station" in out

    def test_includes_variable_count(self, minimal_reference_catalog: Path):
        out = build_reference_table(minimal_reference_catalog)
        # GLEAM 有 2 个变量（ET、Transpiration），FLUXNET 有 1 个（ET）
        # 不强制具体格式，但数字应在某处出现
        assert "2" in out

    def test_escapes_special_chars(self, tmp_path: Path):
        # 构造一个含下划线的数据集名
        catalog = tmp_path / "ref.yaml"
        catalog.write_text(
            "weird_name_with_underscores:\n"
            "  name: weird_name_with_underscores\n"
            "  category: Test\n"
            "  data_type: grid\n"
            "  variables: {}\n"
        )
        out = build_reference_table(catalog)
        # 下划线在表格单元格中必须被转义
        assert r"weird\_name\_with\_underscores" in out

    def test_empty_catalog(self, tmp_path: Path):
        catalog = tmp_path / "empty.yaml"
        catalog.write_text("{}\n")
        out = build_reference_table(catalog)
        # 空 catalog 不该爆，输出至少包含一个 longtable 空骨架或说明文本
        assert r"\begin{longtable}" in out or "无数据集" in out

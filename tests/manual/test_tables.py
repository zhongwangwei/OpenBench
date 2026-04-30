"""Tests for docs.manual.scripts.lib.tables."""
import pytest

from docs.manual.scripts.lib.tables import longtable


class TestLongtable:
    def test_basic_three_column(self):
        out = longtable(
            headers=["字段", "类型", "默认值"],
            rows=[
                ["name", "str", "---"],
                ["years", "list[int]", "---"],
            ],
            col_spec="l l l",
        )
        # 必含元素
        assert r"\begin{longtable}{l l l}" in out
        assert r"\toprule" in out
        assert r"\endhead" in out
        assert r"\bottomrule" in out
        assert r"\end{longtable}" in out
        assert "字段 & 类型 & 默认值" in out
        assert "name & str & ---" in out

    def test_escapes_cell_content(self):
        out = longtable(
            headers=["key", "value"],
            rows=[["proj_name", "100%"]],
            col_spec="l l",
        )
        # 单元格里的 _ 与 % 必须被转义
        assert r"proj\_name" in out
        assert r"100\%" in out
        # 但 & 是分隔符，不能误转
        assert " & " in out

    def test_caption_optional(self):
        out = longtable(
            headers=["a", "b"],
            rows=[["1", "2"]],
            col_spec="l l",
            caption="测试表格",
        )
        assert r"\caption{测试表格}" in out

    def test_empty_rows_renders(self):
        # 空 rows 也要产出有效骨架，避免 LaTeX 报错
        out = longtable(headers=["a", "b"], rows=[], col_spec="l l")
        assert r"\begin{longtable}" in out
        assert r"\end{longtable}" in out

    def test_column_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="column count"):
            longtable(
                headers=["a", "b"],
                rows=[["1", "2", "3"]],  # 3 列但 header 只有 2 列
                col_spec="l l",
            )

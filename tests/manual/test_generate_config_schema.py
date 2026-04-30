"""Tests for generate_config_schema."""
from docs.manual.scripts.generate_config_schema import build_config_schema_table


class TestBuildConfigSchemaTable:
    def test_lists_top_level_dataclasses(self):
        out = build_config_schema_table()
        # 关键 dataclass 名应作为 subsection
        for cls in ("ProjectConfig", "EvaluationConfig", "ReferenceConfig", "OpenBenchConfig"):
            assert cls in out

    def test_lists_required_fields(self):
        out = build_config_schema_table()
        # ProjectConfig 必填字段；下划线被 LaTeX 转义为 \_
        assert "name" in out
        assert r"output\_dir" in out
        assert "years" in out

    def test_lists_optional_with_default(self):
        out = build_config_schema_table()
        # tim_res 默认 None；转义后形如 tim\_res
        assert r"tim\_res" in out

    def test_extracts_inline_comment(self):
        out = build_config_schema_table()
        # ProjectConfig.time_alignment 有注释 "intersection | per_pair | strict"
        assert "intersection" in out

    def test_escapes_special_chars(self):
        out = build_config_schema_table()
        # 字段名 IGBP_groupby 含下划线
        assert r"IGBP\_groupby" in out

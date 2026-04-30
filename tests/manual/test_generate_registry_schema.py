"""Tests for generate_registry_schema."""
from docs.manual.scripts.generate_registry_schema import build_registry_schema_table


class TestBuildRegistrySchemaTable:
    def test_lists_key_dataclasses(self):
        out = build_registry_schema_table()
        # registry/schema.py 含 FallbackVar、VariableMapping 等
        assert "VariableMapping" in out

    def test_lists_fields(self):
        out = build_registry_schema_table()
        # VariableMapping.varname / varunit
        assert "varname" in out
        assert "varunit" in out

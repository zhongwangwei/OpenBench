"""Tests for generate_internal_interfaces."""
from docs.manual.scripts.generate_internal_interfaces import build_interfaces_doc


class TestBuildInterfacesDoc:
    def test_includes_known_abc(self):
        out = build_interfaces_doc()
        # util/interfaces.py 含 IOutputManager（ABC）
        assert "IOutputManager" in out

    def test_lists_subsection_per_class(self):
        out = build_interfaces_doc()
        # 至少 IOutputManager 有抽象方法被列出
        assert r"\subsection*{IOutputManager}" in out

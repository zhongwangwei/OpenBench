"""Tests for docs.manual.scripts.lib.escape."""
import pytest

from docs.manual.scripts.lib.escape import latex_escape


class TestLatexEscape:
    def test_passthrough_plain_text(self):
        assert latex_escape("hello world") == "hello world"

    def test_escapes_underscore(self):
        assert latex_escape("var_name") == r"var\_name"

    def test_escapes_dollar(self):
        assert latex_escape("$5") == r"\$5"

    def test_escapes_percent(self):
        assert latex_escape("50%") == r"50\%"

    def test_escapes_ampersand(self):
        assert latex_escape("a & b") == r"a \& b"

    def test_escapes_hash(self):
        assert latex_escape("#1") == r"\#1"

    def test_escapes_braces(self):
        assert latex_escape("{x}") == r"\{x\}"

    def test_escapes_backslash(self):
        assert latex_escape(r"a\b") == r"a\textbackslash{}b"

    def test_escapes_caret_and_tilde(self):
        assert latex_escape("a^b~c") == r"a\^{}b\~{}c"

    def test_combined(self):
        # 真实场景：YAML key 含下划线 + 单位字符串
        assert latex_escape("kg/m^2/s & 100%") == r"kg/m\^{}2/s \& 100\%"

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            latex_escape(42)

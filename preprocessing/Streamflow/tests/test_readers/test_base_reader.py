"""Tests for base reader ABC and reader factory."""

import pytest
from src.readers.base_reader import BaseReader
from src.readers import READERS, register_reader, get_reader


def test_cannot_instantiate_base():
    with pytest.raises(TypeError):
        BaseReader()


def test_concrete_reader():
    class FakeReader(BaseReader):
        source_name = "fake"

        def read_all(self, config):
            return []

    reader = FakeReader()
    assert reader.source_name == "fake"
    assert reader.read_all({}) == []


def test_register_and_get_reader():
    @register_reader("test_dummy")
    class DummyReader(BaseReader):
        source_name = "test_dummy"

        def read_all(self, config):
            return []

    assert "test_dummy" in READERS
    reader = get_reader("test_dummy")
    assert isinstance(reader, DummyReader)


def test_get_unknown_reader():
    with pytest.raises(ValueError, match="Unknown source"):
        get_reader("nonexistent_source_xyz")

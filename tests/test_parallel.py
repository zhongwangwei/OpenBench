import pytest

from openbench.util.exceptions import ParallelProcessingError
from openbench.util.parallel import ParallelEngine, get_parallel_engine


def test_threading_parallel_map_raises_on_partial_task_failure():
    engine = ParallelEngine(backend="threading", max_workers=2, show_progress=False)

    def sometimes_fails(value):
        if value == 2:
            raise ValueError("boom")
        return value

    with pytest.raises(ParallelProcessingError, match="task\\(s\\) failed"):
        engine.map(sometimes_fails, [1, 2, 3], task_name="unit-test")


def test_parallel_engine_recreate_uses_shutdown(monkeypatch):
    import openbench.util.parallel as parallel_mod

    calls = []

    class FakeEngine:
        def __init__(self, backend, max_workers):
            self.backend = backend
            self.max_workers = max_workers

        def shutdown(self):
            calls.append("shutdown")

    monkeypatch.setattr(parallel_mod, "ParallelEngine", FakeEngine)
    monkeypatch.setattr(parallel_mod, "_parallel_engine", None)

    get_parallel_engine("threading", 1)
    get_parallel_engine("threading", 2)

    assert calls == ["shutdown"]

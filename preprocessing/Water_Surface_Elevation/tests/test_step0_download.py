import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.steps.step0_download import Step0Download, DataStatus

def test_check_completeness_complete():
    step = Step0Download(config={'data_root': '/tmp'})
    with patch.object(Path, 'exists', return_value=True):
        with patch.object(Path, 'glob', return_value=[MagicMock() for _ in range(2500)]):
            status = step._check_source_completeness('hydrosat', Path('/tmp/Hydrosat'))
    assert status.is_complete

def test_check_completeness_incomplete():
    step = Step0Download(config={'data_root': '/tmp'})
    with patch.object(Path, 'exists', return_value=True):
        with patch.object(Path, 'glob', return_value=[MagicMock() for _ in range(100)]):
            status = step._check_source_completeness('hydrosat', Path('/tmp/Hydrosat'))
    assert not status.is_complete

def test_check_completeness_missing():
    step = Step0Download(config={'data_root': '/tmp'})
    with patch.object(Path, 'exists', return_value=False):
        status = step._check_source_completeness('hydrosat', Path('/tmp/Hydrosat'))
    assert not status.is_complete
    assert status.current_files == 0

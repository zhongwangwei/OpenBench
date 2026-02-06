# tests/test_logger.py
import logging
from src.utils.logger import setup_logger, get_logger, ColoredFormatter, ProgressLogger

def test_setup_logger():
    logger = setup_logger("test", log_level="DEBUG")
    assert isinstance(logger, logging.Logger)
    assert logger.level == logging.DEBUG

def test_progress_logger(capsys):
    logger = setup_logger("test_progress", log_level="INFO")
    pl = ProgressLogger(logger, total=200, prefix="Reading: ")
    pl.update(100)
    pl.update(200)
    pl.done()

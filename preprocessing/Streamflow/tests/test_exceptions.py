from src.exceptions import StreamflowError, ReaderError, ValidationError, ConfigurationError, DownloadError

def test_exception_hierarchy():
    assert issubclass(ReaderError, StreamflowError)
    assert issubclass(ValidationError, StreamflowError)
    assert issubclass(ConfigurationError, StreamflowError)
    assert issubclass(DownloadError, StreamflowError)

def test_exception_message():
    e = ReaderError("test message")
    assert str(e) == "test message"

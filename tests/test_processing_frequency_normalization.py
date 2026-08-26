import pandas as pd

from openbench.data._processing_config import USE_NEW_FREQ_ALIASES, ProcessingConfigMixin


def test_compound_month_frequency_is_pandas_compatible():
    processor = ProcessingConfigMixin()
    normalized = processor._normalize_frequency("3month")

    assert normalized == ("3ME" if USE_NEW_FREQ_ALIASES else "3M")
    pd.tseries.frequencies.to_offset(normalized)


def test_normalized_month_end_frequency_is_idempotent():
    processor = ProcessingConfigMixin()

    assert processor._normalize_frequency("ME") == "ME"

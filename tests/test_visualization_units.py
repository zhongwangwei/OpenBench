from openbench.visualization.Fig_toolbox import process_unit


def test_process_unit_accepts_nrmse_display_casing():
    assert process_unit("mm", "mm", "nRMSE") == "(-)"


def test_process_unit_unknown_metric_does_not_crash():
    assert process_unit("mm", "mm", "custom_metric") == "(-)"


def test_process_unit_knows_every_implemented_metric(caplog):
    import logging

    from openbench.core.registry import IMPLEMENTED_METRIC_NAMES

    with caplog.at_level(logging.WARNING, logger="openbench.visualization.Fig_toolbox"):
        for metric in IMPLEMENTED_METRIC_NAMES:
            process_unit("mm", "mm", metric)

    assert "Unknown metric unit" not in caplog.text

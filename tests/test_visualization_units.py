from openbench.visualization.Fig_toolbox import process_unit


def test_process_unit_accepts_nrmse_display_casing():
    assert process_unit("mm", "mm", "nRMSE") == "(-)"


def test_process_unit_unknown_metric_does_not_crash():
    assert process_unit("mm", "mm", "custom_metric") == "(-)"

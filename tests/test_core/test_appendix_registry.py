from openbench.core.registry import IMPLEMENTED_METRICS, METRIC_LABELS, METRICS_ITEMS


def test_appendix_scalar_metrics_are_selectable_and_labeled():
    expected = {
        "MSE",
        "NRMSE",
        "RSR",
        "RSS",
        "NMAE",
        "rSD",
        "PBIAS_HF",
        "PBIAS_LF",
        "pbiasfdc",
        "rSpearman",
        "MIA",
        "RIA",
        "VE",
        "LNSE",
        "mNSE",
        "rNSE",
        "mKGE",
        "KGEkm",
        "KGElf",
    }
    gui_metrics = {name for values in METRICS_ITEMS.values() for name in values}

    assert expected <= IMPLEMENTED_METRICS
    assert expected <= gui_metrics
    assert expected <= METRIC_LABELS.keys()


def test_appendix_methods_requiring_extra_inputs_are_not_pairwise_gui_metrics():
    gui_metrics = {name for values in METRICS_ITEMS.values() for name in values}

    special = {"valindex", "wNSE", "wsNSE", "sKGE", "KGEnp"}
    assert special.isdisjoint(IMPLEMENTED_METRICS)
    assert special.isdisjoint(gui_metrics)

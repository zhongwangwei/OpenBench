import itertools


def test_limited_product_caps_default_balanced_combinations():
    from openbench.visualization._combinations import limited_product

    groups = [range(10), range(10), range(10)]

    out = list(limited_product(groups, {}))

    assert len(out) == 100
    assert out == list(itertools.islice(itertools.product(*groups), 100))


def test_limited_product_respects_explicit_max_combinations():
    from openbench.visualization._combinations import limited_product

    out = list(limited_product([["a", "b"], [1, 2, 3]], {"max_combinations": 3}))

    assert out == [("a", 1), ("a", 2), ("a", 3)]


def test_limited_product_full_mode_keeps_all_combinations_without_explicit_limit():
    from openbench.visualization._combinations import limited_product

    out = list(limited_product([["a", "b"], [1, 2, 3]], {"plotting_mode": "full"}))

    assert out == list(itertools.product(["a", "b"], [1, 2, 3]))


def test_limited_product_fast_mode_uses_tighter_default():
    from openbench.visualization._combinations import limited_product

    out = list(limited_product([range(10), range(10)], {"plotting_mode": "fast"}))

    assert len(out) == 25


def test_default_combo_heavy_fig_options_expose_limits():
    from openbench.config.adapter import build_fig_nml

    fig_nml = build_fig_nml()

    for section_name in ("Parallel_Coordinates", "Portrait_Plot_seasonal"):
        section = fig_nml["Comparison"][section_name]
        assert section["plotting_mode"] == "balanced"
        assert section["max_combinations"] == 100

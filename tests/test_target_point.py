import numpy as np

from openbench.core._comparison_target import _target_point


def test_target_point_is_the_point_of_the_pooled_stations():
    rng = np.random.default_rng(0)
    ref = rng.normal(50.0, 20.0, size=(4, 24))
    sim = ref * rng.uniform(0.6, 1.4, size=(4, 1)) + rng.normal([[-15.0], [10.0], [2.0], [-5.0]], 8.0, size=(4, 24))
    diff = sim - ref
    bias = diff.mean(axis=1)
    rmse = np.sqrt((diff**2).mean(axis=1))
    crmsd = np.sqrt(((diff - bias[:, None]) ** 2).mean(axis=1))

    mean_bias, rmsd, centred = _target_point(bias, rmse)

    pooled = diff.ravel()
    np.testing.assert_allclose(mean_bias, pooled.mean())
    np.testing.assert_allclose(rmsd, np.sqrt((pooled**2).mean()))
    np.testing.assert_allclose(centred, pooled.std())
    np.testing.assert_allclose(rmsd**2, mean_bias**2 + centred**2)
    # the plain means of the three statistics do not close
    assert not np.isclose(rmse.mean() ** 2, bias.mean() ** 2 + crmsd.mean() ** 2, rtol=1e-3)


def test_target_point_skips_missing_and_is_nan_without_samples():
    mean_bias, rmsd, centred = _target_point([3.0, np.nan, -1.0], [5.0, 2.0, np.nan])
    assert (mean_bias, rmsd, centred) == (3.0, 5.0, 4.0)
    assert all(np.isnan(_target_point([np.nan], [np.nan])))

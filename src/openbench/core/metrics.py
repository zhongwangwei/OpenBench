# -*- coding: utf-8 -*-
import logging
from collections.abc import Mapping

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress  # used by br2 metric

# Import CacheSystem - CacheSystem is mandatory for metrics calculation
try:
    from openbench.data.cache import cached, get_cache_manager  # noqa: F401  feature detection

    _HAS_CACHE = True
except ImportError:
    raise RuntimeError(
        "CacheSystem is required for metrics calculation (务必使用CacheSystem). "
        "Please ensure openbench.data.cache is available."
    )


class metrics:
    """
    A class for calculating various statistical metrics for model evaluation.
    """

    def __init__(self):
        """
        Initialize the Metrics class with metadata.
        """
        self.name = "metrics"
        self.version = "0.2"
        self.release = "0.2"
        self.date = "March 2024"
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"

        # NOTE: We deliberately do NOT call `np.seterr(all="ignore")`
        # here. That is a process-wide setting and would silence
        # legitimate runtime warnings in unrelated code (and tests).
        # Individual metrics that need to suppress divide-by-zero or
        # invalid-value warnings use `xr.where(...)` guards or a local
        # `with np.errstate(...)` context.

    def _validate_inputs(self, s, o):
        """
        Validate and align input DataArrays.

        Args:
            s (xr.DataArray): Simulated data
            o (xr.DataArray): Observed data

        Returns:
            tuple: Aligned and validated DataArrays
        """
        # Ensure inputs are xarray DataArrays
        if not isinstance(s, xr.DataArray) or not isinstance(o, xr.DataArray):
            logging.error("Inputs must be xarray DataArrays")
            raise TypeError("Inputs must be xarray DataArrays")

        # Align time dimensions
        s, o = xr.align(s, o, join="inner")

        # Remove NaN values
        mask = np.isfinite(s) & np.isfinite(o)
        return s.where(mask), o.where(mask)

    def percent_bias(self, s, o):
        """
        Calculate Percent Bias using the standard signed observed-sum denominator.

        This metric is intended for variables whose observed aggregate is
        meaningfully non-zero. Sign-changing anomaly or flux series can make
        percent bias unstable or difficult to interpret.

        Args:
            s (xr.DataArray): Simulated data
            o (xr.DataArray): Observed data

        Returns:
            xr.DataArray: Percent bias
        """
        s, o = self._validate_inputs(s, o)
        o_sum = o.sum(dim="time")
        return xr.where(o_sum != 0, 100.0 * (s - o).sum(dim="time") / o_sum, np.nan)

    def absolute_percent_bias(self, s, o):
        """
        Calculate Absolute Percent Bias (APB).

        Args:
            s (xr.DataArray): Simulated data
            o (xr.DataArray): Observed data

        Returns:
            xr.DataArray: Absolute percent bias
        """
        # Validate and align inputs
        s, o = self._validate_inputs(s, o)

        # Calculate absolute percent bias (guard against zero observed sum)
        o_sum = o.sum(dim="time")
        apb = xr.where(o_sum != 0, 100.0 * abs((s - o).sum(dim="time")) / np.abs(o_sum), np.nan)
        return apb

    def RMSE(self, s, o):
        """
        Calculate Root Mean Squared Error (RMSE).

        Args:
            s (xr.DataArray): Simulated data
            o (xr.DataArray): Observed data

        Returns:
            xr.DataArray: Root mean squared error
        """
        # Validate and align inputs
        s, o = self._validate_inputs(s, o)

        # Calculate RMSE
        rmse = np.sqrt(((s - o) ** 2).mean(dim="time"))
        return rmse

    def MSE(self, s, o, dim="time"):
        """Mean square error from the appendix."""
        s, o = self._validate_inputs(s, o)
        return ((s - o) ** 2).mean(dim=dim)

    def NRMSE(self, s, o, dim="time"):
        """RMSE normalized by ``abs(mean(o))`` (appendix NRMSE_mu)."""
        s, o = self._validate_inputs(s, o)
        o_mean = o.mean(dim=dim)
        return xr.where(o_mean != 0, np.sqrt(((s - o) ** 2).mean(dim=dim)) / np.abs(o_mean), np.nan)

    def RSR(self, s, o, dim="time"):
        """RMSE-observations standard deviation ratio."""
        s, o = self._validate_inputs(s, o)
        denom = np.sqrt(((o - o.mean(dim=dim)) ** 2).sum(dim=dim))
        numer = np.sqrt(((s - o) ** 2).sum(dim=dim))
        return xr.where(denom != 0, numer / denom, np.nan)

    def RSS(self, s, o, dim="time"):
        """Residual sum of squares."""
        s, o = self._validate_inputs(s, o)
        return ((s - o) ** 2).sum(dim=dim)

    def NMAE(self, s, o, dim="time"):
        """Normalized mean absolute error using sum(abs(o)) denominator."""
        s, o = self._validate_inputs(s, o)
        denom = np.abs(o).sum(dim=dim)
        return xr.where(denom != 0, np.abs(s - o).sum(dim=dim) / denom, np.nan)

    def ubRMSE(self, s, o):
        """
        Calculate Unbiased Root Mean Squared Error (ubRMSE).

        Args:
            s (xr.DataArray): Simulated data
            o (xr.DataArray): Observed data

        Returns:
            xr.DataArray: Unbiased root mean squared error
        """
        # Validate and align inputs
        s, o = self._validate_inputs(s, o)

        # Calculate unbiased RMSE
        ubrmse = np.sqrt((((s - s.mean(dim="time")) - (o - o.mean(dim="time"))) ** 2).mean(dim="time"))
        return ubrmse

    def CRMSD(self, s, o=None):
        """
        Calculate Centered Root Mean Square Difference (CRMSD).

        Args:
            s (xr.DataArray): Simulated data
            o (xr.DataArray, optional): Observed data. If not provided, the mean along the time dimension is used as the reference.

        Returns:
            xr.DataArray: Centered root mean square difference
        """
        # If observed data is not provided, use the mean of simulated data as reference
        if o is None:
            if not isinstance(s, xr.DataArray):
                logging.error("Input must be an xarray DataArray")
                raise TypeError("Input must be an xarray DataArray")
            if "time" not in s.dims:
                raise ValueError("CRMSD requires a 'time' dimension")
            s = s.where(np.isfinite(s))
            return np.sqrt(((s - s.mean(dim="time")) ** 2).mean(dim="time"))

        # Validate and align inputs
        s, o = self._validate_inputs(s, o)

        # Calculate standard deviations
        std_s = s.std(dim="time")
        std_o = o.std(dim="time")

        # Calculate correlations
        correlations = xr.corr(s, o, dim="time")

        # Apply the CRMSD formula. Clamp the radicand to ≥ 0 — floating-point
        # error can make std_s² + std_o² − 2·std_s·std_o·r slightly negative
        # when std_s ≈ std_o and r ≈ 1, which would otherwise yield NaN.
        radicand = std_s**2 + std_o**2 - 2 * std_s * std_o * correlations
        crmsd = np.sqrt(np.maximum(radicand, 0))
        return crmsd

    def mean_absolute_error(self, s, o):
        """
        Mean Absolute Error
        input:
            s: simulated
            o: observed
        output:
            maes: mean absolute error
        """
        s, o = self._validate_inputs(s, o)
        # np.mean(abs(self.s-self.o))
        k1 = s - o
        var = (abs(k1)).mean(dim="time")
        return var

    def bias(self, s, o):
        """
        Bias
        input:
            s: simulated
            o: observed
        output:
            bias: bias
        """
        s, o = self._validate_inputs(s, o)
        # np.mean(s-o)
        var = (s - o).mean(dim="time")
        return var

    def L(self, s, o, N=5):
        """
        Likelihood
        input:
            s: simulated
            o: observed
        output:
            L: likelihood
        """
        s, o = self._validate_inputs(s, o)
        # np.exp(-N*sum((self.s-self.o)**2)/sum((self.o-np.mean(self.o))**2))
        tmp1 = ((o - o.mean(dim="time")) ** 2).sum(dim="time")
        tmp2 = -N * (((s - o) ** 2).sum(dim="time"))
        # Guard against constant-observation series (tmp1 == 0)
        var = xr.where(tmp1 != 0, np.exp(tmp2 / tmp1), np.nan)
        return var

    def correlation(self, s, o):
        """
        correlation coefficient
        input:
            s: simulated
            o: observed
        output:
            correlation: correlation coefficient
        """
        s, o = self._validate_inputs(s, o)
        corr = xr.corr(s, o, dim=["time"])

        return corr

    def correlation_R2(self, s, o):
        """
        correlation coefficient R2
        input:
            s: simulated
            o: observed
        output:
            correlation: correlation coefficient
        """
        s, o = self._validate_inputs(s, o)
        return (xr.corr(s, o, dim=["time"]) ** 2).clip(min=0, max=1)

    def NSE(self, s, o):
        """
        Nash Sutcliffe efficiency coefficient
        input:
            s: simulated
            o: observed
        output:
            nse: Nash Sutcliffe efficient coefficient
        """
        s, o = self._validate_inputs(s, o)
        # 1 - sum((s-o)**2)/sum((o-np.mean(o))**2)
        _tmp1 = ((o - o.mean(dim="time")) ** 2).sum(dim="time")
        _tmp2 = ((s - o) ** 2).sum(dim="time")
        var = xr.where(_tmp1 != 0, 1 - _tmp2 / _tmp1, np.nan)
        return var

    def KGE(self, s, o):
        """
        Kling-Gupta Efficiency
        input:
            s: simulated
            o: observed
        output:
            kge: Kling-Gupta Efficiency
            cc: correlation
            alpha: ratio of the standard deviation
            beta: ratio of the mean
        """
        s, o = self._validate_inputs(s, o)
        cc = self.correlation(s, o)
        # Guard against constant observation (std=0) and zero-mean observation
        # (e.g. precipitation in a dry month). Without these guards alpha/beta
        # become inf and kge becomes -inf, silently polluting downstream output.
        o_std = o.std(dim="time")
        o_mean = o.mean(dim="time")
        alpha = xr.where(o_std != 0, s.std(dim="time") / o_std, np.nan)
        beta = xr.where(o_mean != 0, s.mean(dim="time") / o_mean, np.nan)
        kge = 1 - ((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) ** 0.5
        return kge  # , cc, alpha, beta

    def KGESS(self, s, o):
        """
        Normalized Kling-Gupta Efficiency
        input:
            s: simulated
            o: observed
        output:
            kgess:Normalized Kling-Gupta Efficiency
        note:
        KGEbench= −0.41 from Knoben et al., 2019)
        Knoben, W. J. M., Freer, J. E., and Woods, R. A.: Technical note: Inherent benchmark or not? Comparing Nash–Sutcliffe and Kling–
        Gupta efficiency scores, Hydrol. Earth Syst. Sci., 23, 4323–4331,
        https://doi.org/10.5194/hess-23-4323-2019, 2019.
        """
        kge = self.KGE(s, o)
        kgess = (kge - (-0.41)) / (1.0 - (-0.41))
        return kgess  # , cc, alpha, beta

    def index_agreement(self, s, o):
        """
            index of agreement
            input:
            s: simulated
            o: observed
        output:
            ia: index of agreement
        """
        from openbench.core.scores import scores

        return scores.index_agreement(self, s, o)

    def kappa_coeff(self, s, o):
        """Calculate Cohen's kappa for integer-coded categorical labels."""
        s, o = xr.align(s, o, join="inner")

        def _kappa_1d(s_values, o_values):
            mask = np.isfinite(s_values) & np.isfinite(o_values)
            s_flat = s_values[mask]
            o_flat = o_values[mask]
            if s_flat.size == 0:
                return np.nan
            s_rounded = np.rint(s_flat)
            o_rounded = np.rint(o_flat)
            if not np.allclose(s_flat, s_rounded, rtol=0.0, atol=1e-8) or not np.allclose(
                o_flat, o_rounded, rtol=0.0, atol=1e-8
            ):
                raise ValueError("kappa_coeff requires integer-coded categorical labels")
            s_flat = s_rounded.astype(int)
            o_flat = o_rounded.astype(int)
            unique_data = np.unique(np.concatenate([s_flat, o_flat]))
            category_count = len(unique_data)
            s_codes = np.searchsorted(unique_data, s_flat)
            o_codes = np.searchsorted(unique_data, o_flat)
            kappa_mat = np.bincount(
                s_codes * category_count + o_codes,
                minlength=category_count**2,
            ).reshape(category_count, category_count)
            total = kappa_mat.sum()
            if total == 0:
                return np.nan
            pa = np.trace(kappa_mat) / total
            pred = kappa_mat.sum(axis=0) / total
            obs = kappa_mat.sum(axis=1) / total
            pe = np.sum(pred * obs)
            if abs(1 - pe) < 1e-10:
                return np.nan
            return (pa - pe) / (1 - pe)

        if "time" in getattr(s, "dims", ()) and "time" in getattr(o, "dims", ()):
            if hasattr(s, "chunks") and s.chunks is not None:
                s = s.chunk({"time": -1})
            if hasattr(o, "chunks") and o.chunks is not None:
                o = o.chunk({"time": -1})
            return xr.apply_ufunc(
                _kappa_1d,
                s,
                o,
                input_core_dims=[["time"], ["time"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
        return xr.DataArray(_kappa_1d(np.asarray(s), np.asarray(o)))

    def rv(self, s, o):
        """
        Relative variability
        (or amplitude ratio)
        input:
            s: simulated
            o: observed
        output:
            rv : relative variability, amplitude ratio
        Reference:
        ****
        """
        s, o = self._validate_inputs(s, o)
        o_std = o.std(dim="time")
        # Protect against division by zero when observed std is 0 or very small.
        return xr.where(o_std != 0, s.std(dim="time") / o_std - 1.0, np.nan)

    def ubNSE(self, s, o):
        """
        Unbiased Nash Sutcliffe efficiency coefficient
        input:
            s: simulated
            o: observed
        output:
            ubnse: Unbiased Nash Sutcliffe efficient coefficient
        """
        s, o = self._validate_inputs(s, o)
        _tmp1 = ((o - o.mean(dim="time")) ** 2).sum(dim="time")
        _tmp2 = (((s - s.mean(dim="time")) - (o - o.mean(dim="time"))) ** 2).sum(dim="time")
        # Mirror the NSE guard above — constant observations would otherwise
        # produce ±inf instead of NaN.
        var = xr.where(_tmp1 != 0, 1 - _tmp2 / _tmp1, np.nan)
        return var

    def ubKGE(self, s, o):
        """
        Unbiased Kling-Gupta Efficiency
        input:
            s: simulated
            o: observed
        output:
            kge: Kling-Gupta Efficiency

        """
        s, o = self.rm_mean(s, o)
        cc = self.correlation(s, o)
        o_std = o.std(dim="time")
        alpha = xr.where(o_std != 0, s.std(dim="time") / o_std, np.nan)
        # With mean-zero inputs beta is undefined (0/0), so ubKGE uses the
        # two-component unbiased form rather than delegating to KGE.
        return 1 - ((cc - 1) ** 2 + (alpha - 1) ** 2) ** 0.5

    def ubcorrelation(self, s, o):
        """
        correlation coefficient
        input:
            s: simulated
            o: observed
        output:
            correlation: correlation coefficient
        """
        s, o = self.rm_mean(s, o)
        var = self.correlation(s, o)
        return var

    def ubcorrelation_R2(self, s, o):
        """
        correlation coefficient R2
        input:
            s: simulated
            o: observed
        output:
            correlation: correlation coefficient
        """
        s, o = self.rm_mean(s, o)
        var = self.correlation_R2(s, o)
        return var

    def rm_mean(self, s, o):
        # Subtract each series' own mean (i.e. "remove bias" so that the
        # series have zero mean). The previous implementation shifted both
        # series by min(s.min, o.min), which is a common-shift, not a mean
        # removal. Note that downstream:
        #   * ubcorrelation / ubcorrelation_R2: correlation is invariant
        #     under a common shift AND under per-series mean removal, so
        #     these return the same value as correlation / correlation_R2.
        #   * ubKGE: with mean-zero inputs, KGE's beta = mean_s / mean_o
        #     becomes 0/0, so ubKGE uses an explicit 2-component
        #     (cc, alpha) reformulation.
        s, o = self._validate_inputs(s, o)
        return s - s.mean(dim="time"), o - o.mean(dim="time")

    def pc_max(self, s, o):
        s, o = self._validate_inputs(s, o)

        o_max = o.max(dim="time")
        return xr.where(o_max != 0, (s.max(dim="time") - o_max) / np.abs(o_max), np.nan)

    def pc_min(self, s, o):
        s, o = self._validate_inputs(s, o)

        # Normalize by |o_min| so a negative observed minimum (e.g. winter
        # temperature minima) doesn't flip the sign of the relative-bias
        # interpretation: a model warmer than obs should always read as a
        # positive deviation, regardless of the absolute reference sign.
        o_min = o.min(dim="time")
        return xr.where(o_min != 0, (s.min(dim="time") - o_min) / np.abs(o_min), np.nan)

    def pc_ampli(self, s, o):
        s, o = self._validate_inputs(s, o)

        # Calculate amplitude (range) for observed data. Keep the guard
        # element-wise and lazy: a global ``np.any`` over a dask-backed grid
        # triggers eager computation during metric graph construction.
        o_range = o.max(dim="time") - o.min(dim="time")
        s_range = s.max(dim="time") - s.min(dim="time")
        safe_o_range = o_range.where((o_range != 0) & o_range.notnull())
        return s_range / safe_o_range - 1.0

    def _chunk_core_dim(self, *arrays, dim="time"):
        out = []
        for array in arrays:
            if hasattr(array, "chunks") and array.chunks is not None and dim in array.dims:
                array = array.chunk({dim: -1})
            out.append(array)
        return out

    def _kge_components(self, s, o, dim="time"):
        s, o = self._validate_inputs(s, o)
        r = xr.corr(s, o, dim=dim)
        s_mean = s.mean(dim=dim)
        o_mean = o.mean(dim=dim)
        s_std = s.std(dim=dim)
        o_std = o.std(dim=dim)
        alpha = xr.where(o_std != 0, s_std / o_std, np.nan)
        beta = xr.where(o_mean != 0, s_mean / o_mean, np.nan)
        cv_s = xr.where(s_mean != 0, s_std / s_mean, np.nan)
        cv_o = xr.where(o_mean != 0, o_std / o_mean, np.nan)
        gamma = xr.where(cv_o != 0, cv_s / cv_o, np.nan)
        return r, alpha, beta, gamma

    def rSD(self, s, o, dim="time"):
        """Ratio of simulated to observed standard deviation."""
        s, o = self._validate_inputs(s, o)
        o_std = o.std(dim=dim)
        return xr.where(o_std != 0, s.std(dim=dim) / o_std, np.nan)

    def PBIAS_HF(self, s, o, quantile=0.98, dim="time"):
        """Percent bias over observed high-flow samples; default threshold is Q98."""
        s, o = self._validate_inputs(s, o)
        s, o = self._chunk_core_dim(s, o, dim=dim)
        threshold = o.quantile(quantile, dim=dim, skipna=True)
        high = o >= threshold
        denom = o.where(high).sum(dim=dim)
        return xr.where(denom != 0, 100.0 * (s - o).where(high).sum(dim=dim) / denom, np.nan)

    def PBIAS_LF(self, s, o, quantile=0.30, dim="time"):
        """Percent bias over observed low-flow samples; default threshold is Q30."""
        s, o = self._validate_inputs(s, o)
        s, o = self._chunk_core_dim(s, o, dim=dim)
        threshold = o.quantile(quantile, dim=dim, skipna=True)
        low = o <= threshold
        selected = o.where(low)
        denom = selected.sum(dim=dim)
        strictly_positive = ((selected > 0) | selected.isnull()).all(dim=dim) & low.any(dim=dim)
        return xr.where(
            strictly_positive & (denom != 0),
            100.0 * (s - o).where(low).sum(dim=dim) / denom,
            np.nan,
        )

    def pbiasfdc(self, s, o, high_quantile=0.66, low_quantile=0.33, dim="time"):
        """Percent bias in the slope of the midsegment of the flow-duration curve."""
        s, o = self._validate_inputs(s, o)
        s, o = self._chunk_core_dim(s, o, dim=dim)
        qs_h = s.quantile(high_quantile, dim=dim, skipna=True)
        qs_l = s.quantile(low_quantile, dim=dim, skipna=True)
        qo_h = o.quantile(high_quantile, dim=dim, skipna=True)
        qo_l = o.quantile(low_quantile, dim=dim, skipna=True)
        valid = (qs_h > 0) & (qs_l > 0) & (qo_h > 0) & (qo_l > 0)
        sim_slope = np.log(qs_h) - np.log(qs_l)
        obs_slope = np.log(qo_h) - np.log(qo_l)
        return xr.where(valid & (obs_slope != 0), 100.0 * (sim_slope - obs_slope) / obs_slope, np.nan)

    def rSpearman(self, s, o, dim="time"):
        """Spearman rank correlation using average ranks for ties."""
        s, o = self._validate_inputs(s, o)
        s, o = self._chunk_core_dim(s, o, dim=dim)

        def _spearman_1d(sim, obs):
            mask = np.isfinite(sim) & np.isfinite(obs)
            if mask.sum() < 2:
                return np.nan
            sim_rank = pd.Series(sim[mask]).rank(method="average").to_numpy()
            obs_rank = pd.Series(obs[mask]).rank(method="average").to_numpy()
            if np.std(sim_rank) == 0 or np.std(obs_rank) == 0:
                return np.nan
            return float(np.corrcoef(sim_rank, obs_rank)[0, 1])

        return xr.apply_ufunc(
            _spearman_1d,
            s,
            o,
            input_core_dims=[[dim], [dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

    def MIA(self, s, o, dim="time"):
        """Modified index of agreement."""
        s, o = self._validate_inputs(s, o)
        o_mean = o.mean(dim=dim)
        denom = (np.abs(s - o_mean) + np.abs(o - o_mean)).sum(dim=dim)
        return xr.where(denom != 0, 1 - np.abs(s - o).sum(dim=dim) / denom, np.nan)

    def RIA(self, s, o, dim="time"):
        """Relative index of agreement for positive observations."""
        s, o = self._validate_inputs(s, o)
        valid_pair = np.isfinite(s) & np.isfinite(o)
        o_mean = o.mean(dim=dim)
        positive = ((o > 0) | ~valid_pair).all(dim=dim) & (valid_pair.sum(dim=dim) > 0) & (o_mean > 0)
        numer = (np.abs(s - o) / o.where(o > 0)).sum(dim=dim)
        denom = ((np.abs(s - o_mean) + np.abs(o - o_mean)) / o_mean).sum(dim=dim)
        return xr.where(positive & (denom != 0), 1 - numer / denom, np.nan)

    def valindex(self, s, o, epsilon=0.0, dim="time"):
        """Fraction of valid pairs with absolute error <= epsilon (default exact match)."""
        s, o = self._validate_inputs(s, o)
        n = np.isfinite(s).sum(dim=dim)
        hits = (np.abs(s - o) <= epsilon).where(np.isfinite(s)).sum(dim=dim)
        return xr.where(n != 0, hits / n, np.nan)

    def VE(self, s, o, dim="time"):
        """Volumetric efficiency for non-negative observed volumes/flows."""
        s, o = self._validate_inputs(s, o)
        valid_pair = np.isfinite(s) & np.isfinite(o)
        denom = o.sum(dim=dim)
        nonnegative = ((o >= 0) | ~valid_pair).all(dim=dim) & (valid_pair.sum(dim=dim) > 0)
        return xr.where(nonnegative & (denom > 0), 1 - np.abs(s - o).sum(dim=dim) / denom, np.nan)

    def LNSE(self, s, o, dim="time"):
        """Log Nash-Sutcliffe efficiency for strictly positive pairs."""
        s, o = self._validate_inputs(s, o)
        valid_pair = np.isfinite(s) & np.isfinite(o)
        positive_pair = (s > 0) & (o > 0)
        valid_domain = (positive_pair | ~valid_pair).all(dim=dim) & (valid_pair.sum(dim=dim) > 0)
        log_s = np.log(s.where(positive_pair))
        log_o = np.log(o.where(positive_pair))
        denom = ((log_o - log_o.mean(dim=dim)) ** 2).sum(dim=dim)
        numer = ((log_s - log_o) ** 2).sum(dim=dim)
        return xr.where(valid_domain & (denom != 0), 1 - numer / denom, np.nan)

    def mNSE(self, s, o, dim="time"):
        """Modified NSE using absolute errors and absolute observed deviations."""
        s, o = self._validate_inputs(s, o)
        denom = np.abs(o - o.mean(dim=dim)).sum(dim=dim)
        return xr.where(denom != 0, 1 - np.abs(s - o).sum(dim=dim) / denom, np.nan)

    def rNSE(self, s, o, dim="time"):
        """Relative NSE; only defined for non-zero observations and observed mean."""
        s, o = self._validate_inputs(s, o)
        valid_pair = np.isfinite(s) & np.isfinite(o)
        o_mean = o.mean(dim=dim)
        nonzero = ((o != 0) | ~valid_pair).all(dim=dim) & (valid_pair.sum(dim=dim) > 0) & (o_mean != 0)
        numer = (((s - o) / o.where(o != 0)) ** 2).sum(dim=dim)
        denom = (((o - o_mean) / o_mean) ** 2).sum(dim=dim)
        return xr.where(nonzero & (denom != 0), 1 - numer / denom, np.nan)

    def wNSE(self, s, o, weights, dim="time"):
        """Weighted NSE using explicit non-negative sample weights."""
        s, o = self._validate_inputs(s, o)
        s, o, weights = xr.align(s, o, weights, join="inner")
        valid_weight_domain = ((weights >= 0) | ~np.isfinite(weights)).all(dim=dim)
        weights = weights.where(np.isfinite(s) & np.isfinite(o) & np.isfinite(weights) & (weights >= 0))
        wsum = weights.sum(dim=dim)
        o_mean = (weights * o).sum(dim=dim) / wsum.where(wsum != 0)
        denom = (weights * (o - o_mean) ** 2).sum(dim=dim)
        numer = (weights * (s - o) ** 2).sum(dim=dim)
        return xr.where(valid_weight_domain & (wsum > 0) & (denom != 0), 1 - numer / denom, np.nan)

    def wsNSE(self, s, o, season_weights, seasons=None, dim="time"):
        """Weighted seasonal NSE using explicit season weights and labels."""
        s, o = self._validate_inputs(s, o)
        if not isinstance(season_weights, Mapping):
            raise TypeError("wsNSE season_weights must map season labels to weights")
        if seasons is None:
            if dim not in o.coords:
                raise ValueError("wsNSE requires explicit seasons or a datetime coordinate")
            try:
                seasons = o[dim].dt.season
            except (AttributeError, TypeError) as exc:
                raise ValueError("wsNSE requires explicit seasons for non-datetime coordinates") from exc
        s, o, seasons = xr.align(s, o, seasons, join="inner")
        seasons = seasons.rename(seasons.name or "season")
        labels = np.asarray(seasons)
        missing = set(np.unique(labels)) - set(season_weights)
        if missing:
            raise ValueError(f"Missing wsNSE weights for seasons: {sorted(missing)}")
        if any(weight < 0 for weight in season_weights.values()):
            raise ValueError("wsNSE season weights must be non-negative")
        weights = xr.DataArray(
            [season_weights[label.item() if hasattr(label, "item") else label] for label in labels],
            coords={dim: o[dim]},
            dims=[dim],
        )
        season_mean = o.groupby(seasons).mean(dim=dim)
        centered = o.groupby(seasons) - season_mean
        denom = (weights * centered**2).sum(dim=dim)
        numer = (weights * (s - o) ** 2).sum(dim=dim)
        return xr.where((weights.sum(dim=dim) > 0) & (denom != 0), 1 - numer / denom, np.nan)

    def mKGE(self, s, o, dim="time"):
        """Modified KGE using coefficient-of-variation ratio gamma."""
        r, _alpha, beta, gamma = self._kge_components(s, o, dim=dim)
        return 1 - ((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2) ** 0.5

    def sKGE(self, s, o, dim="time"):
        """Split KGE components as [r, beta, gamma]."""
        r, _alpha, beta, gamma = self._kge_components(s, o, dim=dim)
        return xr.concat([r, beta, gamma], dim=xr.IndexVariable("component", ["r", "beta", "gamma"]))

    def KGEkm(self, s, o, dim="time"):
        """Known-moments KGE variant with alpha, beta, and CV-ratio gamma."""
        r, alpha, beta, gamma = self._kge_components(s, o, dim=dim)
        return 1 - ((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2) ** 0.5

    def KGElf(self, s, o, quantile=0.30, dim="time"):
        """Low-flow KGE over samples where observed values are <= Q30 by default."""
        s, o = self._validate_inputs(s, o)
        s, o = self._chunk_core_dim(s, o, dim=dim)
        threshold = o.quantile(quantile, dim=dim, skipna=True)
        low = o <= threshold
        r, alpha, beta, _gamma = self._kge_components(s.where(low), o.where(low), dim=dim)
        return 1 - ((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2) ** 0.5

    def KGEnp(self, s, o, dim="time"):
        """Pool et al. non-parametric KGE using normalized flow-duration curves."""
        s, o = self._validate_inputs(s, o)
        s, o = self._chunk_core_dim(s, o, dim=dim)
        rho = self.rSpearman(s, o, dim=dim)

        def _fdc_variability(sim, obs):
            mask = np.isfinite(sim) & np.isfinite(obs)
            sim = sim[mask]
            obs = obs[mask]
            if sim.size < 2 or sim.mean() == 0 or obs.mean() == 0:
                return np.nan
            n = sim.size
            sim_fdc = np.sort(sim)[::-1] / (n * sim.mean())
            obs_fdc = np.sort(obs)[::-1] / (n * obs.mean())
            return 1.0 - 0.5 * np.abs(sim_fdc - obs_fdc).sum()

        alpha_np = xr.apply_ufunc(
            _fdc_variability,
            s,
            o,
            input_core_dims=[[dim], [dim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        o_mean = o.mean(dim=dim)
        beta = xr.where(o_mean != 0, s.mean(dim=dim) / o_mean, np.nan)
        return 1 - ((rho - 1) ** 2 + (alpha_np - 1) ** 2 + (beta - 1) ** 2) ** 0.5

    def APFB(
        self,
        data_array,
        obs_array,
        start_month=1,
        out_per_year=False,
        fun=None,
        epsilon_type="none",
        epsilon_value=None,
    ):
        """
        Calculates the Annual Peak Flow Bias (APFB) along the time dimension of Xarray DataArrays.

        Args:
            data_array (xr.DataArray): Simulated data.
            obs_array (xr.DataArray): Observed data.
            start_month (int, optional): Starting month of the hydrological year (1-12). Defaults to 1 (January).
            out_per_year (bool, optional): If True, returns APFB per year. Defaults to False.
            fun (function, optional): Transformation function to apply to data before calculation. Defaults to None.
            epsilon_type (str, optional): Type of epsilon handling for zero values in 'fun'. Defaults to "none".
            epsilon_value (float, optional): Value for epsilon handling. Defaults to None.

        Returns:
            float or dict: Mean APFB or a dictionary with mean APFB and yearly APFB values.
        """

        # Align and handle missing values. Use inner coordinate alignment
        # rather than selecting sim by every obs timestamp; real model/ref
        # streams can have offset or partial time coverage.
        data_array, obs_array = self._validate_inputs(data_array, obs_array)
        if data_array.sizes.get("time", 0) == 0:
            return {"APFB_value": np.nan, "APFB_per_year": np.nan} if out_per_year else np.nan

        if not out_per_year and "time" in data_array.dims and data_array.ndim > 1:
            if hasattr(data_array, "chunks") and data_array.chunks is not None:
                data_array = data_array.chunk({"time": -1})
            if hasattr(obs_array, "chunks") and obs_array.chunks is not None:
                obs_array = obs_array.chunk({"time": -1})
            time_values = xr.DataArray(
                data_array["time"].values,
                coords={"time": data_array["time"]},
                dims=("time",),
            )

            def _apfb_1d(sim_values, obs_values, times):
                mask = np.isfinite(sim_values) & np.isfinite(obs_values)
                sim_values = sim_values[mask]
                obs_values = obs_values[mask]
                times = times[mask]
                if sim_values.size == 0 or obs_values.size == 0:
                    return np.nan
                if fun is not None:
                    if epsilon_type == "Pushpalatha2012":
                        epsilon = np.nanmean(obs_values) / 100
                    elif epsilon_type == "otherFactor":
                        epsilon = np.nanmean(obs_values) * epsilon_value
                    elif epsilon_type == "otherValue":
                        epsilon = epsilon_value
                    else:
                        epsilon = 0
                    sim_values = fun(sim_values + epsilon)
                    obs_values = fun(obs_values + epsilon)
                try:
                    index = pd.DatetimeIndex(times)
                except Exception:
                    return np.nan
                years = index.year
                if start_month != 1:
                    years = years + (index.month >= start_month).astype(int)
                values = []
                for year in np.intersect1d(np.unique(years), np.unique(years)):
                    year_mask = years == year
                    obs_peak = np.nanmax(obs_values[year_mask])
                    if obs_peak == 0 or np.isnan(obs_peak):
                        continue
                    sim_peak = np.nanmax(sim_values[year_mask])
                    values.append((sim_peak - obs_peak) / obs_peak)
                return float(np.nanmean(values)) if values else np.nan

            return xr.apply_ufunc(
                _apfb_1d,
                data_array,
                obs_array,
                time_values,
                input_core_dims=[["time"], ["time"], ["time"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )

        # Convert to pandas for easier time grouping
        df_sim = data_array.to_pandas().to_frame(name="simulated")
        df_obs = obs_array.to_pandas().to_frame(name="observed")

        # Apply transformation function
        if fun is not None:
            if epsilon_type == "Pushpalatha2012":
                epsilon = df_obs["observed"].mean() / 100
            elif epsilon_type == "otherFactor":
                epsilon = df_obs["observed"].mean() * epsilon_value
            elif epsilon_type == "otherValue":
                epsilon = epsilon_value
            else:
                epsilon = 0

            df_sim["simulated"] = df_sim["simulated"].apply(lambda x: fun(x + epsilon))
            df_obs["observed"] = df_obs["observed"].apply(lambda x: fun(x + epsilon))

        # Group by hydrological year and calculate peak flows. Pandas
        # Period does not support frequencies like "1MS"; compute the water
        # year directly so shifted model/ref timelines still work.
        sim_year = df_sim.index.year
        obs_year = df_obs.index.year
        if start_month != 1:
            sim_year = sim_year + (df_sim.index.month >= start_month).astype(int)
            obs_year = obs_year + (df_obs.index.month >= start_month).astype(int)
        df_sim["year"] = sim_year
        df_obs["year"] = obs_year
        annual_peaks_sim = df_sim.groupby("year")["simulated"].max()
        annual_peaks_obs = df_obs.groupby("year")["observed"].max()

        # Calculate APFB for each year. Guard zero observed peaks (dry
        # years) so the per-year ratio falls to NaN instead of inf, which
        # would otherwise propagate into the mean and silently corrupt
        # the multi-year aggregate.
        apfb_per_year = (annual_peaks_sim - annual_peaks_obs).where(annual_peaks_obs != 0) / annual_peaks_obs.where(
            annual_peaks_obs != 0
        )

        if out_per_year:
            return {"APFB_value": apfb_per_year.mean(), "APFB_per_year": apfb_per_year}
        else:
            return apfb_per_year.mean()

    def br2(self, data_array, obs_array, na_rm=True, use_abs=True, fun=None, epsilon_type="none", epsilon_value=None):
        """
        Calculates the br2 metric (R-squared multiplied by regression slope) along the time dimension.

        Args:
            data_array (xr.DataArray): Simulated data.
            obs_array (xr.DataArray): Observed data.
            na_rm (bool, optional): If True, removes missing values before calculation. Defaults to True.
            use_abs (bool, optional): If True, uses absolute value of slope in calculation. Defaults to True.
            fun (function, optional): Transformation function to apply to data before calculation. Defaults to None.
            epsilon_type (str, optional): Type of epsilon handling for zero values in 'fun'. Defaults to "none".
            epsilon_value (float, optional): Value for epsilon handling. Defaults to None.

        Returns:
            xr.DataArray: An array containing the br2 values for each time step.
        """

        # Align and handle missing values. Do not require identical time
        # axes; compute over the overlap and let the vectorized kernel drop
        # pairwise NaNs.
        data_array, obs_array = self._validate_inputs(data_array, obs_array)

        # Apply transformation function
        if fun is not None:
            if epsilon_type == "Pushpalatha2012":
                epsilon = obs_array.mean(dim="time") / 100
            elif epsilon_type == "otherFactor":
                epsilon = obs_array.mean(dim="time") * epsilon_value
            elif epsilon_type == "otherValue":
                epsilon = epsilon_value
            else:
                epsilon = 0

            data_array = fun(data_array + epsilon)
            obs_array = fun(obs_array + epsilon)

        # Calculate R-squared and regression slope
        def calculate_for_single_time(sim, obs):
            mask = np.isfinite(sim) & np.isfinite(obs)
            sim = sim[mask]
            obs = obs[mask]
            if len(sim) < 2 or len(obs) < 2:
                return np.nan
            if np.nanstd(sim) == 0 or np.nanstd(obs) == 0:
                return np.nan
            r_squared = np.corrcoef(sim, obs)[0, 1] ** 2
            try:
                slope, _, _, _, _ = linregress(obs, sim)  # Force intercept to zero
            except ValueError:
                return np.nan
            # scipy ≥ 1.13 returns NaN (rather than raising) for degenerate
            # inputs; the std==0 guards above should catch most cases, but
            # keep an explicit NaN check before any comparison/arithmetic.
            if np.isnan(slope):
                return np.nan
            if use_abs:
                slope = abs(slope)
            br2_value = r_squared * slope if slope <= 1 else r_squared / slope
            return br2_value

        # Rechunk time dimension to single chunk for apply_ufunc with dask
        if hasattr(data_array, "chunks") and data_array.chunks is not None:
            data_array = data_array.chunk({"time": -1})
        if hasattr(obs_array, "chunks") and obs_array.chunks is not None:
            obs_array = obs_array.chunk({"time": -1})

        br2_values = xr.apply_ufunc(
            calculate_for_single_time,
            data_array,
            obs_array,
            input_core_dims=[["time"], ["time"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        return br2_values

    def cp(self, data_array, obs_array, fun=None, epsilon_type="none", epsilon_value=None):
        """
        Calculates the Coefficient of Persistence (CP) along the time dimension of Xarray DataArrays.

        Args:
            data_array (xr.DataArray): Simulated data.
            obs_array (xr.DataArray): Observed data.
            fun (function, optional): Transformation function to apply to data before calculation. Defaults to None.
            epsilon_type (str, optional): Type of epsilon handling for zero values in 'fun'. Defaults to "none".
            epsilon_value (float, optional): Value for epsilon handling. Defaults to None.

        Returns:
            xr.DataArray: An array containing the CP values for each time step.
        """

        if not isinstance(data_array, xr.DataArray) or not isinstance(obs_array, xr.DataArray):
            raise TypeError("Inputs must be xarray DataArrays")
        data_array, obs_array = xr.align(data_array, obs_array, join="inner")

        # Apply transformation function
        if fun is not None:
            if epsilon_type == "Pushpalatha2012":
                epsilon = obs_array.mean(dim="time") / 100
            elif epsilon_type == "otherFactor":
                epsilon = obs_array.mean(dim="time") * epsilon_value
            elif epsilon_type == "otherValue":
                epsilon = epsilon_value
            else:
                epsilon = 0

            data_array = fun(data_array + epsilon)
            obs_array = fun(obs_array + epsilon)

        # Numerator: model-vs-observation residual squared, summed over the
        # N-1 timesteps for which a persistence baseline exists (t >= 1).
        # Denominator: observed first-difference squared (persistence
        # baseline) over the same N-1 timesteps. Previously the numerator
        # was data_array.diff(dim="time") — that is S_t - S_{t-1}, which is
        # not a model-vs-obs residual and gives an undefined statistic.
        sim_t = data_array.isel(time=slice(1, None))
        obs_t = obs_array.isel(time=slice(1, None))
        obs_prev = obs_array.shift(time=1).isel(time=slice(1, None))
        valid_pairs = np.isfinite(sim_t) & np.isfinite(obs_t) & np.isfinite(obs_prev)

        sim_minus_obs = (sim_t - obs_t).where(valid_pairs)
        diff_obs_obs = (obs_t - obs_prev).where(valid_pairs)

        numerator = (sim_minus_obs**2).sum(dim="time")
        denominator = (diff_obs_obs**2).sum(dim="time")

        cp = xr.where(denominator != 0, 1 - (numerator / denominator), np.nan)
        return cp

    def dr(self, data_array, obs_array, fun=None, epsilon_type="none", epsilon_value=None):
        """
        Calculates the Refined Index of Agreement (dr) along the time dimension of Xarray DataArrays.

        Args:
            data_array (xr.DataArray): Simulated data.
            obs_array (xr.DataArray): Observed data.
            fun (function, optional): Transformation function to apply to data before calculation. Defaults to None.
            epsilon_type (str, optional): Type of epsilon handling for zero values in 'fun'. Defaults to "none".
            epsilon_value (float, optional): Value for epsilon handling. Defaults to None.

        Returns:
            xr.DataArray: An array containing the dr values for each time step.
        """

        # Align on the overlapping timestamps; missing pairs remain NaN and
        # are skipped by xarray reductions below.
        data_array, obs_array = self._validate_inputs(data_array, obs_array)

        # Apply transformation function
        if fun is not None:
            if epsilon_type == "Pushpalatha2012":
                epsilon = obs_array.mean(dim="time") / 100
            elif epsilon_type == "otherFactor":
                epsilon = obs_array.mean(dim="time") * epsilon_value
            elif epsilon_type == "otherValue":
                epsilon = epsilon_value
            else:
                epsilon = 0

            data_array = fun(data_array + epsilon)
            obs_array = fun(obs_array + epsilon)

        # Calculate differences and mean of observations
        diff = np.abs(data_array - obs_array)
        obs_mean = obs_array.mean(dim="time")

        # Calculate terms A and B
        A = diff.sum(dim="time")
        B = 2 * np.abs(obs_array - obs_mean).sum(dim="time")

        # Calculate dr. A constant observed series makes B=0, so the
        # agreement ratio is undefined; returning 1.0 would falsely mark
        # any non-zero model error as perfect agreement.
        with np.errstate(divide="ignore", invalid="ignore"):
            dr = 1 - (A / B)
            dr = xr.where(A > B, (B / A) - 1, dr)  # Handle cases where A > B
        dr = xr.where(B != 0, dr, np.nan)

        return dr

    def smpi(self, s, o, n_bootstrap=100, seed=None):
        # Calculate the Single Model Performance Index (SMPI).
        #
        # The comparison workflow defines SMPI from the climatological mean
        # model-observation difference normalized by observed temporal
        # variance. Keep this API consistent with that path instead of using
        # instantaneous per-time-step differences.
        #
        # `seed` makes the bootstrap reproducible; pass an int (or pre-seeded
        # Generator) for regression tests. Default None keeps prior behavior.
        s, o = self._validate_inputs(s, o)
        obs_var = o.var(dim="time", ddof=1)
        s_climate = s.mean(dim="time")
        o_climate = o.mean(dim="time")

        diff_squared = (s_climate - o_climate) ** 2
        normalized_diff = diff_squared / obs_var.where(obs_var != 0)

        smpi_dims = list(normalized_diff.dims)
        smpi = normalized_diff.mean(dim=smpi_dims, skipna=True) if smpi_dims else normalized_diff

        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        bootstrap_smpi = []
        n_times = s.sizes["time"]
        dask_backed = getattr(s, "chunks", None) is not None or getattr(o, "chunks", None) is not None
        for _ in range(n_bootstrap):
            bootstrap_indices = rng.choice(n_times, size=n_times, replace=True)
            s_boot = s.isel(time=bootstrap_indices)
            o_boot = o.isel(time=bootstrap_indices)
            obs_var_boot = o_boot.var(dim="time", ddof=1)
            diff_boot = (s_boot.mean(dim="time") - o_boot.mean(dim="time")) ** 2
            normalized_boot = diff_boot / obs_var_boot.where(obs_var_boot != 0)
            boot_dims = list(normalized_boot.dims)
            boot_mean = normalized_boot.mean(dim=boot_dims, skipna=True) if boot_dims else normalized_boot
            bootstrap_smpi.append(boot_mean if dask_backed else float(boot_mean))

        if dask_backed:
            bootstrap_da = xr.concat(bootstrap_smpi, dim="bootstrap").chunk({"bootstrap": -1})
            smpi_lower = bootstrap_da.quantile(0.05, dim="bootstrap", skipna=True)
            smpi_upper = bootstrap_da.quantile(0.95, dim="bootstrap", skipna=True)
        else:
            bootstrap_array = np.array(bootstrap_smpi)
            finite_bootstrap = bootstrap_array[np.isfinite(bootstrap_array)]
            if finite_bootstrap.size:
                smpi_lower, smpi_upper = np.percentile(finite_bootstrap, [5, 95])
            else:
                smpi_lower = smpi_upper = np.nan

        return smpi, smpi_lower, smpi_upper

    def _MFM_shared_components(self, s, o):
        """Return default MFM components plus combined MFM from one component pass."""
        omega = self.MFM_omega(s, o)
        varphi = self.MFM_varphi(s, o)
        eta = self.MFM_eta(s, o)
        mfm = 1 - np.sqrt(((1 - omega) ** 2 + (1 - varphi) ** 2 + (1 - eta) ** 2) / 3)
        return {"MFM_omega": omega, "MFM_varphi": varphi, "MFM_eta": eta, "MFM": mfm}

    def MFM_omega(self, s, o, p=1, phase_penalty_scaling=4, phase=True):
        """Return MFM's normalized error with phase penalty component (omega).

        ``p`` selects the error norm. ``phase_penalty_scaling`` controls how
        strongly phase differences affect the cosine penalty; the default 4
        preserves OpenBench's historical behavior and is not a physical constant.
        """
        s, o = self._validate_inputs(s, o)

        def FFT_component(sim, obs):
            """Calculate phase difference using Fast Fourier Transform"""
            N = len(obs)
            if N != len(sim) or N < 4:
                return 0.0

            fft_obs = np.fft.rfft(obs)
            fft_sim = np.fft.rfft(sim)

            # Selects the strongest observed Fourier component representing at least two cycles across the record
            dominant_freq_idx = np.argmax(np.abs(fft_obs[2:])) + 2

            # Calculate phase difference
            phase_obs = np.angle(fft_obs)
            phase_sim = np.angle(fft_sim)
            phase_difference_rad = phase_sim[dominant_freq_idx] - phase_obs[dominant_freq_idx]
            phase_difference_rad = (phase_difference_rad + np.pi) % (2 * np.pi) - np.pi

            return phase_difference_rad

        def calculate_mfm_omega_1d(sim, obs):
            mask = np.isfinite(sim) & np.isfinite(obs)
            sim_clean = sim[mask]
            obs_clean = obs[mask]

            if len(sim_clean) < 3 or len(obs_clean) < 3:
                return np.nan

            if np.mean(obs_clean) == 0:
                return np.nan

            # Normalized error with phase penalty
            nmaep = np.power(np.mean(np.power(np.abs(sim_clean - obs_clean), p)), 1 / p) / abs(np.mean(obs_clean))

            if phase:
                phase_difference_rad = FFT_component(sim_clean, obs_clean)
                phase_penalty = np.cos(phase_difference_rad / phase_penalty_scaling)
                mfm_omega = phase_penalty * np.e ** (-nmaep)
            else:
                mfm_omega = np.e ** (-nmaep)

            return mfm_omega

        if "time" in s.dims:
            # Rechunk time dimension to single chunk for apply_ufunc with dask
            # This is required because time is a core dimension
            if hasattr(s, "chunks") and s.chunks is not None:
                s = s.chunk({"time": -1})
            if hasattr(o, "chunks") and o.chunks is not None:
                o = o.chunk({"time": -1})

            # Stack spatial dimensions for easier iteration
            mfm_omega_values = xr.apply_ufunc(
                calculate_mfm_omega_1d,
                s,
                o,
                input_core_dims=[["time"], ["time"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
        else:
            # No time dimension, return NaN
            mfm_omega_values = xr.full_like(s.isel(time=0) if "time" in s.dims else s, np.nan)

        return mfm_omega_values

    def MFM_varphi(self, s, o, bins_suse=10):
        """Return MFM's variability capture component (varphi)."""
        s, o = self._validate_inputs(s, o)

        def SUSE_component(sim, obs, bins_suse):
            """Calculate Scaled and Unscaled Entropy difference"""
            if len(sim) == 0 or len(obs) == 0:
                return np.nan

            # Scaled case
            min_val = min(sim.min(), obs.min())
            max_val = max(sim.max(), obs.max())
            if min_val == max_val:
                return 0.0  # No entropy difference if all values are the same
            bin_edges_scaled = np.linspace(min_val, max_val, bins_suse + 1)

            hist_sim_s, _ = np.histogram(sim, bins=bin_edges_scaled, density=False)
            hist_obs_s, _ = np.histogram(obs, bins=bin_edges_scaled, density=False)

            total_s_sim = np.sum(hist_sim_s)
            total_s_obs = np.sum(hist_obs_s)

            p_sim_s = hist_sim_s / total_s_sim if total_s_sim > 0 else np.zeros_like(hist_sim_s)
            p_obs_s = hist_obs_s / total_s_obs if total_s_obs > 0 else np.zeros_like(hist_obs_s)

            def entropy(p):
                p = p[p > 0]
                return -np.sum(p * np.log(p)) if len(p) > 0 else 0.0

            Hs = abs(entropy(p_sim_s) - entropy(p_obs_s))

            # Unscaled case
            if sim.min() == sim.max():
                Hu_sim = 0.0
            else:
                bin_edges_u_sim = np.linspace(sim.min(), sim.max(), bins_suse + 1)
                hist_sim_u, _ = np.histogram(sim, bins=bin_edges_u_sim, density=False)
                p_sim_u = hist_sim_u / np.sum(hist_sim_u) if np.sum(hist_sim_u) > 0 else np.zeros_like(hist_sim_u)
                Hu_sim = entropy(p_sim_u)

            if obs.min() == obs.max():
                Hu_obs = 0.0
            else:
                bin_edges_u_obs = np.linspace(obs.min(), obs.max(), bins_suse + 1)
                hist_obs_u, _ = np.histogram(obs, bins=bin_edges_u_obs, density=False)
                p_obs_u = hist_obs_u / np.sum(hist_obs_u) if np.sum(hist_obs_u) > 0 else np.zeros_like(hist_obs_u)
                Hu_obs = entropy(p_obs_u)

            Hu = abs(Hu_sim - Hu_obs)

            return max(Hs, Hu)

        def calculate_mfm_varphi_1d(sim, obs):
            mask = np.isfinite(sim) & np.isfinite(obs)
            sim_clean = sim[mask]
            obs_clean = obs[mask]

            if len(sim_clean) < 3 or len(obs_clean) < 3:
                return np.nan

            # Variability capture
            suse = SUSE_component(sim_clean, obs_clean, bins_suse)
            if np.isnan(suse):
                return np.nan
            mfm_varphi = np.e ** (-suse)

            return mfm_varphi

        if "time" in s.dims:
            # Rechunk time dimension to single chunk for apply_ufunc with dask
            # This is required because time is a core dimension
            if hasattr(s, "chunks") and s.chunks is not None:
                s = s.chunk({"time": -1})
            if hasattr(o, "chunks") and o.chunks is not None:
                o = o.chunk({"time": -1})

            # Stack spatial dimensions for easier iteration
            mfm_varphi_values = xr.apply_ufunc(
                calculate_mfm_varphi_1d,
                s,
                o,
                input_core_dims=[["time"], ["time"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
        else:
            # No time dimension, return NaN
            mfm_varphi_values = xr.full_like(s.isel(time=0) if "time" in s.dims else s, np.nan)

        return mfm_varphi_values

    def MFM_eta(self, s, o, bins_phi=10):
        """Return MFM's distribution similarity component (eta)."""
        s, o = self._validate_inputs(s, o)

        # Helper functions for single time series
        def PHI_component(sim, obs, bins_phi):
            """Calculate Percentage of Histogram Intersection"""
            if len(sim) == 0 or len(obs) == 0:
                return np.nan
            bin_min = min(np.min(sim), np.min(obs))
            bin_max = max(np.max(sim), np.max(obs))
            if bin_min == bin_max:
                return 1.0  # Perfect match if all values are the same
            bin_edges = np.linspace(bin_min, bin_max, bins_phi + 1)
            hist_sim, _ = np.histogram(sim, bins=bin_edges, density=False)
            hist_obs, _ = np.histogram(obs, bins=bin_edges, density=False)
            min_sum = np.sum(np.minimum(hist_sim, hist_obs))
            obs_total = np.sum(hist_obs)
            if obs_total == 0:
                return np.nan
            return min_sum / obs_total

        def calculate_mfm_eta_1d(sim, obs):
            # Remove NaN values
            mask = np.isfinite(sim) & np.isfinite(obs)
            sim_clean = sim[mask]
            obs_clean = obs[mask]

            if len(sim_clean) < 3 or len(obs_clean) < 3:
                return np.nan

            # Distribution similarity
            mfm_eta = PHI_component(sim_clean, obs_clean, bins_phi)
            if np.isnan(mfm_eta):
                return np.nan

            return mfm_eta

        if "time" in s.dims:
            # Rechunk time dimension to single chunk for apply_ufunc with dask
            # This is required because time is a core dimension
            if hasattr(s, "chunks") and s.chunks is not None:
                s = s.chunk({"time": -1})
            if hasattr(o, "chunks") and o.chunks is not None:
                o = o.chunk({"time": -1})

            # Stack spatial dimensions for easier iteration
            mfm_eta_values = xr.apply_ufunc(
                calculate_mfm_eta_1d,
                s,
                o,
                input_core_dims=[["time"], ["time"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
        else:
            # No time dimension, return NaN
            mfm_eta_values = xr.full_like(s.isel(time=0) if "time" in s.dims else s, np.nan)

        return mfm_eta_values

    def MFM(self, s, o, p=1, bins_suse=10, bins_phi=10, phase_penalty_scaling=4, phase=True):
        """
        Calculate Model Fidelity Metric (MFM) for each grid cell.

        MFM integrates four components:
        1. Normalized Mean Absolute p-Error (NMAEp) - relative error
        2. Scaled and Unscaled Entropy difference (SUSE) - variability capture
        3. Percentage of Histogram Intersection (PHI) - distribution matching
        4. Phase Difference Radius - phase difference (optional)

        Args:
            s (xr.DataArray): Simulated data (time, lat, lon)
            o (xr.DataArray): Observed data (time, lat, lon)
            p (float): Exponent for error calculation (default=1, p=1 gives MAE, p=2 gives RMSE)
            bins_suse (int): Number of bins for entropy calculation (default=10)
            bins_phi (int): Number of bins for histogram intersection (default=10)
            phase_penalty_scaling (float): Scaling factor for phase difference penalty. The default 4 preserves
                OpenBench's historical behavior and is not a physical constant.
            phase (bool): Whether to include phase difference component (default=True)

        Returns:
            xr.DataArray: Model Fidelity Metric value (lat, lon)
        """

        # Validate and align inputs
        s, o = self._validate_inputs(s, o)

        # Helper functions for single time series
        def PHI_component(sim, obs, bins_phi):
            """Calculate Percentage of Histogram Intersection"""
            if len(sim) == 0 or len(obs) == 0:
                return np.nan
            bin_min = min(np.min(sim), np.min(obs))
            bin_max = max(np.max(sim), np.max(obs))
            if bin_min == bin_max:
                return 1.0  # Perfect match if all values are the same
            bin_edges = np.linspace(bin_min, bin_max, bins_phi + 1)
            hist_sim, _ = np.histogram(sim, bins=bin_edges, density=False)
            hist_obs, _ = np.histogram(obs, bins=bin_edges, density=False)
            min_sum = np.sum(np.minimum(hist_sim, hist_obs))
            obs_total = np.sum(hist_obs)
            if obs_total == 0:
                return np.nan
            return min_sum / obs_total

        def SUSE_component(sim, obs, bins_suse):
            """Calculate Scaled and Unscaled Entropy difference"""
            if len(sim) == 0 or len(obs) == 0:
                return np.nan

            # Scaled case
            min_val = min(sim.min(), obs.min())
            max_val = max(sim.max(), obs.max())
            if min_val == max_val:
                return 0.0  # No entropy difference if all values are the same
            bin_edges_scaled = np.linspace(min_val, max_val, bins_suse + 1)

            hist_sim_s, _ = np.histogram(sim, bins=bin_edges_scaled, density=False)
            hist_obs_s, _ = np.histogram(obs, bins=bin_edges_scaled, density=False)

            total_s_sim = np.sum(hist_sim_s)
            total_s_obs = np.sum(hist_obs_s)

            p_sim_s = hist_sim_s / total_s_sim if total_s_sim > 0 else np.zeros_like(hist_sim_s)
            p_obs_s = hist_obs_s / total_s_obs if total_s_obs > 0 else np.zeros_like(hist_obs_s)

            def entropy(p):
                p = p[p > 0]
                return -np.sum(p * np.log(p)) if len(p) > 0 else 0.0

            Hs = abs(entropy(p_sim_s) - entropy(p_obs_s))

            # Unscaled case
            if sim.min() == sim.max():
                Hu_sim = 0.0
            else:
                bin_edges_u_sim = np.linspace(sim.min(), sim.max(), bins_suse + 1)
                hist_sim_u, _ = np.histogram(sim, bins=bin_edges_u_sim, density=False)
                p_sim_u = hist_sim_u / np.sum(hist_sim_u) if np.sum(hist_sim_u) > 0 else np.zeros_like(hist_sim_u)
                Hu_sim = entropy(p_sim_u)

            if obs.min() == obs.max():
                Hu_obs = 0.0
            else:
                bin_edges_u_obs = np.linspace(obs.min(), obs.max(), bins_suse + 1)
                hist_obs_u, _ = np.histogram(obs, bins=bin_edges_u_obs, density=False)
                p_obs_u = hist_obs_u / np.sum(hist_obs_u) if np.sum(hist_obs_u) > 0 else np.zeros_like(hist_obs_u)
                Hu_obs = entropy(p_obs_u)

            Hu = abs(Hu_sim - Hu_obs)

            return max(Hs, Hu)

        def FFT_component(sim, obs):
            """Calculate phase difference using Fast Fourier Transform"""
            N = len(obs)
            if N != len(sim) or N < 4:
                return 0.0

            fft_obs = np.fft.rfft(obs)
            fft_sim = np.fft.rfft(sim)

            # Selects the strongest observed Fourier component representing at least two cycles across the record
            dominant_freq_idx = np.argmax(np.abs(fft_obs[2:])) + 2

            # Calculate phase difference
            phase_obs = np.angle(fft_obs)
            phase_sim = np.angle(fft_sim)
            phase_difference_rad = phase_sim[dominant_freq_idx] - phase_obs[dominant_freq_idx]
            phase_difference_rad = (phase_difference_rad + np.pi) % (2 * np.pi) - np.pi

            return phase_difference_rad

        def calculate_mfm_1d(sim, obs):
            """Calculate MFM for a single time series"""
            # Remove NaN values
            mask = np.isfinite(sim) & np.isfinite(obs)
            sim_clean = sim[mask]
            obs_clean = obs[mask]

            if len(sim_clean) < 3 or len(obs_clean) < 3:
                return np.nan

            if np.mean(obs_clean) == 0:
                return np.nan

            # Calculate components
            # 1. Normalized error with phase penalty
            nmaep = np.power(np.mean(np.power(np.abs(sim_clean - obs_clean), p)), 1 / p) / abs(np.mean(obs_clean))

            if phase:
                phase_difference_rad = FFT_component(sim_clean, obs_clean)
                phase_penalty = np.cos(phase_difference_rad / phase_penalty_scaling)
                normalized_error = phase_penalty * np.e ** (-nmaep)
            else:
                normalized_error = np.e ** (-nmaep)

            # 2. Variability capture
            suse = SUSE_component(sim_clean, obs_clean, bins_suse)
            if np.isnan(suse):
                return np.nan
            variability_capture = np.e ** (-suse)

            # 3. Distribution similarity
            distribution_similarity = PHI_component(sim_clean, obs_clean, bins_phi)
            if np.isnan(distribution_similarity):
                return np.nan

            # Calculate MFM
            mfm_value = 1 - np.sqrt(
                ((1 - normalized_error) ** 2 + (1 - variability_capture) ** 2 + (1 - distribution_similarity) ** 2) / 3
            )

            return mfm_value

        # Apply MFM to each grid cell
        # Get dimensions
        if "time" in s.dims:
            # Rechunk time dimension to single chunk for apply_ufunc with dask
            # This is required because time is a core dimension
            if hasattr(s, "chunks") and s.chunks is not None:
                s = s.chunk({"time": -1})
            if hasattr(o, "chunks") and o.chunks is not None:
                o = o.chunk({"time": -1})

            # Stack spatial dimensions for easier iteration
            result = xr.apply_ufunc(
                calculate_mfm_1d,
                s,
                o,
                input_core_dims=[["time"], ["time"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
        else:
            # No time dimension, return NaN
            result = xr.full_like(s.isel(time=0) if "time" in s.dims else s, np.nan)

        return result

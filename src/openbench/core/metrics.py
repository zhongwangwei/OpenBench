# -*- coding: utf-8 -*-
import logging

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
        Calculate Percent Bias.

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
        return xr.corr(s, o, dim=["time"]) ** 2

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
        """Calculate Cohen's kappa along time with multi-dimensional support."""
        s, o = xr.align(s, o, join="inner")

        def _kappa_1d(s_values, o_values):
            mask = np.isfinite(s_values) & np.isfinite(o_values)
            s_flat = s_values[mask].astype(int)
            o_flat = o_values[mask].astype(int)
            if s_flat.size == 0:
                return np.nan
            unique_data = np.unique(np.concatenate([s_flat, o_flat]))
            kappa_mat = np.zeros((len(unique_data), len(unique_data)), dtype=float)
            index = {value: idx for idx, value in enumerate(unique_data)}
            for sv, ov in zip(s_flat, o_flat):
                kappa_mat[index[sv], index[ov]] += 1
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

    def rSD(self, s, o):
        # Ratio of standard deviations
        # also see E. Towler et al.: Benchmarking model simulations of retrospective streamflow in the contiguous US
        # Indicates if flow variability is being over- or underestimated; calculated from rSD in the hydroGOF R package
        raise NotImplementedError("rSD metric is not yet implemented")

    def PBIAS_HF(self, s, o):
        # Percent bias of flows ≥ Q98 (Yilmaz et al., 2008)
        # also see E. Towler et al.: Benchmarking model simulations of retrospective streamflow in the contiguous US
        # Characterizes response to large precipitation events; calculated using flows ≥ the 98th percentile flow with pbias in the hydroGOF R package
        raise NotImplementedError("PBIAS_HF metric is not yet implemented")

    def PBIAS_LF(self, s, o):
        # Percent bias of flows ≤ Q30(Yilmaz et al., 2008)
        # also see E. Towler et al.: Benchmarking model simulations of retrospective streamflow in the contiguous US
        # Characterizes baseflow; calculated following equations in
        # Yilmaz et al. (2008) using logged flows ≤ the 30th percentile (zeros are set to the USGS observational threshold
        # of 0.01 ft3 s−1 (0.000283 m3 s−1))
        raise NotImplementedError("PBIAS_LF metric is not yet implemented")

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
            use_abs (bool, optional): If True, uses absolute value of slope in calculation. Defaults to False.
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
        normalized_diff = xr.where(obs_var != 0, diff_squared / obs_var, np.nan)

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
            normalized_boot = xr.where(obs_var_boot != 0, diff_boot / obs_var_boot, np.nan)
            boot_dims = list(normalized_boot.dims)
            boot_mean = normalized_boot.mean(dim=boot_dims, skipna=True) if boot_dims else normalized_boot
            bootstrap_smpi.append(boot_mean if dask_backed else float(boot_mean))

        if dask_backed:
            bootstrap_da = xr.concat(bootstrap_smpi, dim="bootstrap").chunk({"bootstrap": -1})
            smpi_lower = bootstrap_da.quantile(0.05, dim="bootstrap", skipna=True)
            smpi_upper = bootstrap_da.quantile(0.95, dim="bootstrap", skipna=True)
        else:
            bootstrap_array = np.array(bootstrap_smpi)
            smpi_lower, smpi_upper = np.percentile(bootstrap_array, [5, 95])

        return smpi, smpi_lower, smpi_upper

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
            phase_penalty_scaling (float): Scaling factor for phase difference penalty (default=4)
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
            if N != len(sim) or N < 3:
                return 0.0

            fft_obs = np.fft.fft(obs)
            fft_sim = np.fft.fft(sim)

            np.fft.fftfreq(N, d=1.0)

            # Find dominant frequency
            if N // 2 < 1:
                return 0.0

            if len(sim) > 365:
                # Skip the very lowest Fourier bins (periods longer than
                # ~N/11 samples) before picking the dominant frequency: those
                # bins capture the trend / multi-year drift rather than the
                # sub-seasonal phase we want for MFM. The previous code
                # hard-coded 33, which is ~N/11 for a 1-year daily series
                # but became arbitrarily small relative to N on longer
                # series; scale the floor with N so the cutoff tracks the
                # dataset resolution instead of being a magic number.
                low_freq_floor = max(1, N // 11)
                dominant_freq_idx = max(np.argmax(np.abs(fft_obs[1 : N // 2 + 1])), low_freq_floor) + 1
            else:
                dominant_freq_idx = np.argmax(np.abs(fft_obs[1 : N // 2 + 1])) + 1

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

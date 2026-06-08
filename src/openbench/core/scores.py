import logging

import numpy as np
import xarray as xr


class scores:
    """
    A class for calculating various performance scores for model evaluation.
    The score varies from 0~1, 1 being the best.

    """

    def __init__(self):
        self.name = "scores"
        self.version = "0.1"
        self.release = "0.1"
        self.date = "Mar 2023"
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"

        # NOTE: We deliberately do NOT call `np.seterr(all="ignore")`
        # here — see the matching comment in core/metrics.py. That call
        # is a process-wide setting and would silence legitimate runtime
        # warnings in unrelated code. Individual scores guard their own
        # divisions with `xr.where(...)`.

    def _validate_inputs(self, s, o):
        """
        Validate, coordinate-align, and pairwise-mask input DataArrays.

        Score normalizers (means, variances, seasonal amplitudes) must be
        calculated on the same finite sim/obs pairs as the errors. Otherwise
        a missing model value can be excluded from the numerator but still
        influence the observed variance/mean denominator.
        """
        if not isinstance(s, xr.DataArray) or not isinstance(o, xr.DataArray):
            logging.error("Inputs must be xarray DataArrays")
            raise TypeError("Inputs must be xarray DataArrays")

        s, o = xr.align(s, o, join="inner")
        mask = np.isfinite(s) & np.isfinite(o)
        return s.where(mask), o.where(mask)

    def _calculate_mean_and_anomalies(self, data):
        """
        Calculate mean and anomalies for a given dataset.

        Args:
            data (xarray.DataArray): Input data

        Returns:
            tuple: (mean, anomalies)
        """
        mean = data.mean(dim="time")
        anomalies = data.groupby("time.month") - data.groupby("time.month").mean("time")
        return mean, anomalies

    def index_agreement(self, s, o):
        """
        Calculate index of agreement.

        Args:
            s (xarray.DataArray): Simulated data
            o (xarray.DataArray): Observed data

        Returns:
            xarray.DataArray: Index of agreement
        """
        s, o = self._validate_inputs(s, o)
        numerator = ((o - s) ** 2).sum(dim="time")
        denominator = ((np.abs(s - o.mean(dim="time")) + np.abs(o - o.mean(dim="time"))) ** 2).sum(dim="time")
        # Mirror the metrics.index_agreement guard — denominator is 0 when
        # observation is constant, which would otherwise produce inf.
        return xr.where(denominator != 0, 1 - numerator / denominator, np.nan)

    def nBiasScore(self, s, o):
        """
        Calculate normalized Bias Score.

        Args:
            s (xarray.DataArray): Simulated data
            o (xarray.DataArray): Observed data

        Returns:
            xarray.DataArray: Normalized Bias Score (0 to 1, 1 being best)
        """
        s, o = self._validate_inputs(s, o)
        bias = s.mean(dim="time") - o.mean(dim="time")
        crms = np.sqrt(((o - o.mean(dim="time")) ** 2).mean(dim="time"))
        # Constant observations -> crms=0 -> inf. Return NaN instead so the
        # overall score correctly drops these grid cells.
        return xr.where(crms != 0, np.exp(-np.abs(bias) / crms), np.nan)

    def nRMSEScore(self, s, o):
        """
        Calculate normalized RMSE Score.

        Args:
            s (xarray.DataArray): Simulated data
            o (xarray.DataArray): Observed data

        Returns:
            xarray.DataArray: Normalized RMSE Score (0 to 1, 1 being best)
        """
        s, o = self._validate_inputs(s, o)
        s_mean, o_mean = s.mean(dim="time"), o.mean(dim="time")
        crms = np.sqrt(((o - o_mean) ** 2).mean(dim="time"))
        crmse = np.sqrt((((s - s_mean) - (o - o_mean)) ** 2).mean(dim="time"))
        return xr.where(crms != 0, np.exp(-crmse / crms), np.nan)

    def nPhaseScore(self, s, o):
        """
        Calculate normalized Phase Score.

        Args:
            s (xarray.DataArray): Simulated data
            o (xarray.DataArray): Observed data

        Returns:
            xarray.DataArray: Normalized Phase Score (0 to 1, 1 being best)
        """
        s, o = self._validate_inputs(s, o)
        ref_monthly = o.groupby("time.month").mean("time")
        sim_monthly = s.groupby("time.month").mean("time")
        valid = ref_monthly.notnull().any("month") & sim_monthly.notnull().any("month")
        ref_max_month = ref_monthly.idxmax("month")
        sim_max_month = sim_monthly.idxmax("month")
        phase_shift = (sim_max_month - ref_max_month) * 365 / 12
        return xr.where(valid, 0.5 * (1 + np.cos(2 * np.pi * phase_shift / 365)), np.nan)

    def nIavScore(self, s, o):
        """
        Calculate normalized Interannual Variability Score.

        Args:
            s (xarray.DataArray): Simulated data
            o (xarray.DataArray): Observed data

        Returns:
            xarray.DataArray: Normalized IAV Score (0 to 1, 1 being best)
        """
        s, o = self._validate_inputs(s, o)
        _, s_anom = self._calculate_mean_and_anomalies(s)
        _, o_anom = self._calculate_mean_and_anomalies(o)

        s_iav = np.sqrt((s_anom**2).mean("time"))
        o_iav = np.sqrt((o_anom**2).mean("time"))

        # Tropical evergreen / arid grid cells can have no IAV (o_iav=0),
        # which would otherwise yield inf. Such cells should be NaN — they
        # also fail the np.isnan(...).all() check in Overall_Score and would
        # silently pollute the aggregate.
        return xr.where(o_iav != 0, np.exp(-np.abs(s_iav - o_iav) / o_iav), np.nan)

    def nSpatialScore(self, s, o):
        """
        Calculate normalized Spatial Score.

        Args:
            s (xarray.DataArray): Simulated data
            o (xarray.DataArray): Observed data

        Returns:
            xarray.DataArray: Normalized Spatial Score (0 to 1, 1 being best)
        """
        s, o = self._validate_inputs(s, o)
        smean = s.mean(dim="time").squeeze()
        omean = o.mean(dim="time").squeeze()

        # Calculate the spatial correlation between reference and model
        # spatial_corr = np.corrcoef(smean.values.flatten(), omean.values.flatten())[0, 1]
        try:
            spatial_corr = xr.corr(smean, omean, dim=["lat", "lon"])
        except (ValueError, KeyError):
            # Bare `except:` was previously swallowing KeyboardInterrupt /
            # SystemExit; narrow to the actual missing-dim cases (1D data
            # or non-default coordinate names).
            spatial_corr = xr.corr(smean, omean)

        # Calculate the spatial standard deviation for reference and model
        ref_std = omean.std().squeeze()
        sim_std = smean.std().squeeze()
        # Guard zero standard deviations: ref_std == 0 would make
        # sim_std/ref_std infinite, while sim_std == 0 would make 1/sigma
        # infinite in the score formula below.
        sigma = xr.where((ref_std != 0) & (sim_std != 0), sim_std / ref_std, np.nan)
        # Calculate the spatial score
        spatial_score_0 = 2.0 * (1 + spatial_corr) / (sigma + 1 / sigma) ** 2
        # spatial_score   = xr.full_like(smean, spatial_score_0)
        spatial_score = smean * 0.0 + spatial_score_0
        return spatial_score

    def Overall_Score(self, s, o):
        """
        Calculate Overall Score based on multiple metrics.

        Args:
            s (xarray.DataArray): Simulated data
            o (xarray.DataArray): Observed data

        Returns:
            xarray.DataArray: Overall Score
        """
        bias_score = self.nBiasScore(s, o)
        rmse_score = self.nRMSEScore(s, o)
        phase_score = self.nPhaseScore(s, o)
        iav_score = self.nIavScore(s, o)
        spatial_score = self.nSpatialScore(s, o)

        # Aggregate Scores (Adjust weights as per your ILAMB configuration).
        # Some components are legitimately undefined for only part of a field
        # (for example IAV in cells with no interannual variability). Normalize
        # by the available weighted components per cell rather than decrementing
        # a single global denominator only when an entire component is NaN.
        weighted_components = [
            (bias_score, 1.0),
            (rmse_score, 2.0),
            (phase_score, 1.0),
            (iav_score, 1.0),
            (spatial_score, 1.0),
        ]
        numerator = 0.0
        denominator = 0.0
        for score, weight in weighted_components:
            valid = np.isfinite(score)
            numerator = numerator + score.where(valid, 0.0) * weight
            denominator = denominator + xr.where(valid, weight, 0.0)

        return xr.where(denominator > 0, numerator / denominator, np.nan)

    def nSeasonalityScore(self, s, o):
        """
        Calculate normalized Seasonality Score.

        Args:
            s (xarray.DataArray): Simulated data
            o (xarray.DataArray): Observed data

        Returns:
            xarray.DataArray: Normalized Seasonality Score (0 to 1, 1 being best)
        """
        s, o = self._validate_inputs(s, o)
        s_cycle = s.groupby("time.month").mean("time")
        o_cycle = o.groupby("time.month").mean("time")
        s_amp = s_cycle.max("month") - s_cycle.min("month")
        o_amp = o_cycle.max("month") - o_cycle.min("month")
        # Use annual-cycle amplitude so this score returns one value per
        # spatial cell, consistent with the other normalized score fields.
        relative_error = xr.where(o_amp != 0, (s_amp - o_amp) / o_amp, np.nan)
        return np.exp(-np.abs(relative_error))

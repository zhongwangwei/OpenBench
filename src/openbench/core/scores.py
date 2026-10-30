import numpy as np
import xarray as xr

from openbench.core.metrics import validated_pair, varies_along
from openbench.util.groupby import without_flox


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
        return validated_pair(s, o)

    def _calculate_mean_and_anomalies(self, data):
        """
        Calculate mean and anomalies for a given dataset.

        Args:
            data (xarray.DataArray): Input data

        Returns:
            tuple: (mean, anomalies)
        """
        mean = data.mean(dim="time")
        with without_flox():
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
        return xr.where(varies_along(o) & (crms != 0), np.exp(-np.abs(bias) / crms), np.nan)

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
        return xr.where(varies_along(o) & (crms != 0), np.exp(-crmse / crms), np.nan)

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
        with without_flox():
            ref_monthly = o.groupby("time.month").mean("time")
            sim_monthly = s.groupby("time.month").mean("time")
        ref_amplitude = ref_monthly.max("month") - ref_monthly.min("month")
        sim_amplitude = sim_monthly.max("month") - sim_monthly.min("month")
        complete_cycle = (np.isfinite(ref_monthly) & np.isfinite(sim_monthly)).sum("month") == 12
        valid = complete_cycle & (ref_amplitude > 0) & (sim_amplitude > 0)
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
        # silently pollute the aggregate. A repeated climatology has no IAV either:
        # each calendar month then takes a single value over the years.
        with without_flox():
            month = o.groupby("time.month")
            o_has_iav = (month.max("time") > month.min("time")).any("month")
        return xr.where(o_has_iav & (o_iav != 0), np.exp(-np.abs(s_iav - o_iav) / o_iav), np.nan)

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
        smean = s.mean(dim="time")
        omean = o.mean(dim="time")

        spatial_score_0 = self._spatial_score_from_means(smean, omean)
        # Broadcast the network-wide score back to the spatial field.  This
        # preserves the existing grid output contract (lat/lon dimensions).
        return smean * 0.0 + spatial_score_0

    @staticmethod
    def _spatial_score_from_means(smean, omean, dim=None):
        """Return one spatial score from paired spatial/time-mean values.

        Grid evaluation calls this with a lat/lon field.  Station evaluation
        calls it once after collecting all station means, with ``dim`` set to
        the station dimension.  Degenerate networks return NaN so callers can
        apply their context-specific explicit/implicit error policy.
        """
        smean, omean = xr.align(smean, omean, join="inner")
        valid = np.isfinite(smean) & np.isfinite(omean)
        smean = smean.where(valid)
        omean = omean.where(valid)

        if dim is None and {"lat", "lon"}.issubset(smean.dims) and {"lat", "lon"}.issubset(omean.dims):
            corr_dim = ["lat", "lon"]
        else:
            corr_dim = dim

        # Calculate the spatial correlation between reference and model
        try:
            spatial_corr = xr.corr(smean, omean, dim=corr_dim)
        except (ValueError, KeyError):
            # Bare `except:` was previously swallowing KeyboardInterrupt /
            # SystemExit; narrow to the actual missing-dim cases (1D data
            # or non-default coordinate names).
            spatial_corr = xr.corr(smean, omean)

        # Calculate the spatial standard deviation for reference and model
        ref_std = omean.std(dim=corr_dim).squeeze()
        sim_std = smean.std(dim=corr_dim).squeeze()
        valid_count = valid.sum(dim=corr_dim) if corr_dim is not None else valid.sum()
        # Guard zero standard deviations: ref_std == 0 would make
        # sim_std/ref_std infinite, while sim_std == 0 would make 1/sigma
        # infinite in the score formula below.
        sigma = xr.where((valid_count >= 2) & (ref_std != 0) & (sim_std != 0), sim_std / ref_std, np.nan)
        # Calculate the spatial score
        return 2.0 * (1 + spatial_corr) / (sigma + 1 / sigma) ** 2

    @staticmethod
    def _weighted_score_components(weighted_components):
        """Combine available score components and retain their weighted terms."""
        numerator = 0.0
        denominator = 0.0
        for score, weight in weighted_components:
            valid = np.isfinite(score)
            numerator = numerator + score.where(valid, 0.0) * weight
            denominator = denominator + xr.where(valid, weight, 0.0)
        result = xr.where(denominator > 0, numerator / denominator, np.nan)
        return result, numerator, denominator

    def _overall_nonspatial_components(self, s, o):
        """Return the four per-location Overall Score components and weights."""
        return [
            (self.nBiasScore(s, o), 1.0),
            (self.nRMSEScore(s, o), 2.0),
            (self.nPhaseScore(s, o), 1.0),
            (self.nIavScore(s, o), 1.0),
        ]

    def Overall_Score(self, s, o):
        """
        Calculate Overall Score based on multiple metrics.

        Args:
            s (xarray.DataArray): Simulated data
            o (xarray.DataArray): Observed data

        Returns:
            xarray.DataArray: Overall Score
        """
        # Aggregate Scores (Adjust weights as per your ILAMB configuration).
        # Some components are legitimately undefined for only part of a field
        # (for example IAV in cells with no interannual variability). Normalize
        # by the available weighted components per cell rather than decrementing
        # a single global denominator only when an entire component is NaN.
        weighted_components = self._overall_nonspatial_components(s, o)
        weighted_components.append((self.nSpatialScore(s, o), 1.0))
        result, _, _ = self._weighted_score_components(weighted_components)
        return result

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
        with without_flox():
            s_cycle = s.groupby("time.month").mean("time")
            o_cycle = o.groupby("time.month").mean("time")
        s_amp = s_cycle.max("month") - s_cycle.min("month")
        o_amp = o_cycle.max("month") - o_cycle.min("month")
        complete_cycle = (np.isfinite(s_cycle) & np.isfinite(o_cycle)).sum("month") == 12
        # Use annual-cycle amplitude so this score returns one value per
        # spatial cell, consistent with the other normalized score fields.
        relative_error = xr.where(o_amp != 0, (s_amp - o_amp) / o_amp, np.nan)
        return xr.where(complete_cycle, np.exp(-np.abs(relative_error)), np.nan)

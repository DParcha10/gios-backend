"""
GIOS — Time-Series & Change-Detection Service

Builds temporal composites (monthly / bi-weekly medians), performs
change detection via differencing consecutive composites, identifies
anomalies using z-score thresholds, and generates trend summaries.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import xarray as xr
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


class TimeSeriesService:
    """Temporal analysis of raster index data cubes."""

    # ── Temporal Compositing ────────────────────────────────────────────

    @staticmethod
    def build_composites(
        da: xr.DataArray,
        frequency: str = "MS",
        method: str = "median",
    ) -> xr.DataArray:
        """
        Aggregate a time-dimensional DataArray into temporal composites.

        Parameters
        ----------
        da : xr.DataArray
            Index values with a ``time`` dimension.
        frequency : str
            Pandas offset alias — ``"MS"`` (monthly), ``"2W"`` (bi-weekly),
            ``"QS"`` (quarterly), etc.
        method : str
            Aggregation function: ``"median"`` (default) or ``"mean"``.

        Returns
        -------
        xr.DataArray
            Composite array with resampled time dimension.
        """
        resampled = da.resample(time=frequency)
        if method == "median":
            composites = resampled.median(dim="time", skipna=True)
        elif method == "mean":
            composites = resampled.mean(dim="time", skipna=True)
        else:
            raise ValueError(f"Unsupported method '{method}'. Use 'median' or 'mean'.")

        logger.info(
            "Built %d composites (freq=%s, method=%s)",
            composites.sizes.get("time", 0),
            frequency,
            method,
        )
        return composites

    # ── Change Detection ────────────────────────────────────────────────

    @staticmethod
    def compute_change(
        composites: xr.DataArray,
    ) -> xr.DataArray:
        """
        Compute per-pixel change between consecutive temporal composites.

        Returns a DataArray one time step shorter than *composites*,
        where each slice = ``composite[t+1] − composite[t]``.
        """
        if "time" not in composites.dims:
            raise ValueError("DataArray must have a 'time' dimension.")

        diff = composites.diff(dim="time")
        diff.name = f"d{composites.name}" if composites.name else "change"
        diff.attrs["long_name"] = f"Change in {composites.name}"
        logger.info("Computed change detection (%d steps)", diff.sizes.get("time", 0))
        return diff

    @staticmethod
    def compare_periods(
        da: xr.DataArray,
        period_a: slice,
        period_b: slice,
    ) -> dict[str, float]:
        """
        Compare two explicit time periods and return aggregate statistics.

        Parameters
        ----------
        da : xr.DataArray
            Full time-series DataArray.
        period_a, period_b : slice
            Time slices (e.g. ``slice("2024-01", "2024-06")``).

        Returns
        -------
        dict with keys: period_a_mean, period_b_mean, absolute_change, percent_change.
        """
        mean_a = float(da.sel(time=period_a).mean(skipna=True).values)
        mean_b = float(da.sel(time=period_b).mean(skipna=True).values)
        absolute = mean_b - mean_a
        percent = (absolute / abs(mean_a) * 100) if mean_a != 0 else float("inf")

        return {
            "period_a_mean": round(mean_a, 6),
            "period_b_mean": round(mean_b, 6),
            "absolute_change": round(absolute, 6),
            "percent_change": round(percent, 2),
        }

    # ── Spatial-mean time series extraction ─────────────────────────────

    @staticmethod
    def extract_spatial_mean_series(
        da: xr.DataArray,
    ) -> list[dict]:
        """
        Collapse spatial dimensions to produce a time series of
        spatially-averaged values.

        Returns a list of ``{"timestamp": ..., "value": ...}`` dicts.
        """
        series = da.mean(dim=["x", "y"], skipna=True)
        points = []
        for t in series.time.values:
            val = float(series.sel(time=t).values)
            points.append({
                "timestamp": str(np.datetime_as_string(t, unit="s")),
                "value": round(val, 6) if np.isfinite(val) else None,
            })
        return points

    # ── Anomaly Detection ───────────────────────────────────────────────

    @staticmethod
    def detect_anomalies(
        series: list[dict],
        z_threshold: float = 2.0,
    ) -> list[dict]:
        """
        Flag anomalous observations in a time series using z-scores.

        Parameters
        ----------
        series : list[dict]
            Output of :meth:`extract_spatial_mean_series`.
        z_threshold : float
            Absolute z-score above which a point is deemed anomalous.

        Returns
        -------
        Same list with an added ``"is_anomaly"`` boolean key.
        """
        values = np.array(
            [p["value"] for p in series if p["value"] is not None],
            dtype=np.float64,
        )
        if len(values) < 3:
            for p in series:
                p["is_anomaly"] = False
            return series

        mean = np.nanmean(values)
        std = np.nanstd(values)
        if std == 0:
            for p in series:
                p["is_anomaly"] = False
            return series

        for p in series:
            if p["value"] is None:
                p["is_anomaly"] = False
            else:
                z = abs((p["value"] - mean) / std)
                p["is_anomaly"] = bool(z > z_threshold)

        anomaly_count = sum(1 for p in series if p["is_anomaly"])
        logger.info("Anomaly detection: %d / %d flagged (z>%.1f)", anomaly_count, len(series), z_threshold)
        return series

    # ── Linear Trend ────────────────────────────────────────────────────

    @staticmethod
    def compute_trend(
        series: list[dict],
    ) -> Optional[float]:
        """
        Fit a linear regression to the time series and return the slope
        (units per month).

        Returns ``None`` if insufficient data.
        """
        valid = [(p["timestamp"], p["value"]) for p in series if p["value"] is not None]
        if len(valid) < 3:
            return None

        timestamps = np.array(
            [np.datetime64(t) for t, _ in valid],
            dtype="datetime64[s]",
        )
        values = np.array([v for _, v in valid], dtype=np.float64)

        # Convert to fractional months for the slope
        t_numeric = (timestamps - timestamps[0]).astype(np.float64)  # seconds
        t_months = t_numeric / (30.44 * 86_400)  # approx seconds per month

        slope, _, _, _, _ = scipy_stats.linregress(t_months, values)
        return round(float(slope), 8)


# Module-level singleton
timeseries_service = TimeSeriesService()

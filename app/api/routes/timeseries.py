"""
GIOS — Time-Series API Routes

Endpoints for temporal compositing, trend analysis with anomaly
detection, and period-to-period change detection.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.schemas import (
    ChangeDetectionRequest,
    ChangeDetectionResponse,
    ChangeDetectionResult,
    CompositeRequest,
    SatelliteCollection,
    SpectralIndex,
    TimeSeriesPoint,
    TimeSeriesResponse,
    TrendRequest,
)
from app.services.data_acquisition import data_acquisition_service
from app.services.preprocessing import preprocessing_service
from app.services.indices import index_computation_service
from app.services.timeseries import timeseries_service

router = APIRouter(prefix="/timeseries", tags=["Time-Series"])
logger = logging.getLogger(__name__)

_COLLECTION_MAP = {
    SatelliteCollection.LANDSAT: settings.landsat_collection,
    SatelliteCollection.SENTINEL2: settings.sentinel2_collection,
}


# ── Helper: load → preprocess → compute index ──────────────────────────────

def _load_and_compute(bbox, start_date, end_date, collection_enum, index_name):
    collection = _COLLECTION_MAP[collection_enum]
    ds = data_acquisition_service.load_data_cube(
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
        collection=collection,
    )
    if len(ds.data_vars) == 0:
        raise HTTPException(status_code=404, detail="No imagery found for the given AOI and date range.")

    ds = preprocessing_service.preprocess(ds, collection)
    da = index_computation_service.compute_index(ds, index_name, collection)
    return da


@router.post(
    "/composite",
    response_model=TimeSeriesResponse,
    summary="Generate temporal composites",
)
async def build_composite(req: CompositeRequest) -> TimeSeriesResponse:
    """
    Build temporal composites (e.g. monthly medians) of the requested
    index, and return the spatially-averaged time series.
    """
    try:
        da = _load_and_compute(
            req.bbox.to_list(), req.start_date, req.end_date,
            req.collection, req.index.value,
        )

        composites = timeseries_service.build_composites(da, frequency=req.frequency.value)
        series = timeseries_service.extract_spatial_mean_series(composites)

        points = [
            TimeSeriesPoint(
                timestamp=p["timestamp"],
                value=p["value"] if p["value"] is not None else 0.0,
            )
            for p in series
        ]

        return TimeSeriesResponse(index=req.index, series=points)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Composite generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/trend",
    response_model=TimeSeriesResponse,
    summary="Trend analysis with anomaly detection",
)
async def trend_analysis(req: TrendRequest) -> TimeSeriesResponse:
    """
    Build composites, extract the spatial-mean time series, flag
    anomalies (z-score), and compute a linear trend slope.
    """
    try:
        da = _load_and_compute(
            req.bbox.to_list(), req.start_date, req.end_date,
            req.collection, req.index.value,
        )

        composites = timeseries_service.build_composites(da, frequency=req.frequency.value)
        series = timeseries_service.extract_spatial_mean_series(composites)
        series = timeseries_service.detect_anomalies(series, z_threshold=req.anomaly_z_threshold)
        trend_slope = timeseries_service.compute_trend(series)

        points = [
            TimeSeriesPoint(
                timestamp=p["timestamp"],
                value=p["value"] if p["value"] is not None else 0.0,
                is_anomaly=p.get("is_anomaly", False),
            )
            for p in series
        ]

        return TimeSeriesResponse(
            index=req.index,
            series=points,
            overall_trend=trend_slope,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Trend analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/change-detection",
    response_model=ChangeDetectionResponse,
    summary="Detect change between two periods",
)
async def change_detection(req: ChangeDetectionRequest) -> ChangeDetectionResponse:
    """
    Compare aggregate index values between two time periods and
    quantify absolute and percentage change.
    """
    try:
        # Load the full time span covering both periods
        import datetime as _dt

        overall_start = min(req.period_a_start, req.period_b_start)
        overall_end = max(req.period_a_end, req.period_b_end)

        da = _load_and_compute(
            req.bbox.to_list(), overall_start, overall_end,
            req.collection, req.index.value,
        )

        period_a = slice(
            req.period_a_start.strftime("%Y-%m-%d"),
            req.period_a_end.strftime("%Y-%m-%d"),
        )
        period_b = slice(
            req.period_b_start.strftime("%Y-%m-%d"),
            req.period_b_end.strftime("%Y-%m-%d"),
        )

        stats = timeseries_service.compare_periods(da, period_a, period_b)

        result = ChangeDetectionResult(
            index=req.index,
            period_a_mean=stats["period_a_mean"],
            period_b_mean=stats["period_b_mean"],
            absolute_change=stats["absolute_change"],
            percent_change=stats["percent_change"],
        )

        return ChangeDetectionResponse(results=[result])

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Change detection failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

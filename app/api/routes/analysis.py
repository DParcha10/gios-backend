"""
GIOS — Analysis API Routes

Endpoints for computing spectral indices, zonal statistics, and
algal-bloom detection over satellite imagery.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.schemas import (
    AlgalBloomRequest,
    IndexRequest,
    IndexResponse,
    IndexResultSummary,
    SatelliteCollection,
    SpectralIndex,
    ZonalStatsFeature,
    ZonalStatsRequest,
    ZonalStatsResponse,
)
from app.services.data_acquisition import data_acquisition_service
from app.services.preprocessing import preprocessing_service
from app.services.indices import index_computation_service
from app.services.analysis import spatial_analysis_service

router = APIRouter(prefix="/analysis", tags=["Analysis"])
logger = logging.getLogger(__name__)

_COLLECTION_MAP = {
    SatelliteCollection.LANDSAT: settings.landsat_collection,
    SatelliteCollection.SENTINEL2: settings.sentinel2_collection,
}


@router.post(
    "/indices",
    response_model=IndexResponse,
    summary="Compute spectral indices for an AOI",
)
async def compute_indices(req: IndexRequest) -> IndexResponse:
    """
    Load satellite imagery for the requested AOI and date range,
    apply preprocessing, and compute the requested spectral indices.
    Returns aggregate statistics per time step.
    """
    collection = _COLLECTION_MAP[req.collection]

    try:
        # 1. Load data cube
        ds = data_acquisition_service.load_data_cube(
            bbox=req.bbox.to_list(),
            start_date=req.start_date,
            end_date=req.end_date,
            collection=collection,
        )
        if len(ds.data_vars) == 0:
            raise HTTPException(status_code=404, detail="No imagery found for the given AOI and date range.")

        # 2. Preprocess
        if req.apply_cloud_mask:
            ds = preprocessing_service.preprocess(ds, collection)

        # 3. Compute indices
        summaries: list[IndexResultSummary] = []
        for idx_name in req.indices:
            da = index_computation_service.compute_index(ds, idx_name.value, collection)

            # Aggregate per time step
            if "time" in da.dims:
                for t in da.time.values:
                    slice_t = da.sel(time=t)
                    summaries.append(_summarise(slice_t, idx_name, t))
            else:
                summaries.append(_summarise(da, idx_name))

        return IndexResponse(results=summaries)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Index computation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/zonal-stats",
    response_model=ZonalStatsResponse,
    summary="Compute zonal statistics for an index over vector boundaries",
)
async def compute_zonal_stats(req: ZonalStatsRequest) -> ZonalStatsResponse:
    """
    Compute the requested index, then calculate zonal statistics
    within the supplied polygon boundaries.
    """
    collection = _COLLECTION_MAP[req.collection]

    try:
        ds = data_acquisition_service.load_data_cube(
            bbox=req.bbox.to_list(),
            start_date=req.start_date,
            end_date=req.end_date,
            collection=collection,
        )
        if len(ds.data_vars) == 0:
            raise HTTPException(status_code=404, detail="No imagery found.")

        ds = preprocessing_service.preprocess(ds, collection)
        da = index_computation_service.compute_index(ds, req.index.value, collection)

        # Use the first time step or a temporal mean for zonal stats
        if "time" in da.dims:
            da_2d = da.mean(dim="time", skipna=True)
        else:
            da_2d = da

        stat_names = [s.value for s in req.statistics]
        boundaries_dict = req.boundaries.model_dump()
        raw_stats = spatial_analysis_service.compute_zonal_stats(da_2d, boundaries_dict, stat_names)

        features = [
            ZonalStatsFeature(feature_id=z["feature_id"], statistics=z["statistics"])
            for z in raw_stats
        ]

        import datetime as _dt
        return ZonalStatsResponse(
            index=req.index,
            timestamp=_dt.datetime.utcnow(),
            features=features,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Zonal stats computation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post(
    "/algal-bloom",
    summary="Detect potential algal blooms",
)
async def detect_algal_bloom(req: AlgalBloomRequest) -> dict[str, Any]:
    """
    Run the Cyanobacteria Index (CI_cyano) using Sentinel-2 red-edge bands
    and flag areas exceeding the sensitivity threshold.
    """
    collection = settings.sentinel2_collection

    try:
        ds = data_acquisition_service.load_data_cube(
            bbox=req.bbox.to_list(),
            start_date=req.start_date,
            end_date=req.end_date,
            collection=collection,
        )
        if len(ds.data_vars) == 0:
            raise HTTPException(status_code=404, detail="No Sentinel-2 imagery found.")

        ds = preprocessing_service.preprocess(ds, collection)
        ci = index_computation_service.compute_index(ds, "algal_bloom", collection)

        # Temporal mean of CI
        if "time" in ci.dims:
            ci_mean = ci.mean(dim="time", skipna=True)
        else:
            ci_mean = ci

        total_pixels = int(ci_mean.count().values)
        bloom_pixels = int((ci_mean > req.sensitivity_threshold).sum().values)
        bloom_fraction = bloom_pixels / total_pixels if total_pixels > 0 else 0

        return {
            "index": "algal_bloom",
            "threshold": req.sensitivity_threshold,
            "total_water_pixels_analysed": total_pixels,
            "bloom_flagged_pixels": bloom_pixels,
            "bloom_coverage_fraction": round(bloom_fraction, 4),
            "bloom_detected": bloom_fraction > 0.01,
            "mean_ci_value": round(float(ci_mean.mean(skipna=True).values), 6),
            "max_ci_value": round(float(ci_mean.max(skipna=True).values), 6),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Algal bloom detection failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Private helpers ─────────────────────────────────────────────────────────

def _summarise(da, index: SpectralIndex, timestamp=None) -> IndexResultSummary:
    """Build an IndexResultSummary from a 2-D DataArray slice."""
    import numpy as np
    import datetime as _dt

    vals = da.values[np.isfinite(da.values)]
    ts = (
        _dt.datetime.utcnow()
        if timestamp is None
        else np.datetime_as_string(timestamp, unit="s")
    )
    if isinstance(ts, str):
        ts = _dt.datetime.fromisoformat(ts)

    return IndexResultSummary(
        index=index,
        mean=round(float(np.mean(vals)), 6) if len(vals) else 0.0,
        median=round(float(np.median(vals)), 6) if len(vals) else 0.0,
        min=round(float(np.min(vals)), 6) if len(vals) else 0.0,
        max=round(float(np.max(vals)), 6) if len(vals) else 0.0,
        std=round(float(np.std(vals)), 6) if len(vals) else 0.0,
        valid_pixel_count=len(vals),
        timestamp=ts,
    )

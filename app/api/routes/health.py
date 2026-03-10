"""
GIOS — Health-Check Route
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.config import settings
from app.models.schemas import DataSourceStatus, HealthResponse
from app.services.data_acquisition import data_acquisition_service
from app.services.integration import data_integration_service

router = APIRouter(tags=["Health"])

_start_time = time.time()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
)
async def health_check() -> HealthResponse:
    """
    Return the current health status of the platform, including
    connectivity to upstream data sources.
    """
    uptime = time.time() - _start_time

    # Check STAC catalogue
    stac_ok, stac_ms = data_acquisition_service.check_connectivity()

    # Check NOAA
    noaa_ok, noaa_ms = await data_integration_service.check_noaa_connectivity()

    # Check USGS
    usgs_ok, usgs_ms = await data_integration_service.check_usgs_connectivity()

    sources = [
        DataSourceStatus(
            name="STAC Catalogue (Planetary Computer)",
            url=settings.stac_catalog_url,
            reachable=stac_ok,
            latency_ms=stac_ms,
        ),
        DataSourceStatus(
            name="NOAA / NWS API",
            url="https://api.weather.gov",
            reachable=noaa_ok,
            latency_ms=noaa_ms,
        ),
        DataSourceStatus(
            name="USGS Water Services",
            url="https://waterservices.usgs.gov/nwis",
            reachable=usgs_ok,
            latency_ms=usgs_ms,
        ),
    ]

    return HealthResponse(
        status="ok" if all(s.reachable for s in sources) else "degraded",
        uptime_seconds=round(uptime, 1),
        data_sources=sources,
    )

"""
GIOS — Application Configuration

Typed, environment-driven settings for the entire platform.
Override any default by setting the corresponding env var prefixed with GIOS_
(e.g. GIOS_STAC_CATALOG_URL).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the GIOS backend."""

    # ── General ─────────────────────────────────────────────────────────
    app_name: str = "GIOS Environmental Monitoring Platform"
    debug: bool = False

    # ── STAC / Satellite Data ───────────────────────────────────────────
    stac_catalog_url: str = Field(
        default="https://planetarycomputer.microsoft.com/api/stac/v1",
        description="Root URL of the STAC catalog to query.",
    )
    landsat_collection: str = "landsat-c2-l2"
    sentinel2_collection: str = "sentinel-2-l2a"
    max_cloud_cover: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Maximum allowable cloud-cover percentage for scene filtering.",
    )

    # ── Default AOI (bounding box: west, south, east, north) ───────────
    default_bbox: list[float] = Field(
        default=[-90.0, 29.0, -89.0, 30.5],
        description="Default bounding box [west, south, east, north] in EPSG:4326.",
    )

    # ── Processing Defaults ─────────────────────────────────────────────
    target_crs: str = "EPSG:4326"
    target_resolution: float = Field(
        default=30.0,
        description="Target pixel resolution in metres for analysis outputs.",
    )
    temporal_window_days: int = Field(
        default=365,
        description="Default look-back window in days for time-series queries.",
    )
    composite_frequency: str = Field(
        default="MS",
        description="Pandas offset alias for temporal compositing (e.g. 'MS'=monthly start, '2W'=bi-weekly).",
    )

    # ── External API Keys ───────────────────────────────────────────────
    noaa_api_token: Optional[str] = Field(
        default=None,
        description="NOAA/NWS API token for weather data.",
    )
    usgs_api_key: Optional[str] = Field(
        default=None,
        description="USGS Water Services API key.",
    )

    # ── Caching ─────────────────────────────────────────────────────────
    cache_dir: Path = Field(
        default=Path(".cache/gios"),
        description="Directory for disk-based computation cache.",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Default cache entry time-to-live in seconds.",
    )

    # ── Server ──────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins.",
    )

    model_config = {
        "env_prefix": "GIOS_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# Module-level singleton — import `settings` wherever needed.
settings = Settings()

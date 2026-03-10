"""
GIOS — Pydantic request/response schemas and enumerations.

These models define the API contract between the frontend and backend.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enumerations ────────────────────────────────────────────────────────────


class SatelliteCollection(str, Enum):
    """Supported satellite collections."""

    LANDSAT = "landsat"
    SENTINEL2 = "sentinel2"


class SpectralIndex(str, Enum):
    """Computable spectral / environmental indices."""

    NDVI = "ndvi"
    EVI = "evi"
    NDWI = "ndwi"
    NDMI = "ndmi"
    LST = "lst"
    ALGAL_BLOOM = "algal_bloom"


class CompositeFrequency(str, Enum):
    """Temporal compositing frequency options."""

    WEEKLY = "W"
    BIWEEKLY = "2W"
    MONTHLY = "MS"
    QUARTERLY = "QS"


class StatisticType(str, Enum):
    """Available zonal-statistic aggregations."""

    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    STD = "std"
    COUNT = "count"
    PERCENTILE_25 = "p25"
    PERCENTILE_75 = "p75"
    PERCENTILE_90 = "p90"


# ── Geometry & AOI ──────────────────────────────────────────────────────────


class BBox(BaseModel):
    """Bounding box in EPSG:4326."""

    west: float = Field(..., ge=-180, le=180)
    south: float = Field(..., ge=-90, le=90)
    east: float = Field(..., ge=-180, le=180)
    north: float = Field(..., ge=-90, le=90)

    def to_list(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]


class GeoJSONGeometry(BaseModel):
    """Minimal GeoJSON geometry."""

    type: str
    coordinates: list[Any]


class GeoJSONFeature(BaseModel):
    """A single GeoJSON feature."""

    type: str = "Feature"
    geometry: GeoJSONGeometry
    properties: dict[str, Any] = Field(default_factory=dict)


class GeoJSONFeatureCollection(BaseModel):
    """GeoJSON FeatureCollection for vector boundary input."""

    type: str = "FeatureCollection"
    features: list[GeoJSONFeature]


# ── Request Models ──────────────────────────────────────────────────────────


class SearchParams(BaseModel):
    """Parameters for searching available satellite scenes."""

    bbox: BBox
    start_date: datetime
    end_date: datetime
    collections: list[SatelliteCollection] = Field(
        default=[SatelliteCollection.LANDSAT, SatelliteCollection.SENTINEL2],
    )
    max_cloud_cover: int = Field(default=20, ge=0, le=100)
    limit: int = Field(default=50, ge=1, le=500)


class AcquireRequest(BaseModel):
    """Trigger data acquisition and preprocessing for specific scene IDs."""

    scene_ids: list[str]
    collection: SatelliteCollection
    apply_cloud_mask: bool = True


class IndexRequest(BaseModel):
    """Request computation of spectral indices."""

    bbox: BBox
    start_date: datetime
    end_date: datetime
    indices: list[SpectralIndex]
    collection: SatelliteCollection = SatelliteCollection.LANDSAT
    apply_cloud_mask: bool = True


class ZonalStatsRequest(BaseModel):
    """Request zonal statistics over vector boundaries."""

    bbox: BBox
    start_date: datetime
    end_date: datetime
    index: SpectralIndex
    boundaries: GeoJSONFeatureCollection
    statistics: list[StatisticType] = Field(
        default=[StatisticType.MEAN, StatisticType.MAX, StatisticType.MIN],
    )
    collection: SatelliteCollection = SatelliteCollection.LANDSAT


class AlgalBloomRequest(BaseModel):
    """Dedicated algal-bloom detection request."""

    bbox: BBox
    start_date: datetime
    end_date: datetime
    sensitivity_threshold: float = Field(
        default=0.02,
        description="Minimum cyanobacteria index value to flag as potential bloom.",
    )


class CompositeRequest(BaseModel):
    """Request temporal composite generation."""

    bbox: BBox
    start_date: datetime
    end_date: datetime
    index: SpectralIndex
    frequency: CompositeFrequency = CompositeFrequency.MONTHLY
    collection: SatelliteCollection = SatelliteCollection.LANDSAT


class TrendRequest(BaseModel):
    """Request time-series trend analysis."""

    bbox: BBox
    start_date: datetime
    end_date: datetime
    index: SpectralIndex
    frequency: CompositeFrequency = CompositeFrequency.MONTHLY
    anomaly_z_threshold: float = Field(
        default=2.0,
        description="Z-score threshold above which a value is flagged as anomalous.",
    )
    collection: SatelliteCollection = SatelliteCollection.LANDSAT


class ChangeDetectionRequest(BaseModel):
    """Compare two periods for change detection."""

    bbox: BBox
    period_a_start: datetime
    period_a_end: datetime
    period_b_start: datetime
    period_b_end: datetime
    index: SpectralIndex
    collection: SatelliteCollection = SatelliteCollection.LANDSAT


# ── Response Models ─────────────────────────────────────────────────────────


class SceneMetadata(BaseModel):
    """Metadata for a single satellite scene."""

    scene_id: str
    collection: str
    datetime: datetime
    cloud_cover: Optional[float] = None
    bbox: list[float]
    thumbnail_url: Optional[str] = None
    assets: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of asset key → href for the scene.",
    )


class SearchResponse(BaseModel):
    """Response from a scene search."""

    total_results: int
    scenes: list[SceneMetadata]


class IndexResultPixel(BaseModel):
    """A single pixel-level index result (for point queries)."""

    latitude: float
    longitude: float
    value: float
    timestamp: datetime


class IndexResultSummary(BaseModel):
    """Aggregate summary of a computed index over an AOI."""

    index: SpectralIndex
    mean: float
    median: float
    min: float
    max: float
    std: float
    valid_pixel_count: int
    timestamp: datetime


class IndexResponse(BaseModel):
    """Response from an index computation request."""

    results: list[IndexResultSummary]


class ZonalStatsFeature(BaseModel):
    """Zonal statistics for one vector feature."""

    feature_id: str
    statistics: dict[str, float]


class ZonalStatsResponse(BaseModel):
    """Response from zonal statistics computation."""

    index: SpectralIndex
    timestamp: datetime
    features: list[ZonalStatsFeature]


class TimeSeriesPoint(BaseModel):
    """A single observation in a time series."""

    timestamp: datetime
    value: float
    is_anomaly: bool = False


class TimeSeriesResponse(BaseModel):
    """Response from time-series / trend endpoints."""

    index: SpectralIndex
    series: list[TimeSeriesPoint]
    overall_trend: Optional[float] = Field(
        default=None,
        description="Linear trend slope (units per month).",
    )


class ChangeDetectionResult(BaseModel):
    """Result of change detection between two periods."""

    index: SpectralIndex
    period_a_mean: float
    period_b_mean: float
    absolute_change: float
    percent_change: float
    significant_change_area_km2: Optional[float] = None


class ChangeDetectionResponse(BaseModel):
    """Response from change detection."""

    results: list[ChangeDetectionResult]


class DataSourceStatus(BaseModel):
    """Connectivity status for a single data source."""

    name: str
    url: str
    reachable: bool
    latency_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """System health check response."""

    status: str = "ok"
    uptime_seconds: float
    version: str = "0.1.0"
    data_sources: list[DataSourceStatus] = Field(default_factory=list)

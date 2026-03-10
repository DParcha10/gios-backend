"""
GIOS — Data Acquisition Service

Queries STAC catalogues (Microsoft Planetary Computer / AWS Earth Search) for
Landsat Collection 2 Level-2 and Sentinel-2 Level-2A imagery.  Supports
filtering by AOI bounding box, date range, and maximum cloud cover.  Returns
structured metadata and lazily-loaded xarray DataArrays via odc-stac.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

import planetary_computer
import pystac_client
import xarray as xr
from odc.stac import load as odc_load

from app.config import settings
from app.utils.cache import cache_get, cache_set, make_key

logger = logging.getLogger(__name__)

# ── Band mappings (collection → common name → native band key) ──────────────

BAND_MAP: dict[str, dict[str, str]] = {
    "landsat-c2-l2": {
        "blue": "blue",
        "green": "green",
        "red": "red",
        "nir": "nir08",
        "swir16": "swir16",
        "swir22": "swir22",
        "thermal": "lwir11",
        "qa_pixel": "qa_pixel",
    },
    "sentinel-2-l2a": {
        "blue": "B02",
        "green": "B03",
        "red": "B04",
        "rededge1": "B05",
        "rededge2": "B06",
        "rededge3": "B07",
        "nir": "B08",
        "swir16": "B11",
        "swir22": "B12",
        "scl": "SCL",
    },
}


class DataAcquisitionService:
    """Handles satellite scene discovery and lazy-loading into xarray."""

    def __init__(self) -> None:
        self._catalog_url = settings.stac_catalog_url
        self._client: Optional[pystac_client.Client] = None

    # ── Connection ──────────────────────────────────────────────────────

    def _get_client(self) -> pystac_client.Client:
        if self._client is None:
            self._client = pystac_client.Client.open(
                self._catalog_url,
                modifier=planetary_computer.sign_inplace,
            )
            logger.info("STAC client connected: %s", self._catalog_url)
        return self._client

    # ── Scene Search ────────────────────────────────────────────────────

    def search_scenes(
        self,
        bbox: list[float],
        start_date: datetime,
        end_date: datetime,
        collections: list[str] | None = None,
        max_cloud_cover: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Search the STAC catalogue for matching scenes.

        Returns a list of lightweight metadata dicts (one per scene).
        """
        if collections is None:
            collections = [settings.landsat_collection, settings.sentinel2_collection]
        if max_cloud_cover is None:
            max_cloud_cover = settings.max_cloud_cover

        # Check cache
        cache_key = make_key(
            "search", bbox, str(start_date), str(end_date), collections, max_cloud_cover, limit,
        )
        cached = cache_get(cache_key)
        if cached is not None:
            return cached

        date_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        client = self._get_client()

        search = client.search(
            collections=collections,
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": max_cloud_cover}},
            max_items=limit,
        )

        results: list[dict[str, Any]] = []
        for item in search.items():
            results.append(self._item_to_metadata(item))

        logger.info("Search returned %d scenes", len(results))
        cache_set(cache_key, results, ttl=1800)
        return results

    # ── Lazy Data Loading ───────────────────────────────────────────────

    def load_data_cube(
        self,
        bbox: list[float],
        start_date: datetime,
        end_date: datetime,
        collection: str,
        bands: list[str] | None = None,
        resolution: float | None = None,
    ) -> xr.Dataset:
        """
        Query the catalogue and lazily load matching scenes into an
        xarray Dataset via odc-stac.

        Parameters
        ----------
        bbox : list[float]
            [west, south, east, north] in EPSG:4326.
        start_date, end_date : datetime
            Temporal window.
        collection : str
            STAC collection ID (e.g. ``"landsat-c2-l2"``).
        bands : list[str], optional
            Subset of bands to load (native keys). Loads all mapped bands
            if *None*.
        resolution : float, optional
            Target resolution in metres. Falls back to settings default.

        Returns
        -------
        xr.Dataset
            Lazily-loaded dataset with dimensions (time, y, x).
        """
        if resolution is None:
            resolution = settings.target_resolution

        date_range = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        client = self._get_client()

        search = client.search(
            collections=[collection],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": settings.max_cloud_cover}},
        )

        items = list(search.items())
        if not items:
            logger.warning("No items found for %s in %s", collection, date_range)
            return xr.Dataset()

        if bands is None:
            band_mapping = BAND_MAP.get(collection, {})
            bands = list(band_mapping.values())

        ds = odc_load(
            items,
            bands=bands,
            bbox=bbox,
            resolution=resolution,
            crs=settings.target_crs,
            groupby="solar_day",
            chunks={"time": 1, "x": 2048, "y": 2048},
        )

        logger.info(
            "Loaded data cube: collection=%s, shape=%s, bands=%s",
            collection,
            {k: v for k, v in zip(ds.dims, ds.sizes.values())},
            list(ds.data_vars),
        )
        return ds

    # ── Catalog connectivity check ──────────────────────────────────────

    def check_connectivity(self) -> tuple[bool, float]:
        """
        Ping the STAC catalogue and return (reachable, latency_ms).
        """
        import time

        t0 = time.perf_counter()
        try:
            self._get_client()
            latency = (time.perf_counter() - t0) * 1000
            return True, round(latency, 1)
        except Exception as exc:
            logger.error("STAC catalogue unreachable: %s", exc)
            latency = (time.perf_counter() - t0) * 1000
            return False, round(latency, 1)

    # ── Internals ───────────────────────────────────────────────────────

    @staticmethod
    def _item_to_metadata(item) -> dict[str, Any]:
        props = item.properties
        return {
            "scene_id": item.id,
            "collection": item.collection_id,
            "datetime": props.get("datetime"),
            "cloud_cover": props.get("eo:cloud_cover"),
            "bbox": list(item.bbox) if item.bbox else [],
            "thumbnail_url": (
                item.assets["rendered_preview"].href
                if "rendered_preview" in item.assets
                else None
            ),
            "assets": {k: v.href for k, v in item.assets.items()},
        }


# Module-level singleton
data_acquisition_service = DataAcquisitionService()

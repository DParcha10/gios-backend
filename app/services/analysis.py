"""
GIOS — Spatial Analysis / Zonal Statistics Service

Computes zonal statistics (mean, max, min, std, percentiles) of raster
indices within vector boundaries (counties, watersheds, custom polygons).
Supports both per-zone summaries and pixel-level extraction at point locations.
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio.features
import xarray as xr
from rasterstats import zonal_stats
from shapely.geometry import mapping, shape

from app.utils.geo import geojson_to_shapely

logger = logging.getLogger(__name__)


class SpatialAnalysisService:
    """Zonal statistics and spatial extraction from raster index arrays."""

    # ── Zonal Statistics ────────────────────────────────────────────────

    @staticmethod
    def compute_zonal_stats(
        da: xr.DataArray,
        boundaries: dict[str, Any],
        statistics: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Compute zonal statistics for each feature in a GeoJSON
        FeatureCollection.

        Parameters
        ----------
        da : xr.DataArray
            2-D (y, x) index array (single time step).
        boundaries : dict
            GeoJSON FeatureCollection with polygon features.
        statistics : list[str], optional
            Which stats to compute.  Defaults to
            ``["mean", "max", "min", "std", "count"]``.

        Returns
        -------
        list of dicts — one per feature — with the requested statistics.
        """
        if statistics is None:
            statistics = ["mean", "max", "min", "std", "count"]

        # Convert supported shorthand percentile names
        stat_map = {
            "p25": "percentile_25",
            "p75": "percentile_75",
            "p90": "percentile_90",
        }
        rasterstats_stats = [stat_map.get(s, s) for s in statistics]

        # Extract numpy array and affine transform from xarray
        values = da.values.astype(np.float64)
        transform = da.rio.transform() if hasattr(da, "rio") else _affine_from_xarray(da)

        features = boundaries.get("features", [])
        geometries = [f["geometry"] for f in features]

        results = zonal_stats(
            geometries,
            values,
            affine=transform,
            stats=rasterstats_stats,
            nodata=np.nan,
        )

        # Annotate with feature IDs
        output = []
        for i, (feat, stats_dict) in enumerate(zip(features, results)):
            fid = feat.get("properties", {}).get("id") or feat.get("id") or str(i)
            output.append({
                "feature_id": fid,
                "statistics": {k: round(v, 6) if v is not None else None for k, v in stats_dict.items()},
            })

        logger.info("Zonal stats computed for %d features, stats=%s", len(output), statistics)
        return output

    # ── Time-aware Zonal Statistics ─────────────────────────────────────

    def compute_zonal_stats_timeseries(
        self,
        da: xr.DataArray,
        boundaries: dict[str, Any],
        statistics: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Compute zonal statistics for each feature across every time step.

        Returns a list of dicts with keys:
        ``feature_id``, ``timestamp``, ``statistics``.
        """
        if "time" not in da.dims:
            return [{
                "timestamp": None,
                "zones": self.compute_zonal_stats(da, boundaries, statistics),
            }]

        all_results = []
        for t in da.time.values:
            slice_2d = da.sel(time=t)
            ts_str = str(np.datetime_as_string(t, unit="s"))
            zones = self.compute_zonal_stats(slice_2d, boundaries, statistics)
            all_results.append({"timestamp": ts_str, "zones": zones})

        return all_results

    # ── Point Extraction ────────────────────────────────────────────────

    @staticmethod
    def extract_at_points(
        da: xr.DataArray,
        points: list[dict[str, float]],
    ) -> list[dict[str, Any]]:
        """
        Extract raster values at specific (lon, lat) point locations.

        Parameters
        ----------
        da : xr.DataArray
            2-D or 3-D (time, y, x) array.
        points : list[dict]
            Each dict must contain ``"longitude"`` and ``"latitude"`` keys.

        Returns
        -------
        list of dicts with ``longitude``, ``latitude``, ``values`` (list).
        """
        results = []
        for pt in points:
            lon, lat = pt["longitude"], pt["latitude"]
            try:
                vals = da.sel(x=lon, y=lat, method="nearest")
                if "time" in vals.dims:
                    extracted = [
                        {
                            "timestamp": str(np.datetime_as_string(t, unit="s")),
                            "value": round(float(vals.sel(time=t).values), 6),
                        }
                        for t in vals.time.values
                    ]
                else:
                    extracted = [{"timestamp": None, "value": round(float(vals.values), 6)}]
            except Exception as exc:
                logger.warning("Point extraction failed for (%.4f, %.4f): %s", lon, lat, exc)
                extracted = []

            results.append({
                "longitude": lon,
                "latitude": lat,
                "values": extracted,
            })
        return results

    # ── Rasterize vector boundaries ─────────────────────────────────────

    @staticmethod
    def rasterize_boundaries(
        boundaries: dict[str, Any],
        da: xr.DataArray,
    ) -> np.ndarray:
        """
        Rasterize a GeoJSON FeatureCollection onto the same grid as *da*.

        Each feature is burned with its 1-based index. Background = 0.
        """
        transform = da.rio.transform() if hasattr(da, "rio") else _affine_from_xarray(da)
        shapes = [
            (f["geometry"], i + 1)
            for i, f in enumerate(boundaries.get("features", []))
        ]
        out_shape = (da.sizes["y"], da.sizes["x"])
        rasterized = rasterio.features.rasterize(
            shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.int32,
        )
        return rasterized


# ── Private helpers ─────────────────────────────────────────────────────────


def _affine_from_xarray(da: xr.DataArray):
    """
    Build a rasterio-compatible Affine transform from xarray coordinates
    when rioxarray metadata is unavailable.
    """
    from rasterio.transform import from_bounds

    x = da.coords["x"].values
    y = da.coords["y"].values
    return from_bounds(
        x.min(), y.min(), x.max(), y.max(), len(x), len(y),
    )


# Module-level singleton
spatial_analysis_service = SpatialAnalysisService()

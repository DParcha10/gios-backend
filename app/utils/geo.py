"""
GIOS — Geospatial utility helpers.

Provides common geometry conversions, CRS transforms, AOI validation,
and area calculations used throughout the services layer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import box, mapping, shape
from shapely.ops import transform
from pyproj import Transformer


# ── BBox helpers ────────────────────────────────────────────────────────────


def bbox_to_polygon(bbox: list[float]):
    """Convert [west, south, east, north] to a Shapely Polygon."""
    return box(*bbox)


def bbox_to_geojson(bbox: list[float]) -> dict[str, Any]:
    """Convert [west, south, east, north] to a GeoJSON geometry dict."""
    return mapping(bbox_to_polygon(bbox))


def geojson_to_shapely(geojson: dict[str, Any]):
    """Convert a GeoJSON geometry dict to a Shapely geometry."""
    return shape(geojson)


# ── CRS transforms ─────────────────────────────────────────────────────────


def reproject_geometry(geom, src_crs: str = "EPSG:4326", dst_crs: str = "EPSG:3857"):
    """
    Reproject a Shapely geometry from *src_crs* to *dst_crs*.

    Uses pyproj for thread-safe, high-accuracy transforms.
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transform(transformer.transform, geom)


# ── Area calculations ──────────────────────────────────────────────────────


def area_km2(geom, src_crs: str = "EPSG:4326") -> float:
    """
    Compute the area of a geometry in square kilometres.

    Reprojects to an equal-area projection (EPSG:6933 — cylindrical)
    before computing area.
    """
    projected = reproject_geometry(geom, src_crs, "EPSG:6933")
    return projected.area / 1e6


# ── AOI validation ─────────────────────────────────────────────────────────


def validate_bbox(bbox: list[float]) -> bool:
    """
    Check that a bbox is valid:
      - Has exactly four elements
      - west < east, south < north
      - Values within WGS-84 bounds
    """
    if len(bbox) != 4:
        return False
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180):
        return False
    if not (-90 <= south < north <= 90):
        return False
    return True


def bbox_area_km2(bbox: list[float]) -> float:
    """Convenience: area of a bbox in km²."""
    return area_km2(bbox_to_polygon(bbox))


# ── Point-in-polygon ───────────────────────────────────────────────────────


def point_in_geometry(lon: float, lat: float, geom) -> bool:
    """Check whether a (lon, lat) point falls inside a Shapely geometry."""
    from shapely.geometry import Point

    return geom.contains(Point(lon, lat))


# ── Raster coordinate helpers ──────────────────────────────────────────────


def pixel_coords_to_lonlat(
    transform_affine,
    cols: np.ndarray,
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel (col, row) indices to (longitude, latitude) given an
    affine transform from rasterio.
    """
    xs = transform_affine.c + cols * transform_affine.a + rows * transform_affine.b
    ys = transform_affine.f + cols * transform_affine.d + rows * transform_affine.e
    return xs, ys

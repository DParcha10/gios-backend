"""
GIOS — Spectral Index Computation Service

Computes environmental indices from multispectral satellite band data.
All computations operate on xarray DataArrays for efficient,
chunk-aware (Dask-compatible) processing.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Small epsilon to prevent division by zero
_EPS = np.float32(1e-10)


class IndexComputationService:
    """Computes spectral / environmental indices from xarray datasets."""

    # ── NDVI — Normalised Difference Vegetation Index ───────────────────

    @staticmethod
    def compute_ndvi(nir: xr.DataArray, red: xr.DataArray) -> xr.DataArray:
        """
        NDVI = (NIR − Red) / (NIR + Red)

        Range: [−1, 1].  Healthy vegetation → high positive values.
        """
        ndvi = (nir - red) / (nir + red + _EPS)
        ndvi = ndvi.clip(-1, 1)
        ndvi.name = "ndvi"
        ndvi.attrs["long_name"] = "Normalised Difference Vegetation Index"
        return ndvi

    # ── EVI — Enhanced Vegetation Index ─────────────────────────────────

    @staticmethod
    def compute_evi(
        nir: xr.DataArray,
        red: xr.DataArray,
        blue: xr.DataArray,
        gain: float = 2.5,
        c1: float = 6.0,
        c2: float = 7.5,
        l: float = 1.0,
    ) -> xr.DataArray:
        """
        EVI = G × (NIR − Red) / (NIR + C1×Red − C2×Blue + L)

        Corrects for atmospheric and soil background influences.
        """
        evi = gain * (nir - red) / (nir + c1 * red - c2 * blue + l + _EPS)
        evi = evi.clip(-1, 1)
        evi.name = "evi"
        evi.attrs["long_name"] = "Enhanced Vegetation Index"
        return evi

    # ── NDWI — Normalised Difference Water Index ────────────────────────

    @staticmethod
    def compute_ndwi(green: xr.DataArray, nir: xr.DataArray) -> xr.DataArray:
        """
        NDWI = (Green − NIR) / (Green + NIR)

        Positive values indicate open water surfaces.
        """
        ndwi = (green - nir) / (green + nir + _EPS)
        ndwi = ndwi.clip(-1, 1)
        ndwi.name = "ndwi"
        ndwi.attrs["long_name"] = "Normalised Difference Water Index"
        return ndwi

    # ── NDMI — Normalised Difference Moisture Index ─────────────────────

    @staticmethod
    def compute_ndmi(nir: xr.DataArray, swir: xr.DataArray) -> xr.DataArray:
        """
        NDMI = (NIR − SWIR1.6) / (NIR + SWIR1.6)

        Sensitive to canopy / soil moisture content.
        """
        ndmi = (nir - swir) / (nir + swir + _EPS)
        ndmi = ndmi.clip(-1, 1)
        ndmi.name = "ndmi"
        ndmi.attrs["long_name"] = "Normalised Difference Moisture Index"
        return ndmi

    # ── LST — Land Surface Temperature ──────────────────────────────────

    @staticmethod
    def compute_lst(
        thermal: xr.DataArray,
        scale: float = 0.00341802,
        offset: float = 149.0,
    ) -> xr.DataArray:
        """
        Derive Land Surface Temperature (°C) from Landsat TIRS Band 10.

        Default scale/offset correspond to Landsat Collection 2 Level-2 ST
        product.  ``LST_K = DN × scale + offset``  →  °C = K − 273.15.
        """
        lst_kelvin = thermal.astype(np.float32) * scale + offset
        lst_celsius = lst_kelvin - 273.15
        lst_celsius.name = "lst"
        lst_celsius.attrs["long_name"] = "Land Surface Temperature"
        lst_celsius.attrs["units"] = "°C"
        return lst_celsius

    # ── Algal Bloom / Cyanobacteria Index ───────────────────────────────

    @staticmethod
    def compute_algal_bloom_index(
        red: xr.DataArray,
        rededge1: xr.DataArray,
        rededge2: xr.DataArray,
    ) -> xr.DataArray:
        """
        Cyanobacteria Index (CI_cyano) for algal-bloom detection,
        using Sentinel-2 red-edge bands.

        CI = RedEdge2 − 0.5 × (Red + RedEdge1)

        Elevated values (> ~0.02) indicate potential algal bloom events.
        """
        ci = rededge2 - 0.5 * (red + rededge1)
        ci.name = "algal_bloom"
        ci.attrs["long_name"] = "Cyanobacteria Index (CI_cyano)"
        return ci

    # ── Dispatch: compute by name ───────────────────────────────────────

    def compute_index(
        self,
        ds: xr.Dataset,
        index_name: str,
        collection: str,
    ) -> xr.DataArray:
        """
        Compute a named spectral index from a preprocessed xarray Dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Preprocessed dataset with named bands.
        index_name : str
            One of: ndvi, evi, ndwi, ndmi, lst, algal_bloom.
        collection : str
            STAC collection ID to determine band naming convention.

        Returns
        -------
        xr.DataArray
        """
        band_map = self._resolve_bands(collection)
        index_name = index_name.lower()

        dispatch: dict[str, Callable[..., xr.DataArray]] = {
            "ndvi": lambda: self.compute_ndvi(
                ds[band_map["nir"]], ds[band_map["red"]],
            ),
            "evi": lambda: self.compute_evi(
                ds[band_map["nir"]], ds[band_map["red"]], ds[band_map["blue"]],
            ),
            "ndwi": lambda: self.compute_ndwi(
                ds[band_map["green"]], ds[band_map["nir"]],
            ),
            "ndmi": lambda: self.compute_ndmi(
                ds[band_map["nir"]], ds[band_map["swir16"]],
            ),
            "lst": lambda: self.compute_lst(ds[band_map["thermal"]]),
            "algal_bloom": lambda: self.compute_algal_bloom_index(
                ds[band_map["red"]],
                ds[band_map["rededge1"]],
                ds[band_map["rededge2"]],
            ),
        }

        if index_name not in dispatch:
            raise ValueError(
                f"Unknown index '{index_name}'. "
                f"Supported: {list(dispatch.keys())}"
            )

        logger.info("Computing %s for collection=%s", index_name.upper(), collection)
        return dispatch[index_name]()

    # ── Band-name resolver ──────────────────────────────────────────────

    @staticmethod
    def _resolve_bands(collection: str) -> dict[str, str]:
        """
        Return a mapping of common band names → native keys for a collection.
        """
        from app.services.data_acquisition import BAND_MAP

        mapping = BAND_MAP.get(collection)
        if mapping is None:
            raise ValueError(
                f"No band mapping for collection '{collection}'. "
                f"Supported: {list(BAND_MAP.keys())}"
            )
        return mapping

    # ── Bulk computation ────────────────────────────────────────────────

    def compute_multiple(
        self,
        ds: xr.Dataset,
        index_names: list[str],
        collection: str,
    ) -> dict[str, xr.DataArray]:
        """Compute several indices at once and return a name → DataArray mapping."""
        results: dict[str, xr.DataArray] = {}
        for name in index_names:
            try:
                results[name] = self.compute_index(ds, name, collection)
            except Exception as exc:
                logger.error("Failed to compute %s: %s", name, exc)
        return results


# Module-level singleton
index_computation_service = IndexComputationService()

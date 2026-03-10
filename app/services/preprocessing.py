"""
GIOS — Preprocessing Service

Applies quality masks (Sentinel-2 SCL, Landsat QA_PIXEL), normalises
surface-reflectance values, and handles CRS reprojection / spatial alignment
for cross-sensor analysis.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# ── Landsat QA_PIXEL bit masks ──────────────────────────────────────────────
# Bit 1 → dilated cloud, Bit 3 → cloud, Bit 4 → cloud shadow
_LANDSAT_CLOUD_BITS = (1 << 1) | (1 << 3) | (1 << 4)
# Bit 5 → snow
_LANDSAT_SNOW_BIT = 1 << 5

# ── Sentinel-2 SCL classes to mask ─────────────────────────────────────────
# 0 = no data, 1 = saturated/defective, 3 = cloud shadow,
# 8 = cloud medium prob, 9 = cloud high prob, 10 = thin cirrus
_S2_MASK_CLASSES = {0, 1, 3, 8, 9, 10}


class PreprocessingService:
    """Quality masking, normalisation, and alignment for raster data cubes."""

    # ── Cloud / Quality Masking ─────────────────────────────────────────

    @staticmethod
    def apply_landsat_qa_mask(
        ds: xr.Dataset,
        qa_band: str = "qa_pixel",
        mask_snow: bool = False,
    ) -> xr.Dataset:
        """
        Mask invalid pixels in a Landsat xarray Dataset using the QA_PIXEL band.

        Masked pixels are set to NaN so they are excluded from downstream
        aggregations and index calculations.
        """
        if qa_band not in ds.data_vars:
            logger.warning("QA band '%s' not found — skipping Landsat mask.", qa_band)
            return ds

        qa = ds[qa_band].astype(np.uint16)
        cloud_mask = (qa & _LANDSAT_CLOUD_BITS) == 0  # True → clear

        if mask_snow:
            cloud_mask = cloud_mask & ((qa & _LANDSAT_SNOW_BIT) == 0)

        # Apply mask to every non-QA variable
        masked = ds.drop_vars(qa_band)
        masked = masked.where(cloud_mask)

        logger.info(
            "Landsat QA mask applied — %.1f%% pixels masked.",
            (1 - cloud_mask.mean().values) * 100,
        )
        return masked

    @staticmethod
    def apply_sentinel2_scl_mask(
        ds: xr.Dataset,
        scl_band: str = "SCL",
        extra_mask_classes: Optional[set[int]] = None,
    ) -> xr.Dataset:
        """
        Mask invalid pixels in a Sentinel-2 xarray Dataset using the
        Scene Classification Layer (SCL).
        """
        if scl_band not in ds.data_vars:
            logger.warning("SCL band '%s' not found — skipping S2 mask.", scl_band)
            return ds

        mask_classes = _S2_MASK_CLASSES.copy()
        if extra_mask_classes:
            mask_classes |= extra_mask_classes

        scl = ds[scl_band]
        valid = ~scl.isin(list(mask_classes))

        masked = ds.drop_vars(scl_band)
        masked = masked.where(valid)

        logger.info(
            "Sentinel-2 SCL mask applied — %.1f%% pixels masked.",
            (1 - valid.mean().values) * 100,
        )
        return masked

    def apply_cloud_mask(
        self,
        ds: xr.Dataset,
        collection: str,
    ) -> xr.Dataset:
        """
        Dispatch to the appropriate sensor-specific masking routine.
        """
        if "landsat" in collection.lower():
            return self.apply_landsat_qa_mask(ds)
        elif "sentinel" in collection.lower():
            return self.apply_sentinel2_scl_mask(ds)
        else:
            logger.warning("Unknown collection '%s' — no mask applied.", collection)
            return ds

    # ── Reflectance normalisation ───────────────────────────────────────

    @staticmethod
    def normalise_reflectance(
        ds: xr.Dataset,
        collection: str,
        bands: list[str] | None = None,
    ) -> xr.Dataset:
        """
        Scale raw DN values to [0, 1] surface reflectance.

        - **Landsat C2 L2**: SR bands stored as uint16 with scale=0.0000275,
          offset=-0.2.
        - **Sentinel-2 L2A**: SR bands stored as uint16 with scale=0.0001.
        """
        ds = ds.copy()

        if bands is None:
            bands = [v for v in ds.data_vars]

        if "landsat" in collection.lower():
            scale, offset = 0.0000275, -0.2
        elif "sentinel" in collection.lower():
            scale, offset = 0.0001, 0.0
        else:
            logger.warning("Unknown collection '%s' — skipping normalisation.", collection)
            return ds

        for band in bands:
            if band in ds.data_vars:
                ds[band] = ds[band].astype(np.float32) * scale + offset
                # Clamp to valid reflectance range
                ds[band] = ds[band].clip(0.0, 1.0)

        logger.info("Reflectance normalised for %d bands (collection=%s).", len(bands), collection)
        return ds

    # ── Spatial alignment ───────────────────────────────────────────────

    @staticmethod
    def align_datasets(
        datasets: list[xr.Dataset],
        target_crs: str = "EPSG:4326",
        resolution: float = 30.0,
    ) -> list[xr.Dataset]:
        """
        Reproject and resample a list of datasets to a common CRS and
        resolution so they can be stacked or compared.
        """
        aligned: list[xr.Dataset] = []
        for ds in datasets:
            try:
                reprojected = ds.rio.reproject(
                    target_crs,
                    resolution=resolution,
                    resampling=1,  # bilinear
                )
                aligned.append(reprojected)
            except Exception as exc:
                logger.error("Failed to reproject dataset: %s", exc)
                aligned.append(ds)
        return aligned

    # ── Convenience: full preprocessing pipeline ────────────────────────

    def preprocess(
        self,
        ds: xr.Dataset,
        collection: str,
        normalise: bool = True,
    ) -> xr.Dataset:
        """
        Run the standard preprocessing pipeline:
        1. Apply quality / cloud mask
        2. Normalise reflectance values
        """
        ds = self.apply_cloud_mask(ds, collection)
        if normalise:
            # Exclude non-reflectance bands from normalisation
            exclude = {"qa_pixel", "SCL", "lwir11"}
            bands = [b for b in ds.data_vars if b not in exclude]
            ds = self.normalise_reflectance(ds, collection, bands)
        return ds


# Module-level singleton
preprocessing_service = PreprocessingService()

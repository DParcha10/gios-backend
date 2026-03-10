"""
GIOS — Multi-Source Data Integration Service

Async HTTP clients for supplementary environmental data sources:
  • NOAA / NWS API — weather observations and forecasts
  • USGS Water Services — stream gauge readings, water quality

Includes spatial-join logic to correlate station-level ground data
with satellite-derived indices.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import httpx
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# ── API base URLs ───────────────────────────────────────────────────────────

_NOAA_BASE = "https://api.weather.gov"
_USGS_WATER_BASE = "https://waterservices.usgs.gov/nwis"


class DataIntegrationService:
    """Fetches and correlates external environmental data sources."""

    def __init__(self) -> None:
        self._http: Optional[httpx.AsyncClient] = None

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=30.0)
        return self._http

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()

    # ── NOAA / NWS ──────────────────────────────────────────────────────

    async def get_noaa_stations(
        self,
        bbox: list[float],
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """
        Discover NOAA weather stations within a bounding box.

        Uses the /stations endpoint of the NWS API.
        """
        client = await self._client()
        west, south, east, north = bbox

        headers = {
            "Accept": "application/geo+json",
            "User-Agent": "(gios-monitoring-app, contact@example.com)"
        }
        if settings.noaa_api_token:
            headers["token"] = settings.noaa_api_token

        try:
            resp = await client.get(
                f"{_NOAA_BASE}/stations",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

            # Filter stations within bbox manually (API filtering is limited)
            stations = []
            for feat in data.get("features", []):
                coords = feat.get("geometry", {}).get("coordinates", [None, None])
                lon, lat = coords[0], coords[1]
                if lon is not None and west <= lon <= east and south <= lat <= north:
                    props = feat.get("properties", {})
                    stations.append({
                        "station_id": props.get("stationIdentifier"),
                        "name": props.get("name"),
                        "longitude": lon,
                        "latitude": lat,
                        "elevation_m": props.get("elevation", {}).get("value"),
                    })
                if len(stations) >= limit:
                    break

            logger.info("Found %d NOAA stations in bbox", len(stations))
            return stations

        except httpx.HTTPError as exc:
            logger.error("NOAA stations request failed: %s", exc)
            return []

    async def get_noaa_observations(
        self,
        station_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Fetch recent weather observations for a specific station.
        """
        client = await self._client()
        url = f"{_NOAA_BASE}/stations/{station_id}/observations"
        headers = {
            "Accept": "application/geo+json",
            "User-Agent": "(gios-monitoring-app, contact@example.com)"
        }

        params: dict[str, Any] = {"limit": limit}
        if start_date:
            params["start"] = start_date.isoformat() + "Z"
        if end_date:
            params["end"] = end_date.isoformat() + "Z"

        try:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

            observations = []
            for feat in data.get("features", []):
                props = feat.get("properties", {})
                observations.append({
                    "timestamp": props.get("timestamp"),
                    "temperature_c": _extract_value(props, "temperature"),
                    "humidity_pct": _extract_value(props, "relativeHumidity"),
                    "wind_speed_ms": _extract_value(props, "windSpeed"),
                    "pressure_pa": _extract_value(props, "barometricPressure"),
                    "precipitation_mm": _extract_value(props, "precipitationLastHour"),
                    "description": props.get("textDescription"),
                })

            logger.info("Fetched %d observations for station %s", len(observations), station_id)
            return observations

        except httpx.HTTPError as exc:
            logger.error("NOAA observations request failed for %s: %s", station_id, exc)
            return []

    # ── USGS Water Services ─────────────────────────────────────────────

    async def get_usgs_sites(
        self,
        bbox: list[float],
        site_type: str = "ST",
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """
        Discover USGS water monitoring sites within a bounding box.

        Parameters
        ----------
        bbox : list[float]
            [west, south, east, north].
        site_type : str
            USGS site type code — ``"ST"`` = stream, ``"LK"`` = lake, etc.
        """
        client = await self._client()
        west, south, east, north = bbox

        try:
            resp = await client.get(
                f"{_USGS_WATER_BASE}/site/",
                params={
                    "format": "rdb",
                    "bBox": f"{west},{south},{east},{north}",
                    "siteType": site_type,
                    "siteStatus": "active",
                    "hasDataTypeCd": "iv",
                },
            )
            resp.raise_for_status()
            sites = _parse_usgs_rdb_sites(resp.text, limit)
            logger.info("Found %d USGS sites in bbox", len(sites))
            return sites

        except httpx.HTTPError as exc:
            logger.error("USGS sites request failed: %s", exc)
            return []

    async def get_usgs_instantaneous_values(
        self,
        site_id: str,
        parameter_codes: list[str] | None = None,
        period: str = "P7D",
    ) -> list[dict[str, Any]]:
        """
        Fetch instantaneous (real-time) values for a USGS site.

        Parameters
        ----------
        site_id : str
            USGS site number.
        parameter_codes : list[str], optional
            USGS parameter codes. Common:
            ``"00060"`` – discharge (ft³/s),
            ``"00065"`` – gage height (ft),
            ``"00010"`` – water temperature (°C).
        period : str
            ISO-8601 duration for look-back, e.g. ``"P7D"`` = last 7 days.
        """
        if parameter_codes is None:
            parameter_codes = ["00060", "00065", "00010"]

        client = await self._client()
        try:
            resp = await client.get(
                f"{_USGS_WATER_BASE}/iv/",
                params={
                    "format": "json",
                    "sites": site_id,
                    "parameterCd": ",".join(parameter_codes),
                    "period": period,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            records = []
            for ts in data.get("value", {}).get("timeSeries", []):
                param_name = ts.get("variable", {}).get("variableName", "unknown")
                param_code = ts.get("variable", {}).get("variableCode", [{}])[0].get("value")
                unit = ts.get("variable", {}).get("unit", {}).get("unitCode", "")
                for obs in ts.get("values", [{}])[0].get("value", []):
                    records.append({
                        "timestamp": obs.get("dateTime"),
                        "parameter": param_name,
                        "parameter_code": param_code,
                        "value": float(obs.get("value", "nan")),
                        "unit": unit,
                        "qualifier": obs.get("qualifiers", []),
                    })

            logger.info("Fetched %d USGS records for site %s", len(records), site_id)
            return records

        except httpx.HTTPError as exc:
            logger.error("USGS IV request failed for %s: %s", site_id, exc)
            return []

    # ── Spatial Join / Correlation ──────────────────────────────────────

    @staticmethod
    def spatial_join_stations_to_zones(
        stations: list[dict[str, Any]],
        boundaries: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Assign each station to the vector zone (polygon) it falls within.

        Returns a dict mapping ``feature_id`` → list of station dicts.
        """
        from app.utils.geo import geojson_to_shapely, point_in_geometry

        zone_map: dict[str, list[dict[str, Any]]] = {}
        features = boundaries.get("features", [])

        for feat in features:
            fid = feat.get("properties", {}).get("id") or feat.get("id") or str(id(feat))
            geom = geojson_to_shapely(feat["geometry"])
            matched = [
                s for s in stations
                if point_in_geometry(s["longitude"], s["latitude"], geom)
            ]
            zone_map[fid] = matched

        logger.info(
            "Spatial join: %d stations → %d zones",
            len(stations),
            len(zone_map),
        )
        return zone_map

    # ── Connectivity Checks ─────────────────────────────────────────────

    async def check_noaa_connectivity(self) -> tuple[bool, float]:
        import time

        client = await self._client()
        t0 = time.perf_counter()
        try:
            resp = await client.get(f"{_NOAA_BASE}/", headers={"Accept": "application/json"})
            latency = (time.perf_counter() - t0) * 1000
            return resp.status_code < 500, round(latency, 1)
        except Exception:
            return False, round((time.perf_counter() - t0) * 1000, 1)

    async def check_usgs_connectivity(self) -> tuple[bool, float]:
        import time

        client = await self._client()
        t0 = time.perf_counter()
        try:
            resp = await client.get(
                f"{_USGS_WATER_BASE}/site/",
                params={"format": "rdb", "sites": "01646500", "siteStatus": "all"},
            )
            latency = (time.perf_counter() - t0) * 1000
            return resp.status_code < 500, round(latency, 1)
        except Exception:
            return False, round((time.perf_counter() - t0) * 1000, 1)


# ── Private helpers ─────────────────────────────────────────────────────────


def _extract_value(props: dict, key: str) -> Optional[float]:
    """Extract a numeric value from a NWS observation property."""
    entry = props.get(key)
    if isinstance(entry, dict):
        val = entry.get("value")
        return float(val) if val is not None else None
    return None


def _parse_usgs_rdb_sites(rdb_text: str, limit: int) -> list[dict[str, Any]]:
    """Parse a USGS RDB (tab-delimited) site response into structured dicts."""
    sites = []
    header_line = None
    for line in rdb_text.splitlines():
        if line.startswith("#"):
            continue
        if header_line is None:
            header_line = line.split("\t")
            continue
        # Skip the format-description row (second header row)
        if line.startswith("5s") or all(c in "0123456789sdn " for c in line.replace("\t", " ")):
            continue

        parts = line.split("\t")
        if len(parts) < 4:
            continue
        record = dict(zip(header_line, parts))
        sites.append({
            "site_id": record.get("site_no", ""),
            "name": record.get("station_nm", ""),
            "latitude": _safe_float(record.get("dec_lat_va")),
            "longitude": _safe_float(record.get("dec_long_va")),
            "site_type": record.get("site_tp_cd", ""),
        })
        if len(sites) >= limit:
            break

    return sites


def _safe_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# Module-level singleton
data_integration_service = DataIntegrationService()

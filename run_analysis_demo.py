"""
GIOS — Real Data Analysis Demo

This script demonstrates the GIOS backend's data integration capabilities
by pulling LIVE environmental data from:
  • NOAA / NWS API — weather stations and observations
  • USGS Water Services — stream gauges, discharge, water temperature

Target area: Chesapeake Bay watershed region (Virginia / Maryland)
"""

import asyncio
import json
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.integration import data_integration_service
from app.utils.geo import bbox_to_polygon, area_km2, validate_bbox


# ── Configuration ───────────────────────────────────────────────────────────

# Chesapeake Bay Region bounding box [west, south, east, north]
CHESAPEAKE_BBOX = [-77.5, 37.0, -75.5, 39.5]

# Time window
END_DATE = datetime.utcnow()
START_DATE = END_DATE - timedelta(days=7)

SEPARATOR = "=" * 70


async def main():
    print(SEPARATOR)
    print("  GIOS ENVIRONMENTAL MONITORING PLATFORM")
    print("  Real Data Analysis Demo")
    print(SEPARATOR)
    print()

    # ── 1. AOI Summary ──────────────────────────────────────────────────
    print("[1] AREA OF INTEREST")
    print(f"  Region:     Chesapeake Bay Watershed")
    print(f"  BBox:       {CHESAPEAKE_BBOX}")
    print(f"  Valid:      {validate_bbox(CHESAPEAKE_BBOX)}")
    poly = bbox_to_polygon(CHESAPEAKE_BBOX)
    print(f"  Area:       {area_km2(poly):,.0f} km²")
    print(f"  Time range: {START_DATE.strftime('%Y-%m-%d')} → {END_DATE.strftime('%Y-%m-%d')}")
    print()

    # ── 2. NOAA Weather Stations ────────────────────────────────────────
    print(SEPARATOR)
    print("[2] NOAA WEATHER STATION DISCOVERY")
    print("  Querying api.weather.gov for stations in the Chesapeake region...")
    print()

    stations = await data_integration_service.get_noaa_stations(
        bbox=CHESAPEAKE_BBOX, limit=10,
    )

    if stations:
        print(f"  Found {len(stations)} stations:\n")
        print(f"  {'Station ID':<12} {'Name':<40} {'Lat':>8} {'Lon':>10} {'Elev (m)':>10}")
        print(f"  {'-'*12} {'-'*40} {'-'*8} {'-'*10} {'-'*10}")
        for s in stations:
            elev = f"{s['elevation_m']:.1f}" if s['elevation_m'] is not None else "N/A"
            name = (s['name'] or 'Unknown')[:40]
            print(f"  {s['station_id']:<12} {name:<40} {s['latitude']:>8.4f} {s['longitude']:>10.4f} {elev:>10}")
        print()

        # ── 3. Weather Observations ─────────────────────────────────────
        target_station = stations[0]
        print(SEPARATOR)
        print(f"[3] WEATHER OBSERVATIONS — Station: {target_station['station_id']}")
        print(f"  Name: {target_station['name']}")
        print(f"  Fetching last 7 days of observations...")
        print()

        observations = await data_integration_service.get_noaa_observations(
            station_id=target_station['station_id'],
            start_date=START_DATE,
            end_date=END_DATE,
            limit=20,
        )

        if observations:
            print(f"  Retrieved {len(observations)} observations:\n")
            print(f"  {'Timestamp':<28} {'Temp (°C)':>10} {'Humidity':>10} {'Wind (m/s)':>11} {'Description'}")
            print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*11} {'-'*20}")
            for obs in observations[:15]:
                ts = (obs['timestamp'] or '')[:25]
                temp = f"{obs['temperature_c']:.1f}" if obs['temperature_c'] is not None else "—"
                hum = f"{obs['humidity_pct']:.0f}%" if obs['humidity_pct'] is not None else "—"
                wind = f"{obs['wind_speed_ms']:.1f}" if obs['wind_speed_ms'] is not None else "—"
                desc = (obs['description'] or '—')[:25]
                print(f"  {ts:<28} {temp:>10} {hum:>10} {wind:>11} {desc}")

            # Compute simple analytics
            temps = [o['temperature_c'] for o in observations if o['temperature_c'] is not None]
            humids = [o['humidity_pct'] for o in observations if o['humidity_pct'] is not None]

            if temps:
                print(f"\n  ── Temperature Summary ──")
                print(f"    Min:  {min(temps):.1f} °C")
                print(f"    Max:  {max(temps):.1f} °C")
                print(f"    Mean: {sum(temps)/len(temps):.1f} °C")
                print(f"    Observations: {len(temps)}")

            if humids:
                print(f"\n  ── Humidity Summary ──")
                print(f"    Min:  {min(humids):.0f}%")
                print(f"    Max:  {max(humids):.0f}%")
                print(f"    Mean: {sum(humids)/len(humids):.0f}%")
        else:
            print("  No observations returned (station may be inactive).")
    else:
        print("  No NOAA stations found in bbox (API may be rate-limited).")

    print()

    # ── 4. USGS Water Monitoring Sites ──────────────────────────────────
    print(SEPARATOR)
    print("[4] USGS WATER MONITORING SITES")
    print("  Querying waterservices.usgs.gov for stream gauges...")
    print()

    sites = await data_integration_service.get_usgs_sites(
        bbox=CHESAPEAKE_BBOX, site_type="ST", limit=10,
    )

    if sites:
        print(f"  Found {len(sites)} stream gauge sites:\n")
        print(f"  {'Site ID':<16} {'Name':<44} {'Lat':>8} {'Lon':>10}")
        print(f"  {'-'*16} {'-'*44} {'-'*8} {'-'*10}")
        for s in sites:
            name = (s['name'] or 'Unknown')[:44]
            lat = f"{s['latitude']:.4f}" if s['latitude'] else "N/A"
            lon = f"{s['longitude']:.4f}" if s['longitude'] else "N/A"
            print(f"  {s['site_id']:<16} {name:<44} {lat:>8} {lon:>10}")
        print()

        # ── 5. Real-Time Stream Data ────────────────────────────────────
        target_site = sites[0]
        print(SEPARATOR)
        print(f"[5] REAL-TIME STREAM DATA — Site: {target_site['site_id']}")
        print(f"  Name: {target_site['name']}")
        print(f"  Fetching last 7 days (discharge, gage height, water temp)...")
        print()

        records = await data_integration_service.get_usgs_instantaneous_values(
            site_id=target_site['site_id'],
            parameter_codes=["00060", "00065", "00010"],
            period="P7D",
        )

        if records:
            # Group by parameter
            params = {}
            for r in records:
                pname = r['parameter'] or r['parameter_code']
                if pname not in params:
                    params[pname] = []
                params[pname].append(r)

            for pname, recs in params.items():
                values = [r['value'] for r in recs if r['value'] is not None and r['value'] != -999999.0]
                unit = recs[0].get('unit', '')
                if values:
                    print(f"  ── {pname} ({unit}) ──")
                    print(f"    Records:  {len(values)}")
                    print(f"    Min:      {min(values):.2f}")
                    print(f"    Max:      {max(values):.2f}")
                    print(f"    Mean:     {sum(values)/len(values):.2f}")
                    print(f"    Latest:   {values[-1]:.2f}")
                    # Show last 5 readings
                    print(f"    Last 5 readings:")
                    for r in recs[-5:]:
                        ts = (r['timestamp'] or '')[:19]
                        val = f"{r['value']:.2f}" if r['value'] is not None else "—"
                        print(f"      {ts}  →  {val} {unit}")
                    print()
        else:
            print("  No instantaneous values returned for this site.")
    else:
        print("  No USGS sites found in bbox.")

    print()

    # ── 6. Summary ──────────────────────────────────────────────────────
    print(SEPARATOR)
    print("[6] ANALYSIS SUMMARY")
    print()
    print("  Data sources queried:")
    print("    ✓ NOAA/NWS — Weather stations & observations")
    print("    ✓ USGS Water Services — Stream gauges & real-time readings")
    print()
    print("  Region analysed: Chesapeake Bay Watershed")
    print(f"  NOAA stations found: {len(stations)}")
    print(f"  USGS sites found:   {len(sites)}")
    print()
    print("  This demonstrates the GIOS backend's ability to:")
    print("    • Discover monitoring stations within a geographic region")
    print("    • Pull real-time environmental observations")
    print("    • Aggregate and summarise the data for analysis")
    print("    • Cross-reference multiple data sources")
    print()
    print(SEPARATOR)

    # ── Cleanup ─────────────────────────────────────────────────────────
    await data_integration_service.close()


if __name__ == "__main__":
    asyncio.run(main())

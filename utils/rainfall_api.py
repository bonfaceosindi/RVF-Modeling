"""
NASA POWER daily precipitation API client.

Fetches PRECTOTCORR (corrected total precipitation, mm/day) for any
latitude/longitude and date range — no API key required.
"""

from __future__ import annotations
import json
import numpy as np
from datetime import date, datetime as _dt
from typing import Any, Dict, Tuple

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# ── Known RVF outbreak locations — 3-level hierarchy ──────────────────────────
# Structure: Region → Country/Area → Event label → preset dict
OUTBREAK_HIERARCHY: Dict[str, Dict[str, Dict[str, Any]]] = {
    "East Africa": {
        "Kenya / Tanzania": {
            "2006–07  ·  Nairobi region": {
                "lat": -1.286, "lon": 36.820,
                "start": date(2006, 10, 1), "end": date(2007, 3, 31),
                "note": "Largest recorded outbreak: ~150 000 cases, ~60 000 livestock deaths.",
            },
        },
        "Uganda": {
            "2016  ·  Kabale district": {
                "lat": -1.25, "lon": 29.99,
                "start": date(2016, 5, 1), "end": date(2016, 10, 31),
                "note": "First confirmed RVF outbreak in livestock and humans in Uganda.",
            },
        },
        "Sudan": {
            "2007–08  ·  White Nile State": {
                "lat": 12.86, "lon": 30.22,
                "start": date(2007, 10, 1), "end": date(2008, 2, 29),
                "note": "Flood-associated outbreak along White Nile and Nile State.",
            },
        },
    },
    "Southern Africa": {
        "South Africa": {
            "2010  ·  Free State province": {
                "lat": -29.1, "lon": 26.2,
                "start": date(2010, 1, 1), "end": date(2010, 4, 30),
                "note": "Post-flood outbreak in Northern Cape and Free State provinces.",
            },
        },
    },
    "West Africa": {
        "Mauritania": {
            "2012  ·  Sahel / Senegal River": {
                "lat": 17.0, "lon": -13.0,
                "start": date(2012, 8, 1), "end": date(2012, 12, 31),
                "note": "Endemic region; recurrent outbreaks following Sahel rains.",
            },
        },
    },
    "Middle East": {
        "Yemen / Saudi Arabia": {
            "2000  ·  Hadramawt / Jizan": {
                "lat": 15.55, "lon": 44.01,
                "start": date(2000, 8, 1), "end": date(2000, 12, 31),
                "note": "First major RVF outbreak outside Africa; ~2 000 human cases.",
            },
        },
    },
    "Custom": {
        "— enter manually —": {
            "Custom coordinates": {
                "lat": 0.0, "lon": 38.0,
                "start": date(2006, 1, 1), "end": date(2006, 12, 31),
                "note": "Enter your own coordinates and date range below.",
            },
        },
    },
}

# Flat alias kept for any external code that referenced the old structure
OUTBREAK_PRESETS: Dict[str, Any] = {
    f"{country}  —  {event}": data
    for region, countries in OUTBREAK_HIERARCHY.items()
    for country, events in countries.items()
    for event, data in events.items()
}


def fetch_nasa_power(
    lat: float,
    lon: float,
    start: date,
    end: date,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fetch daily corrected precipitation (mm) from NASA POWER.

    Parameters
    ----------
    lat, lon : float
        Decimal degrees (lat: -90 to 90, lon: -180 to 180).
    start, end : datetime.date
        Inclusive date range. Maximum ~3 years recommended for performance.

    Returns
    -------
    rain : np.ndarray, shape (n_days,)
        Daily precipitation in mm.  Fill values (-999) are replaced with 0.
    meta : dict
        Location string, date range, total/peak rainfall, and number of days.

    Raises
    ------
    RuntimeError  if `requests` is not installed.
    requests.HTTPError  if the API returns a non-2xx status.
    """
    if not _REQUESTS_OK:
        raise RuntimeError(
            "'requests' is not installed. Add it to requirements.txt and redeploy."
        )
    if end < start:
        raise ValueError("end date must be >= start date.")

    resp = requests.get(
        _POWER_URL,
        params={
            "parameters": "PRECTOTCORR",
            "community":  "AG",
            "longitude":  lon,
            "latitude":   lat,
            "start":      start.strftime("%Y%m%d"),
            "end":        end.strftime("%Y%m%d"),
            "format":     "JSON",
        },
        timeout=40,
    )
    resp.raise_for_status()

    raw: Dict[str, float] = (
        resp.json()["properties"]["parameter"]["PRECTOTCORR"]
    )

    # Build ordered daily array — NASA POWER keys are YYYYMMDD strings
    n_days = (end - start).days + 1
    rain = np.zeros(n_days)
    for i, key in enumerate(sorted(raw.keys())):
        if i >= n_days:
            break
        v = raw[key]
        rain[i] = max(0.0, float(v)) if v != -999.0 else 0.0

    # Cardinal direction labels
    lat_label = f"{abs(lat):.3f}°{'N' if lat >= 0 else 'S'}"
    lon_label = f"{abs(lon):.3f}°{'E' if lon >= 0 else 'W'}"

    meta: Dict[str, Any] = {
        "lat":       lat,
        "lon":       lon,
        "location":  f"{lat_label}, {lon_label}",
        "start":     start.isoformat(),
        "end":       end.isoformat(),
        "n_days":    n_days,
        "total_mm":  float(rain.sum()),
        "peak_mm":   float(rain.max()),
        "peak_day":  int(rain.argmax()) + 1,
        "source":    "NASA POWER (PRECTOTCORR)",
    }
    return rain, meta


def fetch_chirps(
    lat: float,
    lon: float,
    start: date,
    end: date,
    poll_interval: float = 3.0,
    max_polls: int = 60,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fetch daily CHIRPS v2.0 precipitation from the ClimateSERV API.

    Resolution: ~0.05° (~5 km).  Coverage: 50°S–50°N, 1981–present.
    No API key required; job is submitted asynchronously and polled.

    Parameters
    ----------
    lat, lon : float
    start, end : datetime.date
    poll_interval : float
        Seconds between progress polls (default 3).
    max_polls : int
        Maximum poll attempts before raising TimeoutError (default 60 → ~3 min).

    Returns
    -------
    rain : np.ndarray, shape (n_days,)
    meta : dict  — same keys as fetch_nasa_power, plus source="CHIRPS v2.0"
    """
    import time

    if not _REQUESTS_OK:
        raise RuntimeError(
            "'requests' is not installed. Add it to requirements.txt and redeploy."
        )
    if end < start:
        raise ValueError("end date must be >= start date.")

    BASE = "https://climateserv.servirglobal.net/api"

    # ── Submit async job ───────────────────────────────────────────────────────
    resp = requests.post(
        f"{BASE}/submitDataRequest/",
        data={
            "datatype":      0,
            "begintime":     start.strftime("%m/%d/%Y"),
            "endtime":       end.strftime("%m/%d/%Y"),
            "intervaltype":  0,
            "operationtype": 5,
            "geometry":      json.dumps({"type": "Point", "coordinates": [lon, lat]}),
        },
        timeout=30,
    )
    resp.raise_for_status()
    request_id = resp.json()[0]

    # ── Poll until finished ────────────────────────────────────────────────────
    for attempt in range(max_polls):
        prog_resp = requests.get(
            f"{BASE}/getDataRequestProgress/{request_id}/", timeout=20
        )
        prog_resp.raise_for_status()
        prog = prog_resp.json()
        if prog and (
            float(prog[0].get("progress", 0)) >= 100
            or str(prog[0].get("status", "")).lower() == "finished"
        ):
            break
        time.sleep(poll_interval)
    else:
        raise TimeoutError(
            f"ClimateSERV job {request_id} did not finish within "
            f"{max_polls * poll_interval:.0f} s."
        )

    # ── Download result ────────────────────────────────────────────────────────
    data_resp = requests.get(
        f"{BASE}/getDataFromRequest/{request_id}/", timeout=40
    )
    data_resp.raise_for_status()
    raw = data_resp.json()

    # Response: [{"date": "MM/DD/YYYY", "value": [mm]}, ...]
    n_days = (end - start).days + 1
    rain = np.zeros(n_days)
    for item in raw:
        try:
            d = _dt.strptime(item["date"], "%m/%d/%Y").date()
            idx = (d - start).days
            vals = item.get("value", [])
            if 0 <= idx < n_days and vals:
                v = float(vals[0])
                rain[idx] = max(0.0, v) if v != -9999.0 else 0.0
        except (KeyError, ValueError):
            continue

    lat_label = f"{abs(lat):.3f}°{'N' if lat >= 0 else 'S'}"
    lon_label = f"{abs(lon):.3f}°{'E' if lon >= 0 else 'W'}"

    meta: Dict[str, Any] = {
        "lat":      lat,
        "lon":      lon,
        "location": f"{lat_label}, {lon_label}",
        "start":    start.isoformat(),
        "end":      end.isoformat(),
        "n_days":   n_days,
        "total_mm": float(rain.sum()),
        "peak_mm":  float(rain.max()),
        "peak_day": int(rain.argmax()) + 1,
        "source":   "CHIRPS v2.0 (ClimateSERV)",
    }
    return rain, meta

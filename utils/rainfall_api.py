"""
NASA POWER daily precipitation API client.

Fetches PRECTOTCORR (corrected total precipitation, mm/day) for any
latitude/longitude and date range — no API key required.
"""

from __future__ import annotations
import numpy as np
from datetime import date
from typing import Any, Dict, Tuple

try:
    import requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# ── Known RVF outbreak locations ───────────────────────────────────────────────
OUTBREAK_PRESETS: Dict[str, Dict[str, Any]] = {
    "East Africa 2006–07  (Kenya / Tanzania)": {
        "lat": -1.286, "lon": 36.820,
        "start": date(2006, 10, 1), "end": date(2007, 3, 31),
        "note": "Largest recorded outbreak: ~150 000 cases, ~60 000 deaths (livestock).",
    },
    "South Africa 2010  (Free State)": {
        "lat": -29.1, "lon": 26.2,
        "start": date(2010, 1, 1), "end": date(2010, 4, 30),
        "note": "Post-flood outbreak in Northern Cape and Free State provinces.",
    },
    "Yemen / Saudi Arabia 2000": {
        "lat": 15.55, "lon": 44.01,
        "start": date(2000, 8, 1), "end": date(2000, 12, 31),
        "note": "First major outbreak outside Africa; ~2 000 human cases.",
    },
    "Mauritania 2012  (West Africa)": {
        "lat": 17.0, "lon": -13.0,
        "start": date(2012, 8, 1), "end": date(2012, 12, 31),
        "note": "Endemic region; recurrent outbreaks following Sahel rains.",
    },
    "Sudan 2007–08  (White Nile)": {
        "lat": 12.86, "lon": 30.22,
        "start": date(2007, 10, 1), "end": date(2008, 2, 29),
        "note": "White Nile and Nile State flood-associated outbreak.",
    },
    "Uganda 2016  (Kabale)": {
        "lat": -1.25, "lon": 29.99,
        "start": date(2016, 5, 1), "end": date(2016, 10, 31),
        "note": "First confirmed Uganda outbreak in livestock and humans.",
    },
    "Custom location": {
        "lat": 0.0, "lon": 38.0,
        "start": date(2006, 1, 1), "end": date(2006, 12, 31),
        "note": "Enter your own coordinates and date range.",
    },
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

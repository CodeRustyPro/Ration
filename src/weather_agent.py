"""
NWS Weather Agent — Heat/Cold Stress DMI Adjustment.

Uses the free NWS API (api.weather.gov, no API key required) to fetch
7-day forecasts and compute Temperature Humidity Index (THI) for
stress-based DMI adjustments.

Science:
  • Heat stress (THI > 72): DMI reduced ~0.45 kg/d per THI unit above 72
    (global meta-analysis across beef cattle studies)
  • Cold stress (wind chill < LCT): DMI increased +1%/°F below LCT
    (Iowa State Extension, NASEM guidelines)
  • LCT defaults: 32°F dry winter coat, 59°F wet coat
  • Mud/rain: 10-15% DMI reduction (Kansas State research)

Cache: data/weather/ (JSON, 3-hour TTL)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

CACHE_DIR = Path("data/weather")
CACHE_TTL_SECONDS = 3 * 3600  # 3 hours

NWS_BASE = "https://api.weather.gov"
NWS_HEADERS = {
    "User-Agent": "(BrasIND Feed Optimizer, hackathon@uiuc.edu)",
    "Accept": "application/geo+json",
}

# Default location: Champaign, IL (UIUC hackathon)
DEFAULT_LAT = 40.1164
DEFAULT_LON = -88.2434


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HourlyForecast:
    """Single hourly forecast data point."""
    time: str           # ISO timestamp
    temp_f: float       # Temperature (°F)
    humidity: float     # Relative humidity (%)
    wind_mph: float     # Wind speed (mph)
    rain: bool          # Precipitation expected
    short_forecast: str # e.g. "Partly Cloudy"
    thi: float = 0.0    # Temperature Humidity Index
    wind_chill_f: float = 0.0  # Wind chill (°F)


@dataclass
class StressResult:
    """Aggregated stress assessment for the next 24-72 hours."""
    dmi_adjustment_factor: float = 1.0   # Multiplier on baseline DMI
    alert_level: str = "normal"          # normal, caution, danger, emergency
    alert_emoji: str = "🟢"
    farmer_message: str = ""             # Plain English summary
    suggested_changes: List[str] = field(default_factory=list)
    avg_thi_24h: float = 0.0
    min_wind_chill_24h: float = 70.0
    rain_hours_72h: int = 0
    hourly_forecasts: List[HourlyForecast] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# THI and wind chill calculations
# ---------------------------------------------------------------------------

def compute_thi(temp_f: float, rh_pct: float) -> float:
    """Temperature Humidity Index for cattle stress assessment.

    Using the NRC formula for Fahrenheit:
    THI = Tdb – [0.55 – (0.55 x RH)] x (Tdb – 58)
    where Tdb is dry-bulb temperature (°F), RH is 0-1 fraction.

    Thresholds (beef cattle):
      < 72: No stress
      72-79: Mild stress (alert)
      79-84: Moderate stress (danger)
      > 84: Severe stress (emergency)
    """
    rh_frac = rh_pct / 100.0
    return temp_f - (0.55 - 0.55 * rh_frac) * (temp_f - 58)


def compute_wind_chill(temp_f: float, wind_mph: float) -> float:
    """NWS Wind Chill Index (2001 formula).

    Only valid for temp ≤ 50°F and wind > 3 mph.
    """
    if temp_f > 50 or wind_mph <= 3:
        return temp_f
    wc = (35.74 + 0.6215 * temp_f
          - 35.75 * (wind_mph ** 0.16)
          + 0.4275 * temp_f * (wind_mph ** 0.16))
    return round(wc, 1)


# ---------------------------------------------------------------------------
# Stress assessment engine
# ---------------------------------------------------------------------------

def assess_stress(
    hourly: List[HourlyForecast],
    lower_critical_temp: float = 32.0,
    coat_condition: str = "dry",
) -> StressResult:
    """Assess heat/cold/mud stress from hourly forecasts.

    Parameters
    ----------
    hourly              List of HourlyForecast objects (next 72h ideally).
    lower_critical_temp LCT in °F. Default 32°F for dry winter coat.
                        Use 59°F for wet coat.
    coat_condition      "dry" or "wet" — affects cold stress multiplier.

    Returns
    -------
    StressResult with DMI adjustment factor, alerts, and farmer messages.
    """
    if not hourly:
        return StressResult(farmer_message="No weather data available.")

    # Use 32°F for dry coat, 59°F for wet coat as LCT
    if coat_condition == "wet":
        lower_critical_temp = 59.0

    # Separate into time windows
    next_24h = hourly[:24] if len(hourly) >= 24 else hourly
    next_72h = hourly[:72] if len(hourly) >= 72 else hourly

    # --- Heat stress assessment (THI-based) ---
    this_24h = [h.thi for h in next_24h]
    avg_thi = sum(this_24h) / len(this_24h) if this_24h else 65.0
    max_thi = max(this_24h) if this_24h else 65.0
    hours_above_72 = sum(1 for t in this_24h if t > 72)

    # --- Cold stress assessment ---
    wind_chills = [h.wind_chill_f for h in next_24h]
    min_wc = min(wind_chills) if wind_chills else 50.0
    hours_below_lct = sum(1 for wc in wind_chills if wc < lower_critical_temp)

    # --- Rain/mud assessment ---
    rain_hours = sum(1 for h in next_72h if h.rain)

    # --- Compute DMI adjustment ---
    dmi_factor = 1.0
    alerts = []
    changes = []

    # Heat stress: -0.45 kg/d per THI unit above 72 for beef cattle
    # Convert to fractional adjustment relative to typical ~8-10 kg/d DMI
    LB_PER_KG = 2.20462
    if avg_thi > 72:
        thi_excess = avg_thi - 72
        # 0.45 kg/d per unit ≈ 1.0 lb/d per unit for typical ~17.6 lb/d DMI
        # = ~5.7% per THI unit. Cap at 30%.
        heat_reduction = min(0.30, thi_excess * 0.057)
        dmi_factor *= (1.0 - heat_reduction)

        if avg_thi > 84:
            alerts.append("🔴 **SEVERE HEAT STRESS** — THI above 84")
            changes.append("Add electrolytes and extra water access")
            changes.append("Shift feeding to evening (after 6 PM)")
            changes.append("Increase roughage to reduce heat of digestion")
        elif avg_thi > 79:
            alerts.append("🟠 **MODERATE HEAT STRESS** — THI 79-84")
            changes.append("Consider shifting primary feeding to evening")
            changes.append("Ensure unlimited clean water access")
        elif avg_thi > 72:
            alerts.append("🟡 **MILD HEAT STRESS** — THI 72-79")
            changes.append("Monitor water consumption closely")

    # Cold stress: +1% NEm per °F below LCT (dry coat), +2% (wet coat)
    if min_wc < lower_critical_temp:
        degrees_below = lower_critical_temp - min_wc
        cold_multiplier = 0.02 if coat_condition == "wet" else 0.01
        cold_increase = min(0.50, degrees_below * cold_multiplier)  # cap at +50%
        dmi_factor *= (1.0 + cold_increase)

        if degrees_below > 30:
            alerts.append("🔴 **SEVERE COLD STRESS** — wind chill severely below LCT")
            changes.append("Increase energy density: add corn grain or fat supplement")
            changes.append("Provide windbreaks and dry bedding")
        elif degrees_below > 15:
            alerts.append("🟠 **MODERATE COLD STRESS** — wind chill well below LCT")
            changes.append("Increase feed amount 15-25%")
        else:
            alerts.append("🟡 **MILD COLD STRESS** — wind chill near LCT")
            changes.append("Monitor feed consumption for increased intake")

    # Rain/mud: 10-15% DMI reduction from mud (Kansas State)
    if rain_hours > 24:
        mud_reduction = 0.12  # 12% average for persistent rain
        dmi_factor *= (1.0 - mud_reduction)
        alerts.append("🟤 **MUD STRESS** — extended rain forecast reduces intake")
        changes.append("Consider adding extra feedbunk space and dry footing")

    # Determine alert level
    if any("SEVERE" in a for a in alerts):
        level, emoji = "emergency", "🔴"
    elif any("MODERATE" in a for a in alerts):
        level, emoji = "danger", "🟠"
    elif alerts:
        level, emoji = "caution", "🟡"
    else:
        level, emoji = "normal", "🟢"
        alerts.append("🟢 **No stress detected** — normal feeding conditions")

    # Build farmer message
    dmi_pct = (dmi_factor - 1.0) * 100
    if abs(dmi_pct) > 1:
        dmi_msg = f"Recommended DMI adjustment: **{dmi_pct:+.0f}%**"
    else:
        dmi_msg = "No DMI adjustment needed"

    farmer_message = (
        f"{' | '.join(alerts)}\n\n"
        f"24h avg THI: {avg_thi:.0f} | Min wind chill: {min_wc:.0f}°F | "
        f"Rain hours (72h): {rain_hours}\n\n"
        f"{dmi_msg}"
    )

    return StressResult(
        dmi_adjustment_factor=round(dmi_factor, 3),
        alert_level=level,
        alert_emoji=emoji,
        farmer_message=farmer_message,
        suggested_changes=changes,
        avg_thi_24h=round(avg_thi, 1),
        min_wind_chill_24h=round(min_wc, 1),
        rain_hours_72h=rain_hours,
        hourly_forecasts=hourly[:72],
        metadata={"lower_critical_temp": lower_critical_temp, "coat": coat_condition},
    )


# ---------------------------------------------------------------------------
# NWS API client
# ---------------------------------------------------------------------------

def _fetch_nws_grid(lat: float, lon: float) -> Optional[str]:
    """Get the hourly forecast URL from NWS points endpoint."""
    cache_path = CACHE_DIR / f"grid_{lat:.4f}_{lon:.4f}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
            return data.get("forecast_hourly_url")

    try:
        url = f"{NWS_BASE}/points/{lat},{lon}"
        resp = requests.get(url, headers=NWS_HEADERS, timeout=10)
        resp.raise_for_status()
        props = resp.json().get("properties", {})
        hourly_url = props.get("forecastHourly")

        if hourly_url:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump({
                    "forecast_hourly_url": hourly_url,
                    "city": props.get("relativeLocation", {}).get("properties", {}).get("city", ""),
                    "state": props.get("relativeLocation", {}).get("properties", {}).get("state", ""),
                    "fetched_at": datetime.now().isoformat(),
                }, f)
        return hourly_url
    except Exception as e:
        print(f"⚠️ NWS grid lookup failed: {e}")
        return None


def fetch_hourly_forecast(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    use_live: bool = False,
) -> List[HourlyForecast]:
    """Fetch hourly forecast from NWS and parse into HourlyForecast objects."""
    cache_path = CACHE_DIR / f"hourly_{lat:.4f}_{lon:.4f}.json"

    # Check cache freshness
    if not use_live and cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            fetched_at = datetime.fromisoformat(cached.get("fetched_at", "2000-01-01"))
            if (datetime.now() - fetched_at).total_seconds() < CACHE_TTL_SECONDS:
                return _parse_nws_periods(cached.get("periods", []))
        except Exception:
            pass

    # Get grid URL
    hourly_url = _fetch_nws_grid(lat, lon)
    if not hourly_url:
        return _load_cached_hourly(cache_path)

    try:
        resp = requests.get(hourly_url, headers=NWS_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        periods = data.get("properties", {}).get("periods", [])

        # Cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({
                "fetched_at": datetime.now().isoformat(),
                "periods": periods,
            }, f, default=str)

        return _parse_nws_periods(periods)
    except Exception as e:
        print(f"⚠️ NWS hourly forecast failed: {e}")
        return _load_cached_hourly(cache_path)


def _load_cached_hourly(cache_path: Path) -> List[HourlyForecast]:
    """Load cached hourly data as fallback."""
    if not cache_path.exists():
        return []
    try:
        with open(cache_path) as f:
            cached = json.load(f)
        return _parse_nws_periods(cached.get("periods", []))
    except Exception:
        return []


def _parse_nws_periods(periods: List[Dict]) -> List[HourlyForecast]:
    """Parse NWS hourly periods into HourlyForecast objects."""
    results = []
    for p in periods:
        temp_f = float(p.get("temperature", 65))
        # NWS may provide temp in Celsius
        if p.get("temperatureUnit") == "C":
            temp_f = temp_f * 9 / 5 + 32

        humidity = float(p.get("relativeHumidity", {}).get("value", 50) or 50)
        wind_str = str(p.get("windSpeed", "5 mph"))
        # Parse wind speed from string like "10 mph" or "10 to 15 mph"
        wind_parts = wind_str.replace("mph", "").strip().split("to")
        try:
            wind_mph = float(wind_parts[-1].strip())
        except (ValueError, IndexError):
            wind_mph = 5.0

        short_fc = str(p.get("shortForecast", ""))
        rain = any(w in short_fc.lower() for w in
                    ["rain", "shower", "storm", "drizzle", "precip", "snow", "sleet"])

        thi = compute_thi(temp_f, humidity)
        wc = compute_wind_chill(temp_f, wind_mph)

        results.append(HourlyForecast(
            time=p.get("startTime", ""),
            temp_f=round(temp_f, 1),
            humidity=round(humidity, 1),
            wind_mph=round(wind_mph, 1),
            rain=rain,
            short_forecast=short_fc,
            thi=round(thi, 1),
            wind_chill_f=round(wc, 1),
        ))

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_weather_stress(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    use_live: bool = False,
    coat_condition: str = "dry",
) -> StressResult:
    """Main entry point: fetch weather and return stress assessment.

    Parameters
    ----------
    lat, lon        Feedlot coordinates (default: Champaign, IL)
    use_live        If True, call NWS API; if False, use cache
    coat_condition  "dry" or "wet" — affects cold stress LCT

    Returns
    -------
    StressResult with dmi_adjustment_factor, alerts, and recommendations.
    """
    hourly = fetch_hourly_forecast(lat, lon, use_live=use_live)
    result = assess_stress(hourly, coat_condition=coat_condition)

    # Add location metadata
    grid_cache = CACHE_DIR / f"grid_{lat:.4f}_{lon:.4f}.json"
    if grid_cache.exists():
        try:
            with open(grid_cache) as f:
                gdata = json.load(f)
            result.metadata["city"] = gdata.get("city", "")
            result.metadata["state"] = gdata.get("state", "")
        except Exception:
            pass

    result.metadata["lat"] = lat
    result.metadata["lon"] = lon
    return result


def get_location_name(lat: float, lon: float) -> str:
    """Get city/state name for display from cached grid data."""
    grid_cache = CACHE_DIR / f"grid_{lat:.4f}_{lon:.4f}.json"
    if grid_cache.exists():
        try:
            with open(grid_cache) as f:
                gdata = json.load(f)
            city = gdata.get("city", "")
            state = gdata.get("state", "")
            if city and state:
                return f"{city}, {state}"
        except Exception:
            pass
    return f"{lat:.2f}°N, {abs(lon):.2f}°W"

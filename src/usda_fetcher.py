"""
USDA AMS Market News (MARS) API client.

Fetches feedstuff and grain price data from the MARS API using the slugs
documented in the spec. Implements three-tier caching:
  Tier 1: st.cache_data (in-memory within a Streamlit session)
  Tier 2: JSON file on disk (pre-populated Friday night)
  Tier 3: Live MARS API call

Slug reference:
  3511 – National Grain & Oilseed Processor Feedstuff (weekly)
  3510 – National Animal By-Product Feedstuff (weekly)
  3512 – National Mill-Feeds & Miscellaneous Feedstuff (weekly)
  3192 – Illinois Grain Bids (daily)
  3618 – National Weekly Grain Co-Products (DDG, wet distillers)
  3668/3669/3667 – Monthly feedstuff averages
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.auth import HTTPBasicAuth

MARS_BASE = "https://marsapi.ams.usda.gov/services/v1.2/reports"
CACHE_DIR = Path("data/usda")

# Slug metadata
SLUGS = {
    3511: {
        "name": "National Grain & Oilseed Processor Feedstuff",
        "freq": "Weekly",
        "key_commodities": ["Soybean Meal", "Canola Meal", "Cottonseed Meal", "DDGS"],
        "cache_file": "slug_3511.json",
    },
    3510: {
        "name": "National Animal By-Product Feedstuff",
        "freq": "Weekly",
        "key_commodities": ["Meat and Bone Meal", "Blood Meal", "Fish Meal", "Feather Meal"],
        "cache_file": "slug_3510.json",
    },
    3512: {
        "name": "National Mill-Feeds & Miscellaneous Feedstuff",
        "freq": "Weekly",
        "key_commodities": ["Wheat Middlings", "Corn Gluten Feed", "Alfalfa Meal", "Rice Bran"],
        "cache_file": "slug_3512.json",
    },
    3192: {
        "name": "Illinois Grain Bids",
        "freq": "Daily",
        "key_commodities": ["Corn", "Soybeans", "Wheat"],
        "cache_file": "slug_3192.json",
    },
    3618: {
        "name": "National Weekly Grain Co-Products",
        "freq": "Weekly",
        "key_commodities": ["Distillers Dried Grains", "Wet Distillers Grains", "Distillers Corn Oil"],
        "cache_file": "slug_3618.json",
    },
}


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def _mars_request(
    slug_id: int,
    api_key: str,
    last_days: int = 90,
    extra_params: Optional[Dict] = None,
) -> Dict:
    """
    Raw MARS API request with retry on 429.
    Returns parsed JSON dict or raises on failure.
    """
    params: Dict[str, Any] = {
        "allSections": "true",
        "lastDays": str(last_days),
    }
    if extra_params:
        params.update(extra_params)

    url = f"{MARS_BASE}/{slug_id}"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                url,
                auth=HTTPBasicAuth(api_key, ""),
                params=params,
                timeout=20,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError(f"MARS API failed for slug {slug_id} after {max_retries} retries")


# ---------------------------------------------------------------------------
# Three-tier caching
# ---------------------------------------------------------------------------

def fetch_slug(
    slug_id: int,
    api_key: Optional[str],
    last_days: int = 90,
    use_live: bool = False,
    force_refresh: bool = False,
) -> Optional[Dict]:
    """
    Fetch data for a given slug using three-tier caching.

    Parameters
    ----------
    slug_id     MARS slug ID (e.g. 3511).
    api_key     USDA API key string. If None, falls back to cache only.
    last_days   Rolling window for live API calls.
    use_live    If True, attempt live API call first; if False use cache.
    force_refresh  If True and use_live, always call API even if cache is fresh.

    Returns
    -------
    dict with {slug_id, fetched_at, data: [...list of records...]} or None.
    """
    cache_path = CACHE_DIR / SLUGS.get(slug_id, {}).get("cache_file", f"slug_{slug_id}.json")

    # Tier 1: try live API if enabled
    if use_live and api_key:
        try:
            raw = _mars_request(slug_id, api_key, last_days=last_days)
            # API returns either a list of section dicts or {"results": [...]}
            if isinstance(raw, list):
                sections = raw
            elif isinstance(raw, dict):
                sections = raw.get("results", [raw])
            else:
                sections = []
            result = {
                "slug_id": slug_id,
                "fetched_at": datetime.now().isoformat(),
                "data": sections,
            }
            _save_cache(cache_path, result)
            return result
        except Exception as e:
            # Fall through to Tier 2
            print(f"⚠️  MARS API slug {slug_id} failed: {e}. Falling back to cache.")

    # Tier 2: disk cache
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    # Tier 3: no data available
    return None


def fetch_all_slugs(
    api_key: Optional[str],
    use_live: bool = False,
    last_days: int = 90,
) -> Dict[int, Optional[Dict]]:
    """Fetch all configured slugs. Returns {slug_id: result_or_none}."""
    results = {}
    for slug_id in SLUGS:
        results[slug_id] = fetch_slug(slug_id, api_key, last_days=last_days, use_live=use_live)
        if use_live:
            time.sleep(1)   # rate limit: 1 req/sec
    return results


def _save_cache(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, default=str, indent=2)


# ---------------------------------------------------------------------------
# Price extraction helpers
# ---------------------------------------------------------------------------

# Maps USDA commodity name → our internal ingredient key.
# Uses exact/prefix matching to avoid false positives.
# Each entry: (match_string, require_all_of, exclude_if_contains, ingredient_key)
# Simple dict for exact substring; tighter matching done in _match_commodity().
_COMMODITY_RULES: List[tuple] = [
    # (required_substr, excluded_substrs, ingredient_key)
    # --- Protein supplements ---
    ("soybean meal",    [],                    "SoybeanMeal"),
    ("soy meal",        [],                    "SoybeanMeal"),
    ("canola meal",     [],                    "CanolaMe"),
    ("cottonseed meal", [],                    "CottonMeal"),
    # --- Energy/grain byproducts ---
    ("distillers dried grains", [],            "DDGS"),
    ("corn distillers",         ["oil", "wet"],"DDGS"),   # dry DDGS
    # --- Grain bids (slug 3192) - must be plain corn/grain corn ---
    ("corn",        ["gluten", "distillers", "oil", "hominy", "silage", "steep"], "CornGrain"),
    # --- Forages ---
    ("alfalfa",     ["seed"],                  "AlfalfaHay"),
    # --- Mill feeds ---
    ("wheat middlings", [],                    "WheatMid"),
    ("wheat midds",     [],                    "WheatMid"),
]


def extract_prices(slug_data: Optional[Dict]) -> Dict[str, List[Dict]]:
    """
    Extract price records from a slug result dict.

    Returns {ingredient_key: [{date, price, unit, region, commodity_raw}, ...]}
    sorted by date descending.
    """
    if not slug_data or not slug_data.get("data"):
        return {}

    # The MARS API returns a list of section dicts.
    # Flatten all "Report Detail" rows from all sections.
    sections = slug_data["data"]
    detail_rows: List[Dict] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        section_name = section.get("reportSection", "")
        if "detail" in section_name.lower() or "detail" in str(section_name).lower():
            detail_rows.extend(section.get("results", []))
    # If no detail section found, try all results
    if not detail_rows:
        for section in sections:
            if isinstance(section, dict):
                rows = section.get("results", [])
                if rows and rows[0].get("commodity"):  # has commodity data
                    detail_rows.extend(rows)

    prices: Dict[str, List[Dict]] = {}

    for rec in detail_rows:
        commodity_raw = str(rec.get("commodity", "") or "").lower().strip()
        ingredient_key = _match_commodity(commodity_raw)
        if not ingredient_key:
            continue

        # Primary price field: avg_price ($/ton).
        # Fallback: average of price_min + price_max.
        price_val = None
        for field in ["avg_price", "average", "price", "wtd_avg", "weighted_avg"]:
            val = rec.get(field)
            if val is not None and val != "":
                try:
                    price_val = float(val)
                    break
                except (ValueError, TypeError):
                    continue
        if price_val is None:
            lo = rec.get("price_min")
            hi = rec.get("price_max")
            try:
                if lo is not None and hi is not None:
                    price_val = (float(lo) + float(hi)) / 2
            except (ValueError, TypeError):
                pass
        if price_val is None:
            continue

        date_str = (
            rec.get("report_date")
            or rec.get("report_begin_date")
            or rec.get("report_end_date")
        )
        unit = str(rec.get("price_unit", "$ Per Ton") or "$ Per Ton")
        region = str(rec.get("trade Loc", rec.get("region", "")) or "")

        entry = {
            "date": date_str,
            "price": price_val,
            "unit": unit,
            "region": region,
            "commodity_raw": commodity_raw,
            "variety": rec.get("variety", ""),
        }

        if ingredient_key not in prices:
            prices[ingredient_key] = []
        prices[ingredient_key].append(entry)

    # Sort by date descending (parse MM/DD/YYYY → sortable key)
    def _date_key(r: Dict) -> str:
        d = r.get("date") or ""
        # Convert MM/DD/YYYY → YYYY-MM-DD for correct lexicographic sort
        parts = d.split("/")
        if len(parts) == 3:
            return f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
        return d

    for key in prices:
        prices[key].sort(key=_date_key, reverse=True)

    return prices


def _match_commodity(name: str) -> Optional[str]:
    """Match a lowercase commodity name to an ingredient key using exclusion rules."""
    for required, excluded, key in _COMMODITY_RULES:
        if required in name:
            if any(excl in name for excl in excluded):
                continue
            return key
    return None


def get_latest_prices(
    all_slug_data: Dict[int, Optional[Dict]],
) -> Dict[str, float]:
    """
    Extract the single most-recent price ($/ton) for each ingredient key
    across all slugs. Returns {ingredient_key: price_per_ton}.
    """
    latest: Dict[str, float] = {}

    for slug_id, slug_result in all_slug_data.items():
        prices = extract_prices(slug_result)
        for ing_key, records in prices.items():
            if not records:
                continue
            rec = records[0]  # most recent

            # Normalize to $/ton (2000 lb)
            price = rec["price"]
            unit = rec["unit"].lower()

            if "cwt" in unit:
                price = price * 20           # $/cwt × 20 = $/ton
            elif "/lb" in unit or "pound" in unit or "per lb" in unit:
                price = price * 2000
            elif "bu" in unit or "bushel" in unit:
                if "corn" in rec["commodity_raw"]:
                    price = price / 56 * 2000   # 56 lb/bu corn
                else:
                    price = price / 60 * 2000   # 60 lb/bu soybeans/wheat
            # "$ per ton", "$/ton", "$ Per Ton" → already $/ton

            if ing_key not in latest:
                latest[ing_key] = price

    return latest


def get_price_history(
    all_slug_data: Dict[int, Optional[Dict]],
    ingredient_key: str,
) -> List[Dict]:
    """
    Return time-series price records for a given ingredient across all slugs.
    [{date, price_per_ton}, ...]
    """
    records: List[Dict] = []
    for slug_result in all_slug_data.values():
        prices = extract_prices(slug_result)
        if ingredient_key in prices:
            records.extend(prices[ingredient_key])

    # Deduplicate by date, keeping most recent record for each date
    seen: Dict[str, float] = {}
    for rec in records:
        d = rec.get("date") or ""
        if d and d not in seen:
            seen[d] = rec["price"]

    result = [{"date": d, "price": p} for d, p in sorted(seen.items())]
    return result


# ---------------------------------------------------------------------------
# Hardcoded fallback prices (when API and cache unavailable)
# Used during development / offline demo mode.
# Based on USDA AMS approximate market prices March 2026.
# ---------------------------------------------------------------------------

FALLBACK_PRICES_PER_TON: Dict[str, float] = {
    "CornGrain":   195.0,   # $/ton DM basis (~$4.80/bu)
    "SoybeanMeal": 380.0,   # $/ton DM basis (48% SBM)
    "DDGS":        185.0,   # $/ton DM basis
    "CornSilage":  120.0,   # $/ton DM basis (Iowa State: 8-10× corn bu price)
    "AlfalfaHay":  250.0,   # $/ton DM basis (good quality)
    "GrassHay":    150.0,   # $/ton DM basis
    "Urea":        580.0,   # $/ton
    "Limestone":   55.0,    # $/ton
    "MineralPremix": 1800.0,
}

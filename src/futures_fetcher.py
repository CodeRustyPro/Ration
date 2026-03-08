"""
yfinance agricultural futures fetcher.

Provides:
  fetch_continuous()   – OHLCV history for continuous contracts (ZC=F, ZM=F …)
  fetch_contract()     – Single contract month (e.g. ZCN26.CBT)
  compute_carry()      – Carry spread between two contract months
  get_all_futures()    – Batch fetch all relevant contracts with caching

Notes:
  • yfinance =F contracts use UN-ADJUSTED rollover; gaps exist at roll dates.
    For ML training, stitch individual months. For display, =F is acceptable.
  • No hay futures exist on any exchange.
  • 10-minute delay on all CBOT/CME ag futures via yfinance.
  • Cache path: data/yfinance/
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

CACHE_DIR = Path("data/yfinance")

# ---------------------------------------------------------------------------
# Ticker definitions
# ---------------------------------------------------------------------------

CONTINUOUS_TICKERS: Dict[str, Dict] = {
    "corn": {
        "ticker": "ZC=F",
        "name": "Corn (CBOT)",
        "units": "cents/bu",
        "lbs_per_bu": 56,
        "ingredient_key": "CornGrain",
    },
    "soybeans": {
        "ticker": "ZS=F",
        "name": "Soybeans (CBOT)",
        "units": "cents/bu",
        "lbs_per_bu": 60,
        "ingredient_key": None,
    },
    "soybean_meal": {
        "ticker": "ZM=F",
        "name": "Soybean Meal (CBOT)",
        "units": "$/short ton",
        "lbs_per_bu": None,
        "ingredient_key": "SoybeanMeal",
    },
    "wheat": {
        "ticker": "ZW=F",
        "name": "Wheat SRW (CBOT)",
        "units": "cents/bu",
        "lbs_per_bu": 60,
        "ingredient_key": None,
    },
    "feeder_cattle": {
        "ticker": "GF=F",
        "name": "Feeder Cattle (CME)",
        "units": "cents/lb",
        "lbs_per_bu": None,
        "ingredient_key": None,
    },
    "live_cattle": {
        "ticker": "LE=F",
        "name": "Live Cattle (CME)",
        "units": "cents/lb",
        "lbs_per_bu": None,
        "ingredient_key": None,
    },
}

# CBOT month codes for constructing individual contract tickers
MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}

# Corn trades H/K/N/U/Z (Mar/May/Jul/Sep/Dec)
CORN_CONTRACT_MONTHS = [3, 5, 7, 9, 12]


# ---------------------------------------------------------------------------
# Core fetch functions
# ---------------------------------------------------------------------------

def fetch_continuous(
    commodity: str,
    period: str = "6mo",
    use_live: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data for a continuous contract.

    Parameters
    ----------
    commodity   Key in CONTINUOUS_TICKERS (e.g. "corn", "soybean_meal").
    period      yfinance period string: "1mo", "3mo", "6mo", "1y", "max".
    use_live    If False, return cached data if available.

    Returns
    -------
    DataFrame with columns [Open, High, Low, Close, Volume] indexed by Date,
    plus a 'close_per_ton' column normalized to $/short ton.
    Returns None if data unavailable.
    """
    if commodity not in CONTINUOUS_TICKERS:
        raise ValueError(f"Unknown commodity '{commodity}'. Options: {list(CONTINUOUS_TICKERS)}")

    meta = CONTINUOUS_TICKERS[commodity]
    ticker = meta["ticker"]
    cache_path = CACHE_DIR / f"{commodity}_continuous.json"

    if not use_live and cache_path.exists():
        return _load_df_cache(cache_path, commodity)

    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return _load_df_cache(cache_path, commodity)

        # Flatten multi-index columns if present (yfinance ≥0.2.x)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)

        # Add $/ton column for normalized comparison
        df["close_per_ton"] = _to_per_ton(df["Close"], meta)

        _save_df_cache(df, cache_path, commodity)
        return df

    except Exception as e:
        return _load_df_cache(cache_path, commodity)


def fetch_contract(ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
    """
    Fetch a specific contract month (e.g. 'ZCN26.CBT').
    Returns OHLCV DataFrame or None.
    """
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return None


def make_contract_ticker(root: str, month: int, year: int, exchange: str = "CBT") -> str:
    """
    Construct a yfinance contract ticker.
    Example: make_contract_ticker("ZC", 7, 26) → "ZCN26.CBT"
    """
    code = MONTH_CODES.get(month, "N")
    yr = str(year)[-2:]
    return f"{root}{code}{yr}.{exchange}"


# ---------------------------------------------------------------------------
# Carry spread analysis
# ---------------------------------------------------------------------------

def compute_carry(
    nearby_ticker: str,
    deferred_ticker: str,
    period: str = "3mo",
) -> Optional[pd.DataFrame]:
    """
    Compute the calendar spread (carry) between two contract months.

    carry_cents  = deferred - nearby  (in contract units)
    carry_pct    = carry / nearby × 100

    Positive carry (contango) → ample supply, no urgency to buy.
    Negative carry (backwardation/inverted) → tight supply, buy/contract now.
    """
    nearby_df = fetch_contract(nearby_ticker, period=period)
    deferred_df = fetch_contract(deferred_ticker, period=period)

    if nearby_df is None or deferred_df is None:
        return None

    spread = pd.DataFrame({
        "nearby": nearby_df["Close"],
        "deferred": deferred_df["Close"],
    }).dropna()

    if spread.empty:
        return None

    spread["carry"] = spread["deferred"] - spread["nearby"]
    spread["carry_pct"] = (spread["carry"] / spread["nearby"]) * 100
    spread["signal"] = spread["carry"].apply(
        lambda x: "contango" if x > 0 else "backwardation"
    )
    return spread


def get_corn_carry_recommendation(bw_lb: float = 800, adg_lb: float = 3.0) -> Dict:
    """
    Compute current corn carry signal and generate a forward-contracting
    recommendation based on the carry structure.

    Uses the two front corn contract months (H, K, N, U, Z).
    """
    now = datetime.now()
    year = now.year
    year2d = year % 100

    # Find next two active corn contract months
    active = [m for m in CORN_CONTRACT_MONTHS if m >= now.month]
    if len(active) < 2:
        active = CORN_CONTRACT_MONTHS[:2]
        year2d += 1

    m1, m2 = active[0], active[1]
    ticker1 = make_contract_ticker("ZC", m1, year2d)
    ticker2 = make_contract_ticker("ZC", m2, year2d)

    carry_df = compute_carry(ticker1, ticker2, period="2mo")

    result = {
        "nearby_ticker": ticker1,
        "deferred_ticker": ticker2,
        "carry_df": carry_df,
        "recommendation": "N/A (no data)",
        "signal": "neutral",
    }

    if carry_df is not None and not carry_df.empty:
        latest_carry = carry_df["carry"].iloc[-1]
        latest_carry_pct = carry_df["carry_pct"].iloc[-1]

        if latest_carry < -10:  # inverted market (cents)
            rec = (
                f"BACKWARDATION: Carry is {latest_carry:.1f}¢ ({latest_carry_pct:.1f}%). "
                "Supply is tight. Consider forward-contracting corn purchases NOW."
            )
            signal = "urgent_buy"
        elif latest_carry < 0:
            rec = (
                f"Mild backwardation: {latest_carry:.1f}¢ ({latest_carry_pct:.1f}%). "
                "Lean toward near-term purchases."
            )
            signal = "lean_buy"
        elif latest_carry < 20:
            rec = (
                f"Slight contango: {latest_carry:.1f}¢ ({latest_carry_pct:.1f}%). "
                "Normal carry. No urgency to pre-contract."
            )
            signal = "neutral"
        else:
            rec = (
                f"CONTANGO: Carry is +{latest_carry:.1f}¢ (+{latest_carry_pct:.1f}%). "
                "Ample supply. Delay purchase or buy deferred contract to capture carry."
            )
            signal = "delay"

        result.update({
            "latest_carry_cents": round(float(latest_carry), 2),
            "latest_carry_pct": round(float(latest_carry_pct), 2),
            "recommendation": rec,
            "signal": signal,
        })

    return result


# ---------------------------------------------------------------------------
# Batch fetch with caching
# ---------------------------------------------------------------------------

def get_all_futures(use_live: bool = False) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Fetch all defined continuous contract histories.
    Returns {commodity_key: DataFrame_or_None}.
    """
    results = {}
    for commodity in CONTINUOUS_TICKERS:
        results[commodity] = fetch_continuous(commodity, period="6mo", use_live=use_live)
    return results


def get_latest_futures_prices() -> Dict[str, float]:
    """
    Return the most recent closing price ($/ton) for each commodity
    that maps to an ingredient key.
    """
    prices = {}
    for commodity, meta in CONTINUOUS_TICKERS.items():
        ing_key = meta.get("ingredient_key")
        if not ing_key:
            continue
        df = fetch_continuous(commodity, period="5d", use_live=True)
        if df is not None and not df.empty and "close_per_ton" in df.columns:
            prices[ing_key] = round(float(df["close_per_ton"].iloc[-1]), 2)
    return prices


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

def _to_per_ton(series: pd.Series, meta: Dict) -> pd.Series:
    """Convert raw yfinance price to $/short ton."""
    units = meta["units"].lower()
    lbs = meta.get("lbs_per_bu")

    if "cents/bu" in units and lbs:
        # cents/bu → $/ton: (cents / 100) × (2000 / lbs_per_bu)
        return series / 100 * 2000 / lbs
    elif "$/short ton" in units or "$/ton" in units:
        return series
    elif "cents/lb" in units:
        # cents/lb → $/ton
        return series / 100 * 2000
    return series   # fallback: assume $/ton


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _save_df_cache(df: pd.DataFrame, path: Path, commodity: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "commodity": commodity,
        "fetched_at": datetime.now().isoformat(),
        "data": df.reset_index().to_dict(orient="records"),
    }
    with open(path, "w") as f:
        json.dump(payload, f, default=str)


def _load_df_cache(path: Path, commodity: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        rows = payload.get("data", [])
        if not rows:
            return None
        df = pd.DataFrame(rows)
        date_col = "Date" if "Date" in df.columns else "index"
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        if "Close" in df.columns:
            meta = CONTINUOUS_TICKERS.get(commodity, {})
            df["close_per_ton"] = _to_per_ton(df["Close"].astype(float), meta)
        return df
    except Exception:
        return None

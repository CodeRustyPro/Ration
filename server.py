"""
Flask API backend for Feed Ration Optimizer.

Wraps existing Python modules (optimizer, economics, stepup, forecaster)
into clean JSON endpoints consumed by the vanilla JS frontend.
"""

import sys
import os
from datetime import datetime, timedelta
from dataclasses import asdict

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(__file__))

from src.nrc_data import INGREDIENTS
from src.optimizer import optimize_ration
from src.usda_fetcher import FALLBACK_PRICES_PER_TON, fetch_all_slugs, get_latest_prices
from src.futures_fetcher import fetch_continuous
from src.forecaster import build_price_series_from_futures, forecast_prices
from src.economics import calculate_costs
from src.stepup import (
    generate_feeding_program,
    compute_exit_analysis,
    compute_sensitivity,
)

app = Flask(__name__, static_folder="static")

# ── Ingredient metadata for the frontend ──────────────────────────────────

INGREDIENT_META = {
    "CornGrain":    {"name": "Corn",         "icon": "corn",     "category": "grain"},
    "DDGS":         {"name": "DDGS",         "icon": "ddgs",     "category": "grain"},
    "CornSilage":   {"name": "Corn Silage",  "icon": "silage",   "category": "roughage"},
    "AlfalfaHay":   {"name": "Alfalfa Hay",  "icon": "alfalfa",  "category": "roughage"},
    "GrassHay":     {"name": "Grass Hay",    "icon": "grass",    "category": "roughage"},
    "SoybeanMeal":  {"name": "Soybean Meal", "icon": "soybean",  "category": "protein"},
    "Urea":         {"name": "Urea",         "icon": "urea",     "category": "supplement"},
    "Limestone":    {"name": "Limestone",    "icon": "ite",       "category": "supplement"},
    "MineralPremix": {"name": "Mineral",     "icon": "mineral",  "category": "supplement"},
}


# ── Cached price data (refreshed on first request) ────────────────────────

_price_cache = {"prices": None, "fetched_at": None}


def _get_usda_prices():
    """Get latest USDA prices with simple time-based caching."""
    now = datetime.now()
    if (_price_cache["prices"] is not None
            and _price_cache["fetched_at"]
            and (now - _price_cache["fetched_at"]).seconds < 3600):
        return _price_cache["prices"]

    try:
        api_key = os.environ.get("USDA_API_KEY", "")
        all_data = fetch_all_slugs(api_key or None, use_live=False)
        prices = get_latest_prices(all_data)
    except Exception:
        prices = {}

    merged = dict(FALLBACK_PRICES_PER_TON)
    merged.update({k: v for k, v in prices.items() if k in INGREDIENTS})
    _price_cache["prices"] = merged
    _price_cache["fetched_at"] = now
    return merged


# ── Static file serving ───────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)


# ── API: Default ingredient data ──────────────────────────────────────────

@app.route("/api/defaults")
def api_defaults():
    usda = _get_usda_prices()
    ingredients = []
    for key, meta in INGREDIENT_META.items():
        price = usda.get(key, FALLBACK_PRICES_PER_TON.get(key, INGREDIENTS[key]["cost_usd_per_ton_dm"]))
        ingredients.append({
            "key": key,
            "name": meta["name"],
            "icon": meta["icon"],
            "category": meta["category"],
            "price": round(price, 0),
            "dm_pct": INGREDIENTS[key]["dm_pct"],
        })
    return jsonify({"ingredients": ingredients})


# ── API: Full optimization ────────────────────────────────────────────────

@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    data = request.json or {}

    # Parse inputs
    ingredients = data.get("ingredients", {})
    cattle = data.get("cattle", {})
    economics = data.get("economics", {})

    start_wt = cattle.get("start_weight", 800)
    target_wt = cattle.get("target_weight", 1350)
    head_count = cattle.get("head_count", 100)
    target_adg = cattle.get("target_adg", 3.0)

    # Build price overrides and excluded list
    excluded = tuple(k for k, v in ingredients.items() if not v.get("enabled", True))
    price_overrides = {}
    for k, v in ingredients.items():
        if v.get("enabled", True) and "price" in v:
            price_overrides[k] = v["price"]

    # Merge with USDA defaults
    usda = _get_usda_prices()
    merged = dict(usda)
    merged.update(price_overrides)

    # Run optimizer
    result = optimize_ration(
        start_wt, target_adg,
        price_overrides=merged,
        dmi_adjustment=1.0,
        use_ionophore=True,
        excluded_ingredients=excluded,
    )

    if not result:
        return jsonify({"error": "No feasible ration found. Try enabling more ingredients or adjusting ADG."}), 400

    # Build profile for economics
    profile = {
        "start_weight": start_wt,
        "target_weight": target_wt,
        "head_count": head_count,
        "purchase_price_cwt": economics.get("purchase_cwt", 370.0),
        "sale_price_cwt": economics.get("sale_cwt", 240.0),
        "yardage_cost": economics.get("yardage", 0.55),
        "interest_rate": economics.get("interest_rate", 8.0),
        "death_loss_pct": economics.get("death_loss", 1.5),
        "freight_dt_cwt": economics.get("freight", 4.0),
        "vet_cost": economics.get("vet_cost", 20.0),
        "transit_shrink_pct": economics.get("transit_shrink", 3.0),
        "pencil_shrink_pct": economics.get("pencil_shrink", 4.0),
        "equity_pct": 0.0,
    }

    gain = target_wt - start_wt
    days_on_feed = gain / target_adg if target_adg > 0 else 180
    feed_cost_day = result["total_cost_per_day"]

    # Economics
    conf = calculate_costs(profile, feed_cost_day, days_on_feed, profile["sale_price_cwt"])

    # Build ration response
    ration_items = []
    for ing_key, v in result["ration"].items():
        dm_pct = INGREDIENTS[ing_key]["dm_pct"]
        as_fed = v["lb_per_day"] * 100.0 / dm_pct
        meta = INGREDIENT_META.get(ing_key, {"name": ing_key, "category": "other", "icon": ""})
        ration_items.append({
            "key": ing_key,
            "name": meta["name"],
            "icon": meta["icon"],
            "category": meta["category"],
            "as_fed_lb": round(as_fed, 1),
            "dm_lb": round(v["lb_per_day"], 2),
            "pct": round(v["pct_of_dmi"], 1),
            "cost_per_day": round(v["cost_per_day"], 3),
        })

    total_as_fed = sum(r["as_fed_lb"] for r in ration_items)

    # Feeding program (4 phases)
    phases_raw = generate_feeding_program(
        start_wt, target_wt, target_adg,
        price_overrides=merged,
        use_ionophore=True,
        excluded_ingredients=excluded,
    )
    phases = []
    if phases_raw:
        for p in phases_raw:
            phase_items = []
            for key, v in p.get("ration", {}).items():
                meta = INGREDIENT_META.get(key, {"name": key, "category": "other"})
                phase_items.append({
                    "name": meta["name"],
                    "category": meta["category"],
                    "pct": v["pct"],
                    "as_fed_lb": v["as_fed_lb"],
                    "cost": v["cost"],
                })
            phases.append({
                "name": p["name"],
                "label": p["label"],
                "desc": p["desc"],
                "day_start": p["day_start"],
                "day_end": p["day_end"],
                "days": p["days"],
                "wt_start": p["wt_start"],
                "wt_end": p["wt_end"],
                "adg": p["adg"],
                "roughage_min": p.get("roughage_min", 0),
                "roughage_max": p.get("roughage_max", 100),
                "cost_per_day": p["cost_per_day"],
                "ration": phase_items,
            })

    # Exit analysis with price forecast
    price_curve = _build_price_curve(days_on_feed)
    exit_data = compute_exit_analysis(profile, feed_cost_day, target_adg, price_curve=price_curve)
    exit_baseline = compute_exit_analysis(profile, feed_cost_day, target_adg, price_curve=None)

    optimal = max(exit_data, key=lambda x: x["profit"]) if exit_data else None
    optimal_baseline = max(exit_baseline, key=lambda x: x["profit"]) if exit_baseline else None

    # Sell date
    sell_date = None
    if optimal:
        sell_date = (datetime.now() + timedelta(days=optimal["day"])).strftime("%b %d")

    # Sensitivity
    sensitivity = compute_sensitivity(profile, feed_cost_day, days_on_feed)

    # Package response
    return jsonify({
        "ration": ration_items,
        "total_as_fed_lb": round(total_as_fed, 1),
        "feed_cost_per_day": round(feed_cost_day, 2),
        "cost_per_lb_gain": round(result.get("cost_per_lb_gain", 0), 3),
        "dmi_lb": round(result["dmi_lb"], 1),
        "economics": {
            "profit_per_head": round(conf.projected_profit_head, 0),
            "feed_cost_head": round(conf.feed_cost_head, 0),
            "total_cost": round(conf.total_all_in_cost, 0),
            "breakeven_cwt": round(conf.breakeven_price_cwt, 1),
            "laid_in_cost": round(conf.laid_in_cost_head, 0),
            "days_on_feed": round(days_on_feed, 0),
            "cost_of_gain": round(conf.cost_of_gain_lb, 3),
            "current_mc": round(exit_data[0]["marginal_cost"], 2) if exit_data else 0,
            "current_mr": round(exit_data[0]["marginal_revenue"], 2) if exit_data else 0,
        },
        "optimal_exit": {
            "day": optimal["day"] if optimal else int(days_on_feed),
            "weight": optimal["weight"] if optimal else target_wt,
            "profit": optimal["profit"] if optimal else 0,
            "adg_at_exit": optimal["daily_gain"] if optimal else target_adg,
            "sell_date": sell_date,
            "sale_price_cwt": optimal.get("sale_price_cwt", profile["sale_price_cwt"]) if optimal else profile["sale_price_cwt"],
        },
        "exit_curve": [
            {
                "day": d["day"], "profit": d["profit"], "weight": d["weight"],
                "adg": d["daily_gain"], "sale_price": d.get("sale_price_cwt", profile["sale_price_cwt"]),
                "marginal_cost": d.get("marginal_cost", 0),
                "marginal_revenue": d.get("marginal_revenue", 0)
            }
            for d in exit_data[::3]  # every 3rd day to keep response small
        ] if exit_data else [],
        "exit_baseline": [
            {"day": d["day"], "profit": d["profit"]}
            for d in exit_baseline[::3]
        ] if exit_baseline else [],
        "has_price_forecast": price_curve is not None,
        "phases": phases,
        "sensitivity": sensitivity[:6] if sensitivity else [],
        "head_count": head_count,
        "target_adg": target_adg,
        "cattle": {"start_weight": start_wt, "target_weight": target_wt},
    })


# ── API: Quick scenario (what-if) ────────────────────────────────────────

@app.route("/api/scenario", methods=["POST"])
def api_scenario():
    """Re-optimize with modified prices or ADG. Returns ration + economics only."""
    data = request.json or {}
    ingredients = data.get("ingredients", {})
    cattle = data.get("cattle", {})
    economics = data.get("economics", {})

    start_wt = cattle.get("start_weight", 800)
    target_wt = cattle.get("target_weight", 1350)
    target_adg = cattle.get("target_adg", 3.0)

    excluded = tuple(k for k, v in ingredients.items() if not v.get("enabled", True))
    usda = _get_usda_prices()
    merged = dict(usda)
    for k, v in ingredients.items():
        if v.get("enabled", True) and "price" in v:
            merged[k] = v["price"]

    result = optimize_ration(
        start_wt, target_adg,
        price_overrides=merged,
        dmi_adjustment=1.0,
        use_ionophore=True,
        excluded_ingredients=excluded,
    )

    if not result:
        return jsonify({"error": "No feasible ration found"}), 400

    profile = {
        "start_weight": start_wt, "target_weight": target_wt,
        "purchase_price_cwt": economics.get("purchase_cwt", 370.0),
        "sale_price_cwt": economics.get("sale_cwt", 240.0),
        "yardage_cost": economics.get("yardage", 0.55),
        "interest_rate": economics.get("interest_rate", 8.0),
        "death_loss_pct": economics.get("death_loss", 1.5),
        "freight_dt_cwt": economics.get("freight", 4.0),
        "vet_cost": economics.get("vet_cost", 20.0),
        "transit_shrink_pct": economics.get("transit_shrink", 3.0),
        "pencil_shrink_pct": economics.get("pencil_shrink", 4.0),
        "equity_pct": 0.0,
    }

    gain = target_wt - start_wt
    days_on_feed = gain / target_adg if target_adg > 0 else 180
    conf = calculate_costs(profile, result["total_cost_per_day"], days_on_feed, profile["sale_price_cwt"])

    ration_items = []
    for ing_key, v in result["ration"].items():
        dm_pct = INGREDIENTS[ing_key]["dm_pct"]
        as_fed = v["lb_per_day"] * 100.0 / dm_pct
        meta = INGREDIENT_META.get(ing_key, {"name": ing_key, "category": "other", "icon": ""})
        ration_items.append({
            "key": ing_key,
            "name": meta["name"],
            "category": meta["category"],
            "as_fed_lb": round(as_fed, 1),
            "pct": round(v["pct_of_dmi"], 1),
            "cost_per_day": round(v["cost_per_day"], 3),
        })

    return jsonify({
        "ration": ration_items,
        "feed_cost_per_day": round(result["total_cost_per_day"], 2),
        "economics": {
            "profit_per_head": round(conf.projected_profit_head, 0),
            "days_on_feed": round(days_on_feed, 0),
            "breakeven_cwt": round(conf.breakeven_price_cwt, 1),
            "cost_of_gain": round(conf.cost_of_gain_lb, 3),
        },
    })


# ── Helper: Build price curve from cattle futures ─────────────────────────

def _build_price_curve(base_days):
    """Build a day→$/cwt price curve from live cattle futures forecast."""
    try:
        series = build_price_series_from_futures("live_cattle", use_live=False)
        if series.empty or len(series) < 10:
            return None
        fc = forecast_prices(series, horizon=40, method="auto", commodity="Live Cattle")
        if not fc or not fc.mean:
            return None

        import pandas as pd
        forecast_dates = pd.to_datetime(fc.dates)
        today = pd.Timestamp.now()
        max_days = int(base_days) + 60

        curve = {}
        for day in range(1, max_days + 1):
            target_date = today + pd.Timedelta(days=day)
            diffs = abs(forecast_dates - target_date)
            nearest_idx = diffs.argmin()
            # $/ton → $/cwt (live cattle: cents/lb → $/ton in fetcher → /20 = $/cwt)
            curve[day] = fc.mean[nearest_idx] / 20.0
        return curve
    except Exception:
        return None


# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8000)

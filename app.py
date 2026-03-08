"""
Feed Ration Optimizer — Streamlit Dashboard
UIUC Precision Digital Agriculture Hackathon 2026

"The nutritionist in your pocket."

Farmer-first design:
  1. Onboarding: What feeds do you have? → Your cattle
  2. Dashboard: AI recommendation → 3 KPIs → As-fed ration + mixer
  3. Feeding Program: 4-phase step-up protocol
  4. When to Sell: Price-aware optimal exit timing + ADG sweep
"""

import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Feed Ration Optimizer",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── LIGHT MODE CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
header {visibility: hidden;}
footer {visibility: hidden;}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E0E4E0;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
div[data-testid="stMetricLabel"] {
    font-size: 0.85rem; font-weight: 500; color: #6B7280;
    letter-spacing: 0.3px;
}
div[data-testid="stMetricValue"] {
    font-size: 1.9rem; font-weight: 700; color: #1A1A1A;
    font-family: 'Inter', system-ui, sans-serif;
}
div[data-testid="stTabs"] button {
    font-size: 1.05rem; font-weight: 600;
    padding-bottom: 10px; padding-top: 10px;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom-color: #2E7D32;
}
</style>
""", unsafe_allow_html=True)

import sys
sys.path.insert(0, ".")

from src.nrc_data import INGREDIENTS, compute_requirements, estimate_dmi
from src.optimizer import optimize_ration
from src.usda_fetcher import FALLBACK_PRICES_PER_TON, fetch_all_slugs, get_latest_prices
from src.futures_fetcher import CONTINUOUS_TICKERS, fetch_continuous, get_all_futures
from src.forecaster import build_price_series_from_futures, forecast_prices
from src.confinement import cumulative_cost_series
from src.economics import calculate_costs, CostResult
from src.units import format_price, as_fed_price_to_dm
from src.stepup import generate_feeding_program, compute_exit_analysis, compute_sensitivity

try:
    from src.weather_agent import get_weather_stress, get_location_name
    _HAS_WEATHER = True
except ImportError:
    _HAS_WEATHER = False

try:
    from src.market_agent import interpret_shadow_prices, generate_ration_verdict
    _HAS_MARKET = True
except ImportError:
    _HAS_MARKET = False

try:
    from src.ai_agent import explain_ration
    _HAS_AI = True
except ImportError:
    _HAS_AI = False

# Colors optimized for light backgrounds
GREEN  = "#2E7D32"
ORANGE = "#E65100"
RED    = "#C62828"
BLUE   = "#1565C0"
GRAY   = "#9E9E9E"

INGREDIENT_CARDS = {
    "CornGrain":    {"icon": "🌽", "short": "Corn"},
    "DDGS":         {"icon": "🌾", "short": "DDGS"},
    "CornSilage":   {"icon": "🌿", "short": "Corn Silage"},
    "AlfalfaHay":   {"icon": "🍀", "short": "Alfalfa Hay"},
    "GrassHay":     {"icon": "🌱", "short": "Grass Hay"},
    "SoybeanMeal":  {"icon": "🫘", "short": "Soybean Meal"},
    "Urea":         {"icon": "💧", "short": "Urea"},
    "Limestone":    {"icon": "🪨", "short": "Limestone"},
    "MineralPremix": {"icon": "💊", "short": "Mineral"},
}


# ═══════════════════════════════════════════════════════════════════════════
# CACHED DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_usda_prices(use_live: bool, api_key: str):
    all_data = fetch_all_slugs(api_key or None, use_live=use_live)
    return all_data, get_latest_prices(all_data)

@st.cache_data(ttl=600, show_spinner=False)
def run_optimizer(bw_lb, adg_lb, _price_overrides_hash, price_overrides_dict,
                  dmi_adj, use_ionophore, excluded_ingredients):
    return optimize_ration(
        bw_lb, adg_lb, price_overrides=price_overrides_dict or None,
        dmi_adjustment=dmi_adj, use_ionophore=use_ionophore,
        excluded_ingredients=excluded_ingredients,
    )

@st.cache_data(ttl=600, show_spinner=False)
def run_feeding_program(start_wt, target_wt, target_adg, _price_hash,
                        price_overrides_dict, use_ionophore, excluded_ingredients):
    return generate_feeding_program(
        start_wt, target_wt, target_adg,
        price_overrides=price_overrides_dict or None,
        use_ionophore=use_ionophore,
        excluded_ingredients=excluded_ingredients,
    )

@st.cache_data(ttl=600, show_spinner=False)
def run_adg_sweep(profile_hash, profile, feed_prices_dict, ionophore, excluded):
    scenarios = []
    for test_adg in np.arange(1.5, 4.25, 0.25):
        res = optimize_ration(
            profile["start_weight"], test_adg,
            price_overrides=feed_prices_dict or None,
            dmi_adjustment=1.0, use_ionophore=ionophore,
            excluded_ingredients=excluded,
        )
        if res:
            gain = profile["target_weight"] - profile["start_weight"]
            days = gain / test_adg if test_adg > 0 else 0
            cost_res = calculate_costs(
                profile, feed_cost_per_day=res["total_cost_per_day"],
                days_on_feed=days, current_market_price_cwt=profile["sale_price_cwt"],
            )
            scenarios.append({
                "adg_lb": test_adg, "days": days,
                "feed_per_day": res["total_cost_per_day"],
                "profit": cost_res.projected_profit_head,
                "breakeven": cost_res.breakeven_price_cwt,
                "total_cost": cost_res.total_all_in_cost,
            })
    return scenarios

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_cattle_price_forecast(use_live: bool):
    """Fetch live cattle futures and forecast sale prices forward."""
    try:
        price_series = build_price_series_from_futures("live_cattle", use_live=use_live)
        if price_series.empty or len(price_series) < 10:
            return None
        # Forecast 40 weeks ahead (~280 days covers most feeding periods)
        fc = forecast_prices(price_series, horizon=40, method="auto", commodity="Live Cattle")
        return fc
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# ONBOARDING
# ═══════════════════════════════════════════════════════════════════════════

if "profile" not in st.session_state:
    st.session_state.profile = None

if st.session_state.profile is None:
    st.markdown("# The Nutritionist in Your Pocket")
    st.markdown("Tell us what you have. We'll build the cheapest ration that meets every nutritional requirement.")
    st.markdown("---")

    with st.form("onboarding"):
        st.markdown("### Step 1: What feeds do you have?")
        st.caption("Toggle ON the feeds available on your operation. Enter YOUR cost (what you paid), not market price.")

        selected = {}
        prices_input = {}

        keys = list(INGREDIENT_CARDS.keys())
        for row_start in range(0, len(keys), 3):
            cols = st.columns(3)
            for i, col in enumerate(cols):
                idx = row_start + i
                if idx >= len(keys):
                    break
                key = keys[idx]
                card = INGREDIENT_CARDS[key]
                fallback = FALLBACK_PRICES_PER_TON.get(key, INGREDIENTS[key]["cost_usd_per_ton_dm"])

                with col:
                    with st.container(border=True):
                        on = st.checkbox(
                            f"{card['icon']} {card['short']}",
                            value=True,
                            key=f"ing_{key}",
                        )
                        selected[key] = on
                        if on:
                            p = st.number_input(
                                "Your cost ($/ton)", value=float(fallback),
                                step=10.0, key=f"price_{key}",
                                label_visibility="collapsed",
                            )
                            prices_input[key] = p

        st.markdown("### Step 2: Your cattle")
        c1, c2, c3 = st.columns(3)
        head_count = c1.number_input("Head count", 1, 5000, 100, step=10)
        start_wt = c2.number_input("Current weight (lb)", 400, 1500, 800, step=25)
        target_wt = c3.number_input("Target weight (lb)", 800, 1800, 1350, step=25)

        with st.expander("Economics (optional — good defaults provided)"):
            st.caption("Defaults reflect March 2026 Midwest market conditions (CME, ISU Extension).")
            ec1, ec2 = st.columns(2)
            purchase = ec1.number_input("Purchase price ($/cwt)", 200.0, 500.0, 370.0, step=5.0,
                                        help="CME Feeder Cattle Index: $369/cwt (Mar 2026)")
            sale = ec2.number_input("Expected sale price ($/cwt)", 150.0, 400.0, 240.0, step=5.0,
                                    help="Cash fed steers: ~$240/cwt live (Mar 2026)")
            ec3, ec4, ec5 = st.columns(3)
            yardage = ec3.number_input("Yardage ($/hd/d)", 0.0, 3.0, 0.55, step=0.05,
                                       help="ISU/SDSU: $0.45-0.75 for farmer-feeders")
            interest = ec4.number_input("Interest rate (%)", 0.0, 15.0, 8.0,
                                        help="Commercial: 7-9%. FSA direct: 4.75%")
            death_loss = ec5.number_input("Death loss (%)", 0.0, 10.0, 1.5,
                                          help="Yearlings 1-2.5%, auction calves 3-5%")
            ec6, ec7, ec8 = st.columns(3)
            freight = ec6.number_input("Freight ($/cwt)", 0.0, 20.0, 4.0)
            vet = ec7.number_input("Vet/Processing ($/hd)", 0.0, 100.0, 20.0,
                                   help="K-State: $13-25/head")
            shrink = ec8.number_input("Transit shrink (%)", 0.0, 10.0, 3.0,
                                      help="2-4% for short Midwest hauls, 5-7% for 8+ hrs")

        if st.form_submit_button("Optimize My Ration", type="primary", use_container_width=True):
            excluded = [k for k, on in selected.items() if not on]
            custom_prices = {}
            for k, p in prices_input.items():
                fb = FALLBACK_PRICES_PER_TON.get(k, INGREDIENTS[k]["cost_usd_per_ton_dm"])
                if p != fb:
                    custom_prices[k] = p

            st.session_state.profile = {
                "start_weight": start_wt,
                "target_weight": target_wt,
                "head_count": head_count,
                "target_adg": 3.0,
                "purchase_price_cwt": purchase,
                "sale_price_cwt": sale,
                "yardage_cost": yardage,
                "interest_rate": interest,
                "death_loss_pct": death_loss,
                "freight_dt_cwt": freight,
                "vet_cost": vet,
                "transit_shrink_pct": shrink,
                "pencil_shrink_pct": 4.0,
                "equity_pct": 0.0,
                "excluded_ingredients": excluded,
                "custom_prices": custom_prices,
                "ingredient_prices": prices_input,
            }
            st.rerun()
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# POST-ONBOARDING: LOAD PROFILE & SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

prof = st.session_state.profile
if prof is None:
    prof = {
        "start_weight": 800, "target_weight": 1350, "head_count": 100,
        "target_adg": 3.0, "purchase_price_cwt": 370.0, "sale_price_cwt": 240.0,
        "yardage_cost": 0.55, "interest_rate": 8.0, "death_loss_pct": 1.5,
        "freight_dt_cwt": 4.0, "vet_cost": 20.0, "transit_shrink_pct": 3.0,
        "pencil_shrink_pct": 4.0, "equity_pct": 0.0,
        "excluded_ingredients": [], "custom_prices": {}, "ingredient_prices": {},
    }

with st.sidebar:
    st.markdown("## My Operation")
    if st.button("Start Over"):
        st.session_state.profile = None
        st.rerun()

    st.caption(f"**{prof['head_count']} head** | {prof['start_weight']} → {prof['target_weight']} lb")
    st.divider()

    target_adg = st.slider("Target ADG (lb/d)", 1.5, 4.5, prof.get("target_adg", 3.0), 0.1)
    use_ionophore = st.toggle("Ionophore (monensin)", value=True)

    custom_prices = dict(prof.get("custom_prices", {}))
    ingredient_prices = dict(prof.get("ingredient_prices", {}))
    excluded = list(prof.get("excluded_ingredients", []))

    st.divider()
    with st.expander("Edit Ingredient Prices"):
        for key in INGREDIENT_CARDS:
            if key in excluded:
                continue
            card = INGREDIENT_CARDS[key]
            fallback = FALLBACK_PRICES_PER_TON.get(key, INGREDIENTS[key]["cost_usd_per_ton_dm"])
            current = ingredient_prices.get(key, fallback)
            new_p = st.number_input(
                f"{card['icon']} {card['short']} ($/ton)",
                value=float(current), step=10.0, key=f"sb_price_{key}",
            )
            if new_p != fallback:
                custom_prices[key] = new_p
            ingredient_prices[key] = new_p

    with st.expander("Edit Economics"):
        prof["purchase_price_cwt"] = st.number_input("Purchase ($/cwt)", 150.0, 500.0, prof["purchase_price_cwt"])
        prof["sale_price_cwt"] = st.number_input("Sale ($/cwt)", 150.0, 400.0, prof["sale_price_cwt"])
        prof["yardage_cost"] = st.number_input("Yardage ($/hd/d)", 0.0, 3.0, prof["yardage_cost"])
        prof["head_count"] = st.number_input("Head count", 1, 5000, prof["head_count"], step=10)

    st.divider()
    dmi_adj = 1.0
    use_live = False
    with st.expander("Advanced"):
        dmi_adj = st.slider("DMI adjustment", 0.85, 1.15, 1.0, 0.01)
        use_live = st.toggle("Live API data", value=False)

    if st.button("Re-Optimize", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# LOAD DATA & OPTIMIZE
# ═══════════════════════════════════════════════════════════════════════════

try:
    api_key = st.secrets.get("api_keys", {}).get("usda", "")
except Exception:
    api_key = ""

with st.spinner("Loading market data..."):
    usda_all, usda_prices = load_usda_prices(use_live, api_key)

merged_prices = dict(FALLBACK_PRICES_PER_TON)
merged_prices.update({k: v for k, v in usda_prices.items() if k in INGREDIENTS})
merged_prices.update(ingredient_prices)
merged_prices.update(custom_prices)

excluded_tuple = tuple(sorted(excluded))

prices_hash = tuple(sorted(merged_prices.items()))
prof_hash = tuple(sorted((k, v) for k, v in prof.items() if not isinstance(v, (list, dict))))

result = run_optimizer(
    prof["start_weight"], target_adg, prices_hash, merged_prices,
    dmi_adj, use_ionophore, excluded_tuple,
)

# Header
data_source = "LIVE" if use_live else "CACHED"
st.markdown(
    f"## Feed Ration Optimizer "
    f"<span style='font-size:0.7em; color:#9E9E9E;'>{data_source}</span>",
    unsafe_allow_html=True,
)

if result is None:
    st.error("No feasible ration found. Try adjusting ADG, enabling more ingredients, or changing weight targets.")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# ECONOMICS
# ═══════════════════════════════════════════════════════════════════════════

gain_needed = prof["target_weight"] - prof["start_weight"]
days_on_feed = gain_needed / target_adg if target_adg > 0 else 0

conf = calculate_costs(
    prof,
    feed_cost_per_day=result["total_cost_per_day"],
    days_on_feed=days_on_feed,
    current_market_price_cwt=prof["sale_price_cwt"],
)


# ═══════════════════════════════════════════════════════════════════════════
# AI RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════

if _HAS_AI and _HAS_MARKET:
    shadow_prices = interpret_shadow_prices(result.get("shadow_prices", {}), result["dmi_lb"])
    explanation = explain_ration(
        ration=result["ration"],
        shadow_prices=shadow_prices,
        target_adg=target_adg,
        body_weight=prof["start_weight"],
        cost_per_day=result["total_cost_per_day"],
    )
    st.info(f"**What you should do:** {explanation}")
elif _HAS_MARKET:
    verdict = generate_ration_verdict(
        result["nutrient_totals"], result["requirements"],
        result.get("cost_per_lb_gain", 0.5), target_adg, result["dmi_lb"],
    )
    st.info(f"**Assessment:** {verdict['verdict']}")


# ═══════════════════════════════════════════════════════════════════════════
# HERO KPIs
# ═══════════════════════════════════════════════════════════════════════════

c1, c2, c3 = st.columns(3)

profit_val = conf.projected_profit_head
profit_color = "normal" if profit_val >= 0 else "inverse"
c1.metric(
    "Projected Profit",
    f"${profit_val:,.0f}/hd",
    delta=f"${profit_val:,.0f}" if profit_val != 0 else None,
    delta_color=profit_color,
    border=True,
)
c2.metric(
    "Feed Cost",
    f"${result['total_cost_per_day']:.2f}/hd/day",
    help=f"Cost of gain: ${result.get('cost_per_lb_gain', 0):.3f}/lb",
    border=True,
)
c3.metric(
    "Days to Market",
    f"{days_on_feed:.0f} days",
    help=f"{prof['start_weight']} → {prof['target_weight']} lb at {target_adg:.1f} lb/d ADG",
    border=True,
)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════

tab_ration, tab_program, tab_timing = st.tabs([
    "Today's Ration",
    "Feeding Program",
    "When to Sell",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1: TODAY'S RATION
# ═══════════════════════════════════════════════════════════════════════════

with tab_ration:
    head_count = prof.get("head_count", 100)

    ration_rows = []
    for ing_key, v in result["ration"].items():
        dm_pct = INGREDIENTS[ing_key]["dm_pct"]
        dm_lb = v["lb_per_day"]
        as_fed_lb = dm_lb * 100.0 / dm_pct
        batch_lb = as_fed_lb * head_count

        ration_rows.append({
            "Ingredient": v["display_name"].split(",")[0],
            "As-Fed (lb/hd/day)": round(as_fed_lb, 1),
            "% of Diet": f"{v['pct_of_dmi']:.1f}%",
            "Cost ($/hd/day)": f"${v['cost_per_day']:.3f}",
            "Mixer Batch (lb)": f"{batch_lb:,.0f}",
            "_dm_lb": dm_lb,
            "_dm_pct": dm_pct,
        })

    col_table, col_mixer = st.columns([1.5, 1])

    with col_table:
        st.subheader("Your Ration (As-Fed)")
        st.caption("Amounts shown as-fed — what you weigh into the mixer.")

        display_df = pd.DataFrame(ration_rows)[[
            "Ingredient", "As-Fed (lb/hd/day)", "% of Diet", "Cost ($/hd/day)",
        ]]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        with st.expander("Dry Matter Detail"):
            dm_rows = [{
                "Ingredient": r["Ingredient"],
                "DM (lb/hd/day)": f"{r['_dm_lb']:.2f}",
                "DM %": f"{r['_dm_pct']:.0f}%",
                "As-Fed (lb)": f"{r['As-Fed (lb/hd/day)']:.1f}",
            } for r in ration_rows]
            st.dataframe(pd.DataFrame(dm_rows), use_container_width=True, hide_index=True)

    with col_mixer:
        st.subheader(f"Mixer Batch ({head_count} head)")
        st.caption("Load these amounts into your mixer for one feeding.")

        total_batch_lb = sum(
            float(r["As-Fed (lb/hd/day)"]) * head_count for r in ration_rows
        )
        st.metric("Total Batch Weight", f"{total_batch_lb:,.0f} lb", border=True)

        for r in ration_rows:
            batch = float(r["As-Fed (lb/hd/day)"]) * head_count
            st.markdown(f"**{r['Ingredient']}**: {batch:,.0f} lb")

        st.divider()
        st.caption(f"Total feed cost: **${result['total_cost_per_day'] * head_count:,.0f}/day** for {head_count} head")

    with st.expander("Cost Breakdown"):
        st.caption(f"{prof['start_weight']} lb → {prof['target_weight']} lb ({days_on_feed:.0f} days)")
        cost_data = {
            "Category": ["Feed", "Yardage", "Interest", "Vet/Processing", "Death Loss", "Total All-In"],
            "Total ($/hd)": [
                f"${conf.feed_cost_head:,.0f}",
                f"${conf.yardage_cost_head:,.0f}",
                f"${conf.total_interest:,.0f}",
                f"${conf.vet_cost:,.0f}",
                f"${conf.death_loss_cost:,.0f}",
                f"${conf.total_all_in_cost:,.0f}",
            ],
            "Per Day ($/hd)": [
                f"${result['total_cost_per_day']:.2f}",
                f"${prof['yardage_cost']:.2f}",
                f"${conf.total_interest / max(days_on_feed, 1):.2f}",
                f"${conf.vet_cost / max(days_on_feed, 1):.2f}",
                f"${conf.death_loss_cost / max(days_on_feed, 1):.2f}",
                f"${conf.total_all_in_cost / max(days_on_feed, 1):.2f}",
            ],
        }
        st.dataframe(pd.DataFrame(cost_data), use_container_width=True, hide_index=True)

        cp1, cp2, cp3 = st.columns(3)
        cp1.metric("Laid-In Cost", f"${conf.laid_in_cost_head:,.0f}/hd", border=True)
        shrunk_rev = conf.shrunk_sale_weight_lb * prof["sale_price_cwt"] / 100.0
        cp2.metric("Revenue", f"${shrunk_rev:,.0f}/hd", border=True)
        cp3.metric("Breakeven", f"${conf.breakeven_price_cwt:.1f}/cwt", border=True)

    with st.expander("Nutrient Compliance"):
        nt = result["nutrient_totals"]
        req = result["requirements"]
        dmi_lb = req["dmi_lb"]
        NUTRIENT_DISPLAY = {
            "cp": ("Crude Protein", "%", "cp_min", "cp_max"),
            "tdn": ("TDN", "%", "tdn_min", "tdn_max"),
            "ca": ("Calcium", "%", "ca_min", "ca_max"),
            "p": ("Phosphorus", "%", "p_min", "p_max"),
            "ndf": ("NDF", "%", "ndf_min", "ndf_max"),
            "sulfur": ("Sulfur", "%", "sulfur_min", "sulfur_max"),
        }
        comp_rows = []
        for nut_key, (label, unit, lo_key, hi_key) in NUTRIENT_DISPLAY.items():
            val = nt.get(nut_key)
            if val is None:
                continue
            lo = req.get(lo_key)
            hi = req.get(hi_key)
            if lo is not None and hi is not None:
                status = "OK" if lo <= val <= hi else ("LOW" if val < lo else "HIGH")
            elif lo is not None:
                status = "OK" if val >= lo else "LOW"
            elif hi is not None:
                status = "OK" if val <= hi else "HIGH"
            else:
                status = "—"
            comp_rows.append({
                "Nutrient": f"{label} ({unit})",
                "Actual": f"{val:.2f}",
                "Required": f"{lo:.2f}–{hi:.2f}" if lo and hi else (f">={lo:.2f}" if lo else f"<={hi:.2f}"),
                "Status": status,
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
        st.caption(f"DMI: **{result['dmi_lb']:.1f} lb/d** | MCP: **{result['mcp_g_per_day']:.0f} g/d** | "
                   f"Solver: {result.get('solver', 'CBC')}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2: FEEDING PROGRAM — 4-Phase Step-Up
# ═══════════════════════════════════════════════════════════════════════════

with tab_program:
    st.markdown("### Feeding Program: Receiving to Finishing")
    st.caption(
        "Cattle must be stepped up gradually from high-roughage receiving diets "
        "to high-concentrate finishing diets over 4-6 weeks to prevent acidosis. "
        "Each phase below is an optimized ration for that stage."
    )

    with st.spinner("Generating 4-phase program..."):
        phases = run_feeding_program(
            prof["start_weight"], prof["target_weight"], target_adg,
            prices_hash, merged_prices, use_ionophore, excluded_tuple,
        )

    if phases:
        total_feed_cost = sum(
            p["cost_per_day"] * p["days"] for p in phases if p["cost_per_day"]
        )
        st.metric(
            "Total Feed Cost (all phases)",
            f"${total_feed_cost:,.0f}/hd",
            help="Sum of daily feed cost across all 4 phases",
            border=True,
        )

        for phase in phases:
            with st.container(border=True):
                st.markdown(f"#### {phase['label']}")
                st.caption(
                    f"{phase['desc']} | "
                    f"Day {phase['day_start']}–{phase['day_end']} | "
                    f"{phase['wt_start']} → {phase['wt_end']} lb | "
                    f"ADG {phase['adg']} lb/d | "
                    f"Roughage {phase['roughage_min']:.0f}–{phase['roughage_max']:.0f}%"
                    if 'roughage_min' in phase else phase['desc']
                )

                if phase["ration"]:
                    rows = []
                    for key, v in phase["ration"].items():
                        rows.append({
                            "Ingredient": v["name"].split(",")[0],
                            "As-Fed (lb/hd/day)": f"{v['as_fed_lb']:.1f}",
                            "% of Diet": f"{v['pct']:.1f}%",
                            "Cost": f"${v['cost']:.3f}/d",
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                    if phase["cost_per_day"]:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Feed Cost", f"${phase['cost_per_day']:.2f}/hd/day")
                        c2.metric("Phase Duration", f"{phase['days']} days")
                        phase_total = phase["cost_per_day"] * phase["days"]
                        c3.metric("Phase Total", f"${phase_total:,.0f}/hd")
                else:
                    st.warning("No feasible ration for this phase. Consider enabling more roughage sources.")

    else:
        st.warning("Could not generate feeding program.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3: WHEN TO SELL — Price-Aware Exit + Sensitivity + ADG Sweep
# ═══════════════════════════════════════════════════════════════════════════

with tab_timing:
    # ── SENSITIVITY ANALYSIS ──
    st.markdown("### What Moves Your Profit Most")
    st.caption(
        "Cattle prices dwarf every other variable. A $10/cwt swing in sale price "
        "moves ~$130/head — more than all operating costs combined."
    )

    sensitivity = compute_sensitivity(prof, result["total_cost_per_day"], days_on_feed)

    if sensitivity:
        s_names = [s["parameter"] for s in sensitivity]
        s_impacts = [s["impact_per_head"] for s in sensitivity]
        s_colors = [GREEN if imp > 0 else RED for imp in s_impacts]

        fig_tornado = go.Figure(go.Bar(
            y=s_names[::-1],
            x=s_impacts[::-1],
            orientation="h",
            marker_color=s_colors[::-1],
            text=[f"${abs(v):,.0f}" for v in s_impacts[::-1]],
            textposition="outside",
            hovertemplate="%{y}: $%{x:,.0f}/head<extra></extra>",
        ))
        fig_tornado.update_layout(
            template="plotly_white", height=320,
            xaxis_title="Profit Impact ($/head)",
            yaxis_title="",
            margin=dict(t=10, b=40, l=120, r=60),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_tornado, use_container_width=True)

        with st.expander("Sensitivity Detail"):
            sens_table = [{
                "Parameter": s["parameter"],
                "Your Value": s["base_value"],
                "Scenario": s["scenario"],
                "$/head": f"${s['impact_per_head']:+,.0f}",
                f"$/{prof.get('head_count', 200)} head": f"${s['impact_herd']:+,.0f}",
            } for s in sensitivity]
            st.dataframe(pd.DataFrame(sens_table), use_container_width=True, hide_index=True)

    # ── OPTIMAL EXIT WITH PRICE PREDICTION ──
    st.markdown("---")
    st.markdown("### When to Stop Feeding")
    st.caption(
        "Feed efficiency declines as cattle get heavier. At some point, the cost of "
        "one more day exceeds the value of the weight gained."
    )

    # Build price curve from live cattle futures forecast
    cattle_forecast = fetch_cattle_price_forecast(use_live)
    price_curve = None
    has_price_forecast = False

    if cattle_forecast and cattle_forecast.mean:
        has_price_forecast = True
        # Convert $/ton forecasted prices to $/cwt
        # Live cattle futures are in cents/lb → fetcher converts to $/ton
        # $/ton → $/cwt: $/ton * 100 / 2000 = $/ton / 20
        forecast_dates = pd.to_datetime(cattle_forecast.dates)
        today = pd.Timestamp.now()

        price_curve = {}
        total_gain = prof["target_weight"] - prof["start_weight"]
        base_days = int(total_gain / max(target_adg, 0.5))
        max_days = base_days + 60

        for day in range(1, max_days + 1):
            target_date = today + pd.Timedelta(days=day)
            # Find nearest forecast date
            if len(forecast_dates) > 0:
                diffs = abs(forecast_dates - target_date)
                nearest_idx = diffs.argmin()
                # $/ton to $/cwt
                forecasted_cwt = cattle_forecast.mean[nearest_idx] / 20.0
                price_curve[day] = forecasted_cwt

    # Run exit analysis
    exit_data = compute_exit_analysis(
        prof, result["total_cost_per_day"], target_adg,
        price_curve=price_curve,
    )

    # Also run constant-price baseline for comparison
    exit_data_baseline = compute_exit_analysis(
        prof, result["total_cost_per_day"], target_adg,
        price_curve=None,
    )

    if exit_data:
        best_exit = max(exit_data, key=lambda x: x["profit"])
        best_baseline = max(exit_data_baseline, key=lambda x: x["profit"])
        target_day = int(days_on_feed)

        em1, em2, em3, em4 = st.columns(4)
        em1.metric("Optimal Exit Day", f"Day {best_exit['day']}", border=True)
        em2.metric("Optimal Exit Weight", f"{best_exit['weight']:,} lb", border=True)
        em3.metric("Peak Profit", f"${best_exit['profit']:,.0f}/hd", border=True)
        em4.metric("ADG at Exit", f"{best_exit['daily_gain']:.1f} lb/d",
                   help="ADG declines as cattle get heavier (NASEM 2016)", border=True)

        # Show price prediction insight
        if has_price_forecast and best_exit["day"] != best_baseline["day"]:
            day_diff = best_exit["day"] - best_baseline["day"]
            if day_diff < 0:
                st.success(
                    f"Price forecast suggests selling **{abs(day_diff)} days earlier** "
                    f"at {best_exit['weight']:,} lb — the market is predicted stronger "
                    f"(${best_exit.get('sale_price_cwt', prof['sale_price_cwt']):.0f}/cwt) "
                    f"making it profitable to sell lighter."
                )
            else:
                st.info(
                    f"Price forecast suggests holding **{day_diff} more days** "
                    f"to {best_exit['weight']:,} lb — prices are predicted to improve."
                )

        # Profit vs Day chart
        days_list = [d["day"] for d in exit_data]
        profits = [d["profit"] for d in exit_data]
        weights = [d["weight"] for d in exit_data]
        adgs = [d["daily_gain"] for d in exit_data]

        fig_exit = go.Figure()

        # Show baseline (constant price) as dashed line for comparison
        if has_price_forecast:
            fig_exit.add_trace(go.Scatter(
                x=[d["day"] for d in exit_data_baseline],
                y=[d["profit"] for d in exit_data_baseline],
                mode="lines",
                line=dict(color=GRAY, width=2, dash="dash"),
                name="Constant Price",
                hovertemplate="Day %{x}<br>Profit (constant): $%{y:,.0f}/hd<extra></extra>",
            ))

        # Main profit line (with or without price forecast)
        line_name = "Profit (price forecast)" if has_price_forecast else "Profit ($/hd)"
        fig_exit.add_trace(go.Scatter(
            x=days_list, y=profits, mode="lines",
            line=dict(color=GREEN, width=3), name=line_name,
            hovertemplate=(
                "Day %{x}<br>Profit: $%{y:,.0f}/hd<br>"
                "Weight: %{customdata[0]:,} lb<br>"
                "ADG: %{customdata[1]:.1f} lb/d<extra></extra>"
            ),
            customdata=list(zip(weights, adgs)),
        ))

        # Mark optimal
        fig_exit.add_trace(go.Scatter(
            x=[best_exit["day"]], y=[best_exit["profit"]],
            mode="markers",
            marker=dict(size=16, color="#FFB300", symbol="star",
                        line=dict(color="#1A1A1A", width=1.5)),
            name=f"Optimal: Day {best_exit['day']} ({best_exit['weight']:,} lb)",
        ))

        # Mark current target
        if abs(target_day - best_exit["day"]) > 3:
            target_profit = next(
                (d["profit"] for d in exit_data if d["day"] == target_day), None
            )
            if target_profit is not None:
                fig_exit.add_trace(go.Scatter(
                    x=[target_day], y=[target_profit],
                    mode="markers",
                    marker=dict(size=14, color=BLUE, symbol="diamond",
                                line=dict(color="#1A1A1A", width=1.5)),
                    name=f"Target: Day {target_day}",
                ))

        fig_exit.add_hline(y=0, line_dash="dash", line_color=GRAY, opacity=0.4)
        fig_exit.update_layout(
            template="plotly_white", height=380,
            xaxis_title="Day on Feed", yaxis_title="Profit ($/head)",
            hovermode="closest", legend=dict(orientation="h", y=1.02),
            margin=dict(t=10, b=50, l=60, r=20),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_exit, use_container_width=True)

        # Marginal analysis insight
        exit_at_target = next((d for d in exit_data if d["day"] == target_day), exit_data[-1])
        mc = exit_at_target["marginal_cost"]
        mr = exit_at_target["marginal_revenue"]
        if mr > mc:
            diff = mr - mc
            st.success(
                f"At Day {target_day}: marginal revenue **${mr:.2f}/d** > "
                f"marginal cost **${mc:.2f}/d** — keep feeding. "
                f"Net gain: **${diff:.2f}/hd/day**. "
                f"Optimal exit: Day {best_exit['day']} ({best_exit['weight']:,} lb)."
            )
        else:
            st.warning(
                f"At Day {target_day}: marginal cost **${mc:.2f}/d** exceeds "
                f"marginal revenue **${mr:.2f}/d**. Consider selling at "
                f"Day {best_exit['day']} ({best_exit['weight']:,} lb) instead."
            )

        # Show sale price forecast if available
        if has_price_forecast:
            with st.expander("Forecasted Sale Price Over Time"):
                sale_prices = [d.get("sale_price_cwt", prof["sale_price_cwt"]) for d in exit_data]
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(
                    x=days_list, y=sale_prices,
                    mode="lines", line=dict(color=BLUE, width=2),
                    name="Forecasted Sale Price",
                    hovertemplate="Day %{x}<br>$%{y:.1f}/cwt<extra></extra>",
                ))
                fig_price.add_hline(
                    y=prof["sale_price_cwt"], line_dash="dash",
                    line_color=GRAY, opacity=0.5,
                    annotation_text=f"Your estimate: ${prof['sale_price_cwt']:.0f}/cwt",
                )
                fig_price.update_layout(
                    template="plotly_white", height=250,
                    xaxis_title="Day on Feed",
                    yaxis_title="Sale Price ($/cwt)",
                    margin=dict(t=10, b=40, l=60, r=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )
                st.plotly_chart(fig_price, use_container_width=True)
                st.caption(
                    f"Source: Live Cattle futures (LE=F) forecast via {cattle_forecast.method}. "
                    f"Last actual: ${cattle_forecast.last_actual_price/20:.0f}/cwt "
                    f"({cattle_forecast.last_actual_date})."
                )

        with st.expander("Exit Analysis Detail"):
            detail = [{
                "Day": d["day"],
                "Weight (lb)": f"{d['weight']:,}",
                "ADG (lb/d)": f"{d['daily_gain']:.1f}",
                "Sale Price": f"${d.get('sale_price_cwt', prof['sale_price_cwt']):.0f}/cwt",
                "Cost of Gain": f"${d['cost_of_gain']:.3f}/lb",
                "Profit ($/hd)": f"${d['profit']:,.0f}",
                "Marg Cost": f"${d['marginal_cost']:.2f}",
                "Marg Rev": f"${d['marginal_revenue']:.2f}",
            } for d in exit_data if d["day"] % 10 == 0 or d["day"] == best_exit["day"]]
            st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)

    # ── ADG SWEEP ──
    st.markdown("---")
    st.markdown("### Push Harder vs. Slow Down")
    st.caption(
        "Lower ADG = cheaper daily feed but more days of yardage and interest. "
        "Higher ADG = pricier diet but fewer days. Which maximizes profit?"
    )

    with st.spinner("Sweeping ADG scenarios..."):
        scenarios = run_adg_sweep(prof_hash, prof, merged_prices, use_ionophore, excluded_tuple)

    if scenarios:
        adgs = [s["adg_lb"] for s in scenarios]
        profits = [s["profit"] for s in scenarios]
        days_list = [s["days"] for s in scenarios]
        breakevens = [s["breakeven"] for s in scenarios]
        best = max(scenarios, key=lambda s: s["profit"])

        bs1, bs2, bs3 = st.columns(3)
        bs1.metric("Best ADG", f"{best['adg_lb']:.1f} lb/d", border=True)
        bs2.metric("Best Profit", f"${best['profit']:,.0f}/hd", border=True)
        bs3.metric("Days at Best ADG", f"{best['days']:.0f}", border=True)

        fig_adg = go.Figure()
        fig_adg.add_trace(go.Scatter(
            x=adgs, y=profits, mode="lines+markers",
            line=dict(color=GREEN, width=3), marker=dict(size=8),
            text=[
                f"ADG: {a:.1f}<br>Profit: ${p:,.0f}/hd<br>"
                f"Days: {d:.0f}<br>Breakeven: ${b:.0f}/cwt"
                for a, p, d, b in zip(adgs, profits, days_list, breakevens)
            ],
            hovertemplate="%{text}<extra></extra>",
        ))
        fig_adg.add_trace(go.Scatter(
            x=[best["adg_lb"]], y=[best["profit"]],
            mode="markers",
            marker=dict(size=16, color="#FFB300", symbol="star",
                        line=dict(color="#1A1A1A", width=1.5)),
            name=f"Best: {best['adg_lb']:.1f} lb/d",
        ))

        current = next((s for s in scenarios if abs(s["adg_lb"] - target_adg) < 0.2), None)
        if current and abs(current["adg_lb"] - best["adg_lb"]) > 0.2:
            fig_adg.add_trace(go.Scatter(
                x=[current["adg_lb"]], y=[current["profit"]],
                mode="markers",
                marker=dict(size=14, color=BLUE, symbol="diamond",
                            line=dict(color="#1A1A1A", width=1.5)),
                name=f"Current: {target_adg:.1f} lb/d",
            ))

        fig_adg.add_hline(y=0, line_dash="dash", line_color=GRAY, opacity=0.4)
        fig_adg.update_layout(
            template="plotly_white", height=380,
            xaxis_title="Average Daily Gain (lb/day)",
            yaxis_title="Profit ($/head)",
            hovermode="closest", legend=dict(orientation="h", y=1.02),
            margin=dict(t=10, b=50, l=60, r=20),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_adg, use_container_width=True)

        table = [{
            "ADG": f"{s['adg_lb']:.1f}",
            "Days": f"{s['days']:.0f}",
            "Feed $/d": f"${s['feed_per_day']:.2f}",
            "Breakeven": f"${s['breakeven']:.0f}/cwt",
            "Profit/hd": f"${s['profit']:,.0f}",
        } for s in scenarios]
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    else:
        st.warning("No feasible ADG scenarios found.")

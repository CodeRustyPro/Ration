"""
Market Intelligence Agent — Price Alerts & Ration Switch Recommendations.

Analyzes ingredient price trends from cached futures data to generate
proactive alerts when price ratios change enough to warrant ration
reformulation.

Key decision thresholds (from agricultural economics research):
  • DDGS:Corn price ratio > 0.7 → increase DDGS inclusion
  • DDGS:Corn price ratio < 0.5 → favor corn over DDGS
  • SBM:DDGS price ratio > 2.0 → replace SBM protein with DDGS
  • Compare $/lb TDN across all energy sources for substitution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.nrc_data import INGREDIENTS
from src.usda_fetcher import FALLBACK_PRICES_PER_TON


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PriceAlert:
    """A single market intelligence alert."""
    severity: str       # "high", "medium", "low"
    emoji: str          # 🔴, 🟠, 🟡
    title: str          # Short headline
    detail: str         # Explanation with numbers
    action: str         # What to do
    savings_per_head_per_day: float = 0.0  # Estimated savings


@dataclass
class MarketInsight:
    """Aggregated market intelligence output."""
    alerts: List[PriceAlert] = field(default_factory=list)
    price_ratios: Dict[str, float] = field(default_factory=dict)
    opportunity_score: float = 0.0  # 0-100, how much room for savings
    total_potential_savings: float = 0.0  # $/head/day


# ---------------------------------------------------------------------------
# Price analysis
# ---------------------------------------------------------------------------

def _compute_price_per_lb_tdn(price_per_ton: float, tdn_pct: float) -> float:
    """Cost per lb of TDN = (price $/ton DM) / (2000 lb/ton × TDN fraction)."""
    if tdn_pct <= 0:
        return float("inf")
    return price_per_ton / (2000 * tdn_pct / 100)


def _compute_price_per_lb_cp(price_per_ton: float, cp_pct: float) -> float:
    """Cost per lb of crude protein."""
    if cp_pct <= 0:
        return float("inf")
    return price_per_ton / (2000 * cp_pct / 100)


def analyze_prices(
    current_prices: Dict[str, float],
    historical_prices: Optional[Dict[str, float]] = None,
) -> MarketInsight:
    """Analyze current ingredient prices and generate alerts.

    Parameters
    ----------
    current_prices      {ingredient_key: $/ton DM} — current market prices.
    historical_prices   {ingredient_key: $/ton DM} — 30-day average (optional).

    Returns
    -------
    MarketInsight with prioritized alerts and recommendations.
    """
    prices = dict(FALLBACK_PRICES_PER_TON)
    prices.update(current_prices)

    if historical_prices is None:
        historical_prices = dict(FALLBACK_PRICES_PER_TON)

    alerts: List[PriceAlert] = []
    ratios: Dict[str, float] = {}

    corn_price = prices.get("CornGrain", 195)
    ddgs_price = prices.get("DDGS", 185)
    sbm_price = prices.get("SoybeanMeal", 380)
    silage_price = prices.get("CornSilage", 120)
    alfalfa_price = prices.get("AlfalfaHay", 230)
    grass_price = prices.get("GrassHay", 150)

    # --- DDGS:Corn price ratio ---
    ddgs_corn_ratio = ddgs_price / corn_price if corn_price > 0 else 1.0
    ratios["DDGS:Corn"] = round(ddgs_corn_ratio, 3)

    if ddgs_corn_ratio < 0.50:
        alerts.append(PriceAlert(
            severity="high", emoji="🔴",
            title="DDGS is very cheap relative to corn",
            detail=f"DDGS:Corn ratio is {ddgs_corn_ratio:.2f} (threshold: <0.50). "
                   f"DDGS at ${ddgs_price:.0f}/ton vs corn at ${corn_price:.0f}/ton.",
            action="Maximize DDGS to 30-40% of diet. It provides protein AND energy cheaper than corn + SBM combined.",
            savings_per_head_per_day=0.08,
        ))
    elif ddgs_corn_ratio > 0.95:
        alerts.append(PriceAlert(
            severity="medium", emoji="🟠",
            title="DDGS has lost its cost advantage over corn",
            detail=f"DDGS:Corn ratio is {ddgs_corn_ratio:.2f} (threshold: >0.95). "
                   f"DDGS at ${ddgs_price:.0f}/ton vs corn at ${corn_price:.0f}/ton.",
            action="Reduce DDGS and increase corn grain. Use Urea for protein supplementation at lower cost.",
            savings_per_head_per_day=0.05,
        ))

    # --- SBM:DDGS price ratio ---
    sbm_ddgs_ratio = sbm_price / ddgs_price if ddgs_price > 0 else 2.0
    ratios["SBM:DDGS"] = round(sbm_ddgs_ratio, 3)

    if sbm_ddgs_ratio > 2.2:
        alerts.append(PriceAlert(
            severity="high", emoji="🔴",
            title="Soybean meal is expensive — switch protein to DDGS",
            detail=f"SBM costs {sbm_ddgs_ratio:.1f}× more than DDGS. "
                   f"SBM at ${sbm_price:.0f}/ton vs DDGS at ${ddgs_price:.0f}/ton.",
            action="Replace SBM protein with DDGS. At 30% CP, DDGS provides 62% of SBM's protein per ton at a fraction of the cost.",
            savings_per_head_per_day=0.10,
        ))
    elif sbm_ddgs_ratio < 1.5:
        alerts.append(PriceAlert(
            severity="low", emoji="🟡",
            title="SBM is competitively priced — consider quality protein",
            detail=f"SBM:DDGS ratio is {sbm_ddgs_ratio:.1f}×. "
                   f"SBM has better amino acid profile and no sulfur risk.",
            action="SBM is worth considering for younger cattle where bypass protein matters.",
            savings_per_head_per_day=0.0,
        ))

    # --- Energy cost comparison ($/lb TDN) ---
    energy_costs = {}
    for key in ["CornGrain", "DDGS", "CornSilage", "AlfalfaHay", "GrassHay"]:
        if key in prices and key in INGREDIENTS:
            tdn = INGREDIENTS[key]["tdn"]
            cost = _compute_price_per_lb_tdn(prices[key], tdn)
            energy_costs[key] = cost

    if energy_costs:
        cheapest = min(energy_costs, key=energy_costs.get)
        most_expensive = max(energy_costs, key=energy_costs.get)
        ratios["cheapest_energy"] = cheapest
        cost_spread = energy_costs[most_expensive] / energy_costs[cheapest] if energy_costs[cheapest] > 0 else 1

        if cost_spread > 2.5:
            alerts.append(PriceAlert(
                severity="medium", emoji="🟠",
                title=f"Energy cost spread is wide — favor {INGREDIENTS[cheapest]['display_name'].split(',')[0]}",
                detail=f"Cheapest energy: {INGREDIENTS[cheapest]['display_name']} at "
                       f"${energy_costs[cheapest]:.3f}/lb TDN. "
                       f"Most expensive: {INGREDIENTS[most_expensive]['display_name']} at "
                       f"${energy_costs[most_expensive]:.3f}/lb TDN.",
                action=f"The optimizer should already favor {INGREDIENTS[cheapest]['display_name'].split(',')[0]}, "
                       f"but verify its inclusion isn't hitting its max constraint.",
                savings_per_head_per_day=0.04,
            ))

    # --- Price movements vs. historical ---
    for key, current in prices.items():
        hist = historical_prices.get(key)
        if hist and hist > 0:
            pct_change = (current - hist) / hist * 100
            if abs(pct_change) > 10:
                ing_name = INGREDIENTS.get(key, {}).get("display_name", key)
                direction = "surged" if pct_change > 0 else "dropped"
                alerts.append(PriceAlert(
                    severity="medium" if abs(pct_change) > 15 else "low",
                    emoji="📈" if pct_change > 0 else "📉",
                    title=f"{ing_name.split(',')[0]} price has {direction} {abs(pct_change):.0f}%",
                    detail=f"Current: ${current:.0f}/ton vs 30-day avg: ${hist:.0f}/ton.",
                    action="Re-optimize your ration with current prices to lock in savings."
                           if pct_change < 0 else
                           "Consider forward contracting or substituting with a cheaper alternative.",
                    savings_per_head_per_day=0.02 if pct_change < -10 else 0.0,
                ))

    # No alerts? That's fine — tell the farmer
    if not alerts:
        alerts.append(PriceAlert(
            severity="low", emoji="🟢",
            title="Markets are stable — no ration changes needed",
            detail="All ingredient price ratios are within normal ranges.",
            action="Your current ration is well-optimized for today's market.",
        ))

    # Sort by severity and savings
    severity_order = {"high": 0, "medium": 1, "low": 2}
    alerts.sort(key=lambda a: (severity_order.get(a.severity, 3), -a.savings_per_head_per_day))

    total_savings = sum(a.savings_per_head_per_day for a in alerts)
    opportunity = min(100, total_savings / 0.30 * 100)  # normalize to 0-100

    return MarketInsight(
        alerts=alerts,
        price_ratios=ratios,
        opportunity_score=round(opportunity, 1),
        total_potential_savings=round(total_savings, 3),
    )


# ---------------------------------------------------------------------------
# Shadow Price Interpretation
# ---------------------------------------------------------------------------

# Maps constraint names from the LP to farmer-readable descriptions
_CONSTRAINT_LABELS = {
    "cp_min": ("Crude Protein minimum", "protein"),
    "cp_max": ("Crude Protein maximum", "protein"),
    "tdn_min": ("TDN (energy) minimum", "energy"),
    "tdn_max": ("TDN (energy) maximum", "energy"),
    "ca_min": ("Calcium minimum", "mineral"),
    "ca_max": ("Calcium maximum", "mineral"),
    "p_min": ("Phosphorus minimum", "mineral"),
    "p_max": ("Phosphorus maximum", "mineral"),
    "sulfur_min": ("Sulfur minimum", "mineral"),
    "sulfur_max": ("Sulfur maximum (NRC 0.30%)", "mineral"),
    "ndf_min": ("NDF fiber minimum", "fiber"),
    "ndf_max": ("NDF fiber maximum", "fiber"),
    "nem_min": ("Maintenance energy (NEm)", "energy"),
    "neg_min": ("Gain energy (NEg)", "energy"),
    "EE_max": ("Fat (ether extract) maximum", "fat"),
    "Roughage_min": ("Roughage minimum (rumen health)", "roughage"),
    "Roughage_max": ("Roughage maximum", "roughage"),
    "Ca_P_ratio": ("Calcium:Phosphorus ratio ≥ 2:1", "mineral"),
}

_CONSTRAINT_ACTIONS = {
    "sulfur_max": "If you can tolerate higher sulfur (monitor for polio), increasing the cap allows more DDGS.",
    "Roughage_max": "Your cattle's ADG target limits roughage. This is correct — higher-energy diets need less forage.",
    "Roughage_min": "The 8% roughage floor protects rumen health. Removing it risks acidosis.",
    "cp_min": "Consider adding 0.2-0.5% Urea to meet protein at lower cost.",
    "tdn_min": "Increasing corn grain or DDGS inclusion will raise diet energy density.",
    "neg_min": "More energy-dense feeds (corn, DDGS) are needed to meet gain target.",
    "Ca_P_ratio": "Limestone is the cheapest way to fix the Ca:P ratio. Add 0.5-1% more.",
    "EE_max": "Fat content is at the 6.5% ceiling. DDGS is the main fat contributor — reduce if needed.",
}


def interpret_shadow_prices(
    shadow_prices: Dict[str, float],
    dmi_lb: float,
) -> List[Dict]:
    """Translate LP shadow prices into farmer-readable insights.

    Parameters
    ----------
    shadow_prices  {constraint_name: dual_value} from get_shadow_prices().
    dmi_lb         Daily DMI in pounds (for per-head calculations).

    Returns
    -------
    List of dicts with keys: constraint, label, category, impact_per_day,
    direction, insight, action — sorted by absolute impact.
    """
    insights = []

    for constraint, dual in shadow_prices.items():
        if abs(dual) < 0.0001:
            continue  # Not binding

        label_info = _CONSTRAINT_LABELS.get(constraint)
        if not label_info:
            continue

        label, category = label_info

        # Shadow price = $/day change per unit RHS relaxation
        # Positive dual on ≥ constraint → relaxing (lowering) it saves money
        # Negative dual on ≤ constraint → relaxing (raising) it saves money
        impact = abs(dual)
        if "_min" in constraint or constraint in ("Ca_P_ratio", "neg_min", "nem_min"):
            direction = "Lowering this requirement"
        else:
            direction = "Raising this limit"

        action = _CONSTRAINT_ACTIONS.get(constraint, "Consider adjusting this constraint if nutritionally safe.")

        insights.append({
            "constraint": constraint,
            "label": label,
            "category": category,
            "impact_per_day": round(impact, 4),
            "impact_30_head": round(impact * 30, 2),
            "direction": direction,
            "insight": f"{direction} by 1 unit would save ${impact:.3f}/head/day (${impact*30:.2f} for 30 head).",
            "action": action,
        })

    # Sort by impact (biggest cost driver first)
    insights.sort(key=lambda x: -x["impact_per_day"])
    return insights


# ---------------------------------------------------------------------------
# Ration Verdict (Farmer-Speak Summary)
# ---------------------------------------------------------------------------

def generate_ration_verdict(
    nutrient_totals: Dict,
    requirements: Dict,
    cost_per_lb_gain: float,
    adg_lb: float,
    dmi_lb: float,
    use_ionophore: bool = True,
) -> Dict:
    """Generate a farmer-readable ration assessment.

    Returns
    -------
    Dict with keys: verdict, energy_label, risk_flags, compliance_summary,
    efficiency_rating, comparison_text
    """
    tdn = nutrient_totals.get("tdn", 70)
    ndf = nutrient_totals.get("ndf", 20)
    cp = nutrient_totals.get("cp", 12)
    roughage_max = requirements.get("roughage_max", 15)

    # Energy classification
    if tdn >= 80:
        energy_label = "🔥 Hot Diet (High Energy)"
        energy_note = "High-concentrate finishing ration. Watch for acidosis."
    elif tdn >= 70:
        energy_label = "⚡ Moderate-High Energy"
        energy_note = "Good finishing energy density."
    elif tdn >= 63:
        energy_label = "🌿 Growing Diet (Moderate Energy)"
        energy_note = "Suitable for backgrounding/stocker programs."
    else:
        energy_label = "🌾 Maintenance Diet (Low Energy)"
        energy_note = "Low-energy — appropriate only for maintenance."

    # Risk flags
    risk_flags = []
    if ndf < 15:
        risk_flags.append("⚠️ **Low fiber (NDF < 15%)** — acidosis risk. Ensure adequate effective fiber.")
    if roughage_max < 10 and tdn > 82:
        risk_flags.append("⚠️ **Very low roughage** — monitor for bloat and liver abscesses.")
    if cp > 15:
        risk_flags.append("💡 **High protein** — you may be over-supplementing. Check if Urea or SBM can be reduced.")
    if cp < 10 and adg_lb > 2.0:
        risk_flags.append("⚠️ **Low protein for this ADG target** — gain may be limited by protein, not energy.")

    # Efficiency rating relative to industry benchmarks
    # Industry average COG for finishing: ~$0.55-0.65/lb gain (2025-26)
    if cost_per_lb_gain < 0.45:
        efficiency = "🏆 **Excellent** — well below industry average"
    elif cost_per_lb_gain < 0.55:
        efficiency = "✅ **Good** — competitive with top-performing feedlots"
    elif cost_per_lb_gain < 0.65:
        efficiency = "⚠️ **Average** — room for improvement"
    else:
        efficiency = "🔴 **Above average cost** — review ingredient sourcing"

    # Compliance summary
    violations = 0
    borderline = 0
    for key in ["cp", "tdn", "ca", "p", "sulfur", "ndf"]:
        val = nutrient_totals.get(key)
        lo_key = f"{key}_min"
        hi_key = f"{key}_max"
        lo = requirements.get(lo_key)
        hi = requirements.get(hi_key)
        if val is not None and lo is not None and val < lo:
            violations += 1
        elif val is not None and hi is not None and val > hi:
            violations += 1
        elif val is not None and lo is not None:
            margin = (val - lo) / (lo + 0.001)
            if margin < 0.05:
                borderline += 1

    if violations == 0 and borderline == 0:
        compliance = "✅ All NRC requirements met with comfortable margins"
    elif violations == 0:
        compliance = f"✅ All requirements met — {borderline} nutrient(s) are borderline"
    else:
        compliance = f"⚠️ {violations} requirement(s) not met — review ration"

    # Build verdict
    cost_str = f"${cost_per_lb_gain:.3f}/lb"
    verdict = (
        f"**{energy_label}** — This {adg_lb:.1f} lb/d ration costs **{cost_str}** per lb of gain. "
        f"{efficiency}."
    )

    return {
        "verdict": verdict,
        "energy_label": energy_label,
        "energy_note": energy_note,
        "risk_flags": risk_flags,
        "compliance_summary": compliance,
        "efficiency_rating": efficiency,
        "cost_per_lb_gain": cost_per_lb_gain,
    }

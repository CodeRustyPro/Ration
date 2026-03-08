"""
Multi-phase feeding program generator.

Produces a 4-phase step-up protocol for transitioning cattle from
receiving to finishing. Each phase has different roughage constraints
reflecting safe rumen adaptation (Samuelson et al. 2016, NASEM 2016).

Phase 1 — Receiving:  40-55% roughage (rumen adaptation, stress recovery)
Phase 2 — Step-Up:    20-35% roughage (increasing concentrate)
Phase 3 — Transition: 12-20% roughage (near-finishing density)
Phase 4 — Finisher:    8-12% roughage (full finishing ration)
"""

from typing import Dict, List, Optional, Tuple

from src.nrc_data import INGREDIENTS
from src.optimizer import optimize_ration

PHASES = [
    {
        "name": "Receiving",
        "label": "Phase 1 — Receiving",
        "desc": "High roughage for rumen adaptation and stress recovery",
        "rough_min": 40.0,
        "rough_max": 55.0,
        "adg_frac": 0.50,
        "day_frac": 0.10,
    },
    {
        "name": "Step-Up",
        "label": "Phase 2 — Step-Up",
        "desc": "Gradually increasing concentrate",
        "rough_min": 20.0,
        "rough_max": 35.0,
        "adg_frac": 0.65,
        "day_frac": 0.15,
    },
    {
        "name": "Transition",
        "label": "Phase 3 — Transition",
        "desc": "Near-finishing energy density",
        "rough_min": 12.0,
        "rough_max": 20.0,
        "adg_frac": 0.85,
        "day_frac": 0.15,
    },
    {
        "name": "Finisher",
        "label": "Phase 4 — Finisher",
        "desc": "Full finishing ration for maximum gain",
        "rough_min": 8.0,
        "rough_max": 12.0,
        "adg_frac": 1.0,
        "day_frac": 0.60,
    },
]


def generate_feeding_program(
    start_wt: float,
    target_wt: float,
    target_adg: float,
    price_overrides: Optional[Dict[str, float]] = None,
    use_ionophore: bool = True,
    excluded_ingredients: Optional[Tuple] = None,
) -> List[Dict]:
    """Generate a 4-phase feeding program with as-fed amounts.

    Returns a list of phase dicts each containing:
      - Phase info (name, description, day range, weight range)
      - Optimizer result (ration with as-fed amounts, cost)
    """
    total_gain = target_wt - start_wt
    total_days = total_gain / target_adg if target_adg > 0 else 180

    results = []
    cum_days = 0
    cum_wt = start_wt

    for i, phase in enumerate(PHASES):
        phase_adg = round(target_adg * phase["adg_frac"], 1)
        phase_adg = max(phase_adg, 0.5)

        if i == len(PHASES) - 1:
            # Last phase: absorb remaining gain to reach target weight
            remaining_gain = max(0, target_wt - cum_wt)
            phase_days = max(7, int(remaining_gain / phase_adg))
        else:
            phase_days = max(7, int(total_days * phase["day_frac"]))

        mid_wt = cum_wt + phase_adg * phase_days / 2

        opt = optimize_ration(
            mid_wt,
            phase_adg,
            price_overrides=price_overrides,
            use_ionophore=use_ionophore,
            excluded_ingredients=excluded_ingredients,
            roughage_override=(phase["rough_min"], phase["rough_max"]),
        )

        end_wt = cum_wt + phase_adg * phase_days

        as_fed_ration = {}
        cost_per_day = None
        if opt:
            cost_per_day = opt["total_cost_per_day"]
            for key, v in opt["ration"].items():
                dm_pct = INGREDIENTS[key]["dm_pct"]
                as_fed_ration[key] = {
                    "name": v["display_name"],
                    "dm_lb": v["lb_per_day"],
                    "as_fed_lb": round(v["lb_per_day"] * 100.0 / dm_pct, 2),
                    "pct": v["pct_of_dmi"],
                    "cost": v["cost_per_day"],
                }

        results.append({
            "name": phase["name"],
            "label": phase["label"],
            "desc": phase["desc"],
            "day_start": cum_days + 1,
            "day_end": cum_days + phase_days,
            "days": phase_days,
            "wt_start": round(cum_wt),
            "wt_end": round(end_wt),
            "adg": phase_adg,
            "roughage_min": phase["rough_min"],
            "roughage_max": phase["rough_max"],
            "cost_per_day": cost_per_day,
            "ration": as_fed_ration,
        })

        cum_days += phase_days
        cum_wt = end_wt

    return results


# ---------------------------------------------------------------------------
# ADG decline model
# ---------------------------------------------------------------------------
# As cattle get heavier, feed efficiency declines. Fatter cattle require
# more net energy per pound of gain (NASEM 2016: NEg scales as BW^0.75).
# A constant ADG assumption overstates gain at heavier weights.
#
# Empirical model (K-State Focus on Feedlots, ISU closeouts):
#   ADG declines roughly 0.15 lb/d per 100 lb of BW above 900 lb
#   (e.g., 3.5 lb/d at 900 lb → 2.9 lb/d at 1300 lb)
#
# This produces a more realistic S-curve for weight gain and catches
# the profit peak earlier than a linear gain assumption.

def _adg_at_weight(base_adg: float, start_wt: float, current_wt: float) -> float:
    """Estimate ADG adjusted for declining feed efficiency at heavier weights.

    Below 900 lb, ADG equals the base rate. Above 900 lb, ADG declines
    by ~0.15 lb/d per 100 lb, floored at 50% of base ADG.
    """
    threshold = 900.0
    if current_wt <= threshold:
        return base_adg
    excess = (current_wt - threshold) / 100.0
    decline = 0.15 * excess
    return max(base_adg * 0.50, base_adg - decline)


def compute_exit_analysis(
    profile: Dict,
    feed_cost_per_day: float,
    adg: float,
    max_extra_days: int = 60,
    price_curve: Optional[Dict[int, float]] = None,
) -> List[Dict]:
    """Compute profit at different exit points to find optimal market timing.

    Uses a declining-efficiency ADG model: as cattle get heavier, gain
    slows (NASEM 2016, K-State closeouts). This produces a realistic
    profit peak rather than the monotonically-increasing profit curve
    that a constant-ADG assumption would generate.

    Parameters
    ----------
    price_curve : optional dict mapping day number → forecasted sale price ($/cwt).
                  When provided, revenue at each day uses the predicted market price
                  instead of the constant profile sale price. This enables
                  "sell lighter into a stronger market" analysis.

    Returns list of dicts with keys:
      day, weight, profit, daily_gain, marginal_cost, marginal_revenue,
      total_cost, revenue, cost_of_gain, sale_price_cwt
    """
    start_wt = profile["start_weight"]
    target_wt = profile["target_weight"]
    sale_price = profile["sale_price_cwt"]
    purchase_cost = start_wt * profile["purchase_price_cwt"] / 100.0
    freight = start_wt * profile.get("freight_dt_cwt", 0.0) / 100.0
    laid_in = purchase_cost + freight
    pencil = profile.get("pencil_shrink_pct", 4.0)
    yardage = profile.get("yardage_cost", 0.55)
    rate = profile.get("interest_rate", 8.0) / 100.0
    dl = profile.get("death_loss_pct", 1.5)
    vet = profile.get("vet_cost", 0.0)
    eff_laid_in = laid_in / (1.0 - dl / 100.0) if dl < 100 else laid_in

    total_gain = target_wt - start_wt
    base_days = int(total_gain / max(adg, 0.5))
    max_days = base_days + max_extra_days

    results = []
    current_wt = start_wt

    for day in range(1, max_days + 1):
        # ADG declines at heavier weights (feed efficiency deterioration)
        day_adg = _adg_at_weight(adg, start_wt, current_wt)
        current_wt += day_adg

        # Use forecasted sale price if available, otherwise constant
        day_sale_price = sale_price
        if price_curve:
            day_sale_price = price_curve.get(day, sale_price)

        feed = feed_cost_per_day * day
        yard = yardage * day
        interest = eff_laid_in * rate * day / 365.0
        total_cost = eff_laid_in + feed + yard + interest + vet
        shrunk_wt = current_wt * (1.0 - pencil / 100.0)
        revenue = shrunk_wt * day_sale_price / 100.0
        profit = revenue - total_cost

        # Marginal analysis (day-specific because ADG varies)
        marginal_cost = feed_cost_per_day + yardage + eff_laid_in * rate / 365.0
        marginal_revenue = day_adg * day_sale_price / 100.0 * (1.0 - pencil / 100.0)

        # Cost of gain (operating costs only, excluding purchase)
        gain_so_far = current_wt - start_wt
        cog = (feed + yard + interest + vet) / gain_so_far if gain_so_far > 0 else 0

        results.append({
            "day": day,
            "weight": round(current_wt),
            "daily_gain": round(day_adg, 2),
            "profit": round(profit, 2),
            "marginal_cost": round(marginal_cost, 2),
            "marginal_revenue": round(marginal_revenue, 2),
            "total_cost": round(total_cost, 2),
            "revenue": round(revenue, 2),
            "cost_of_gain": round(cog, 3),
            "sale_price_cwt": round(day_sale_price, 2),
        })

    return results


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------
# Extension research (ISU, K-State) shows cattle prices dominate profit
# variability. This function quantifies the $/head impact of plausible
# changes to each parameter, ranked by sensitivity.

def compute_sensitivity(
    profile: Dict,
    feed_cost_per_day: float,
    days_on_feed: float,
) -> List[Dict]:
    """Compute $/head profit sensitivity to each economic parameter.

    Uses the actual profile as the base case, then perturbs each parameter
    by a realistic amount (based on Extension research) and measures the
    profit impact.

    Returns list of {parameter, base_value, scenario, impact_per_head,
    impact_200_head} sorted by absolute impact descending.
    """
    from src.economics import calculate_costs

    # Base case profit
    base = calculate_costs(
        profile, feed_cost_per_day, days_on_feed, profile["sale_price_cwt"]
    )
    base_profit = base.projected_profit_head

    # Perturbation scenarios: (parameter_key, display_name, delta, unit, fmt)
    # Deltas calibrated to Extension sensitivity tables (ISU, K-State)
    scenarios = [
        ("sale_price_cwt", "Sale price", 10.0, "$/cwt", ".0f"),
        ("purchase_price_cwt", "Purchase price", -10.0, "$/cwt", ".0f"),
        ("yardage_cost", "Yardage", -0.25, "$/hd/d", ".2f"),
        ("death_loss_pct", "Death loss", -1.0, "%", ".1f"),
        ("pencil_shrink_pct", "Pencil shrink", -1.0, "%", ".1f"),
        ("interest_rate", "Interest rate", -1.0, "%", ".1f"),
        ("vet_cost", "Vet/processing", -5.0, "$/hd", ".0f"),
        ("freight_dt_cwt", "Freight", -1.0, "$/cwt", ".0f"),
    ]

    head_count = profile.get("head_count", 200)

    results = []
    for key, name, delta, unit, fmt in scenarios:
        perturbed = dict(profile)
        perturbed[key] = profile.get(key, 0) + delta
        try:
            perturbed_result = calculate_costs(
                perturbed, feed_cost_per_day, days_on_feed,
                perturbed.get("sale_price_cwt", profile["sale_price_cwt"]),
            )
            impact = perturbed_result.projected_profit_head - base_profit
        except Exception:
            impact = 0

        # Direction label
        sign = "+" if delta > 0 else "-"
        scenario_label = f"{sign}{abs(delta):{fmt}} {unit}"

        results.append({
            "parameter": name,
            "base_value": f"{profile.get(key, 0):{fmt}}",
            "scenario": scenario_label,
            "impact_per_head": round(impact, 0),
            "impact_herd": round(impact * head_count, 0),
        })

    # Add feed cost sensitivity directly (not a profile parameter)
    # Corn is ~60% of feed cost; a $1/bu swing ≈ $35/ton ≈ ~$0.33/hd/d
    feed_delta = 0.33
    feed_perturbed = calculate_costs(
        profile, feed_cost_per_day - feed_delta, days_on_feed, profile["sale_price_cwt"]
    )
    feed_impact = feed_perturbed.projected_profit_head - base_profit
    results.append({
        "parameter": "Feed cost (corn -$1/bu)",
        "base_value": f"${feed_cost_per_day:.2f}/d",
        "scenario": f"-$0.33/d",
        "impact_per_head": round(feed_impact, 0),
        "impact_herd": round(feed_impact * head_count, 0),
    })

    # Sort by absolute impact descending
    results.sort(key=lambda x: -abs(x["impact_per_head"]))
    return results

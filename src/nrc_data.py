"""
NRC/NASEM 2016 beef cattle nutrient requirements and feed ingredient composition.

Requirements are computed dynamically from body weight (BW) and target ADG,
following NASEM 2016 (8th edition) equations and OSU Extension E-974 (2025).

All composition values are on a DM (dry matter) basis.

Audit fixes (2026-03):
  • Roughage: piecewise/exponential replacing linear (Samuelson et al. 2016)
  • TDN: concave power-law replacing linear slope (NRC tabular data)
  • DMI: added DMIR (DMI Required) approach (NASEM BCNRM preferred)
  • Ionophore: default adjustment (97.3% of US feedlots; Samuelson et al. 2016)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Ingredient composition database (DM basis)
# ---------------------------------------------------------------------------
# Each ingredient entry:
#   dm_pct   – dry matter content (%)
#   cp       – crude protein (% of DM)
#   tdn      – total digestible nutrients (% of DM)
#   nem      – net energy for maintenance (Mcal/lb DM)
#   neg      – net energy for gain (Mcal/lb DM)
#   ca       – calcium (% of DM)
#   p        – phosphorus (% of DM)
#   ndf      – neutral detergent fiber (% of DM)
#   ee       – ether extract / fat (% of DM)
#   sulfur   – sulfur (% of DM)
#   uip_pct  – undegradable intake protein (% of CP)
#   min_pct  – minimum inclusion (% of total diet DM)
#   max_pct  – maximum inclusion (% of total diet DM)
#   cost_usd_per_ton_dm – placeholder; overridden by live USDA prices

INGREDIENTS: Dict[str, dict] = {
    "CornGrain": {
        "display_name": "Corn, Dry Rolled",
        "dm_pct": 88.0,
        "cp": 9.0,
        "tdn": 88.0,
        "nem": 0.98,
        "neg": 0.67,
        "ca": 0.03,
        "p": 0.30,
        "ndf": 10.0,
        "ee": 3.8,
        "sulfur": 0.12,
        "uip_pct": 40.0,
        "min_pct": 0.0,
        "max_pct": 85.0,
        "cost_usd_per_ton_dm": 220.0,   # fallback price $/ton DM
    },
    "SoybeanMeal": {
        "display_name": "Soybean Meal 48%",
        "dm_pct": 90.0,
        "cp": 48.0,
        "tdn": 85.0,
        "nem": 0.98,
        "neg": 0.67,
        "ca": 0.30,
        "p": 0.68,
        "ndf": 10.0,
        "ee": 1.5,
        "sulfur": 0.40,
        "uip_pct": 35.0,
        "min_pct": 0.0,
        "max_pct": 20.0,
        "cost_usd_per_ton_dm": 420.0,
    },
    "DDGS": {
        "display_name": "Distillers Dried Grains w/ Solubles",
        "dm_pct": 90.0,
        "cp": 30.0,
        "tdn": 85.0,
        "nem": 0.96,
        "neg": 0.66,
        "ca": 0.06,
        "p": 0.80,
        "ndf": 36.0,
        "ee": 10.0,
        "sulfur": 0.45,   # high-S feed; limits inclusion when combined with SBM
        "uip_pct": 55.0,
        "min_pct": 0.0,
        "max_pct": 40.0,
        "cost_usd_per_ton_dm": 200.0,
    },
    "CornSilage": {
        "display_name": "Corn Silage",
        "dm_pct": 34.0,
        "cp": 8.0,
        "tdn": 69.0,
        "nem": 0.75,
        "neg": 0.47,
        "ca": 0.26,
        "p": 0.22,
        "ndf": 45.0,
        "ee": 3.2,
        "sulfur": 0.12,
        "uip_pct": 20.0,
        "min_pct": 0.0,
        "max_pct": 20.0,   # LP-level roughage_max constraint further limits at high ADG
        "cost_usd_per_ton_dm": 120.0,   # $/ton DM (Iowa State rule: 8-10× corn bu price)
    },
    "AlfalfaHay": {
        "display_name": "Alfalfa Hay",
        "dm_pct": 89.0,
        "cp": 19.0,
        "tdn": 58.0,
        "nem": 0.58,
        "neg": 0.32,
        "ca": 1.35,
        "p": 0.28,
        "ndf": 44.0,
        "ee": 2.2,
        "sulfur": 0.25,
        "uip_pct": 25.0,
        "min_pct": 0.0,
        "max_pct": 10.0,   # LP-level roughage_max constraint further limits at high ADG
        "cost_usd_per_ton_dm": 250.0,
    },
    "GrassHay": {
        "display_name": "Grass Hay",
        "dm_pct": 89.0,
        "cp": 8.0,
        "tdn": 53.0,
        "nem": 0.49,
        "neg": 0.24,
        "ca": 0.45,
        "p": 0.22,
        "ndf": 62.0,
        "ee": 2.0,
        "sulfur": 0.18,
        "uip_pct": 30.0,
        "min_pct": 0.0,
        "max_pct": 10.0,   # LP-level roughage_max constraint further limits at high ADG
        "cost_usd_per_ton_dm": 160.0,
    },
    "Urea": {
        "display_name": "Urea (281% CP equivalent)",
        "dm_pct": 99.0,
        "cp": 281.0,   # 46% N × 6.25
        "tdn": 0.0,
        "nem": 0.00,
        "neg": 0.00,
        "ca": 0.00,
        "p": 0.00,
        "ndf": 0.0,
        "ee": 0.0,
        "sulfur": 0.00,
        "uip_pct": 0.0,
        "min_pct": 0.0,
        "max_pct": 1.0,   # hard cap: >1% risks ammonia toxicity
        "cost_usd_per_ton_dm": 600.0,
    },
    "Limestone": {
        "display_name": "Limestone (Ca supplement)",
        "dm_pct": 100.0,
        "cp": 0.0,
        "tdn": 0.0,
        "nem": 0.00,
        "neg": 0.00,
        "ca": 36.0,
        "p": 0.00,
        "ndf": 0.0,
        "ee": 0.0,
        "sulfur": 0.00,
        "uip_pct": 0.0,
        "min_pct": 0.0,
        "max_pct": 2.0,
        "cost_usd_per_ton_dm": 60.0,
    },
    "MineralPremix": {
        "display_name": "Mineral/Vitamin Premix",
        "dm_pct": 98.0,
        "cp": 0.0,
        "tdn": 0.0,
        "nem": 0.00,
        "neg": 0.00,
        "ca": 12.0,
        "p": 6.0,
        "ndf": 0.0,
        "ee": 0.0,
        "sulfur": 0.00,
        "uip_pct": 0.0,
        "min_pct": 0.1,
        "max_pct": 0.5,
        "cost_usd_per_ton_dm": 2000.0,
    },
}

# Ingredients considered roughage (effective NDF sources; rumen health)
ROUGHAGE_INGREDIENTS = {"CornSilage", "AlfalfaHay", "GrassHay"}

# ---------------------------------------------------------------------------
# NASEM 2016 requirement equations
# ---------------------------------------------------------------------------
# BW in lb, ADG in lb/d
# DMI estimated at 2.2% of BW (Merck Vet Manual: finishing cattle 2.0-2.3% BW)
# All requirements returned as absolute daily amounts (lb or Mcal) for LP constraints.

LB_PER_KG = 2.20462


def estimate_dmi(bw_lb: float, adjustment: float = 1.0) -> float:
    """Estimate dry matter intake (lb/day) at 2.2% of body weight.

    The 2.2% estimate sits at the low end of reasonable (NASEM linear eq
    gives 2.17-2.43% BW) but is acceptable for heavy finishing cattle,
    cattle receiving ionophores, or high-energy diets.
    """
    return bw_lb * 0.022 * adjustment


def compute_requirements(bw_lb: float, adg_lb: float) -> Dict[str, float]:
    """
    Compute NASEM 2016 nutrient requirements for a medium-frame beef steer.

    Returns a dict of {nutrient: value} where:
      - cp, tdn, ca, p, sulfur, ndf, ee  → % of diet DM
      - nem, neg                           → Mcal per lb of diet DM  (NOT %)
    """
    dmi = estimate_dmi(bw_lb)
    bw_kg = bw_lb / LB_PER_KG
    adg_kg = adg_lb / LB_PER_KG

    # --- Daily energy requirements (for post-solve validation only) ---
    nem_req_mcal = 0.077 * (bw_kg ** 0.75)
    ebw = 0.891 * bw_kg
    neg_req_mcal = 0.0635 * (ebw ** 0.75) * (adg_kg ** 1.097)

    # --- Absolute daily energy requirements (Mcal/day) from NASEM 2016 ---
    # These scale with both BW AND ADG, creating genuine LP tradeoffs in the
    # Pareto front: higher ADG → more NEg required → denser/costlier diet.
    adg_kg = adg_lb / LB_PER_KG
    ebw_kg = 0.891 * bw_kg
    nem_req_mcal = 0.077 * (bw_kg ** 0.75)           # maintenance Mcal/day
    neg_req_mcal = 0.0635 * (ebw_kg ** 0.75) * max(adg_kg, 0.01) ** 1.097  # gain Mcal/day

    # Lower bound only: must meet NASEM minimum for maintenance (NEm) and gain (NEg).
    # No upper bound — in a least-cost LP the optimizer won't over-feed energy
    # voluntarily, and ingredient inclusion caps handle practical limits.
    # Use a sentinel value (None) for "no upper bound".
    nem_min = nem_req_mcal    # Mcal/day (must cover maintenance)
    nem_max = None            # no upper bound
    neg_min = neg_req_mcal    # Mcal/day (must cover gain target)
    neg_max = None            # no upper bound

    # --- CP requirement (% of DM, from NASEM 2016 Table / OSU E-974) ---
    if bw_lb <= 600:
        cp_min_pct = 13.5
    elif bw_lb <= 800:
        cp_min_pct = 13.5 - (bw_lb - 600) / 200 * 1.0   # 13.5 → 12.5
    elif bw_lb <= 1000:
        cp_min_pct = 12.5 - (bw_lb - 800) / 200 * 0.5   # 12.5 → 12.0
    else:
        cp_min_pct = 12.0 - (bw_lb - 1000) / 200 * 0.5  # 12.0 → 11.5

    cp_min_pct = max(cp_min_pct, 7.5)
    cp_max_pct = cp_min_pct + 3.0

    # --- TDN requirement (% of DM) — CONCAVE POWER-LAW FIT ---
    # The old linear slope (+5%/lb/d) underestimated TDN by 5-9% in the
    # critical ADG 2.0-3.5 range. NRC tabular data shows the actual slope
    # is ~10.5%/lb from ADG 1.5→2.5, then ~5%/lb from 2.5→3.5 — a concave
    # (diminishing returns) curve, because NE_gain scales as ADG^1.097
    # but TDN% reflects both gain and maintenance divided by DMI.
    #
    # New fit: tdn_min = 63.0 + 10.5 × max(0, adg - 1.5)^0.65
    #   ADG 1.5 → 63.0%  (backgrounding baseline, matches NRC 62.5-63.0%)
    #   ADG 2.0 → 69.5%  (NRC: ~69%, old code: 65.5%)
    #   ADG 2.5 → 73.2%  (NRC: 73-74%, old code: 68%)
    #   ADG 3.0 → 76.0%  (NRC: 76-77%, old code: 70.5%)
    #   ADG 3.5 → 78.3%  (NRC: 78-79%, old code: 73%)
    #   ADG 4.0 → 80.3%  (old code: 75.5%)
    adg_above = max(0.0, adg_lb - 1.5)
    tdn_min_pct = 63.0 + 10.5 * (adg_above ** 0.65)
    tdn_min_pct = max(63.0, min(tdn_min_pct, 83.0))
    tdn_max_pct = 88.0

    # --- Mineral requirements (% of DM) ---
    ca_maintenance = 0.0154 * bw_kg                        # g/d
    ca_gain = 7.1 * adg_kg                                 # g/d
    ca_total_g = ca_maintenance + ca_gain
    ca_pct = (ca_total_g / 1000) / (dmi / LB_PER_KG) * 100

    p_maintenance = 0.0143 * bw_kg
    p_gain = 4.5 * adg_kg
    p_total_g = p_maintenance + p_gain
    p_pct = (p_total_g / 1000) / (dmi / LB_PER_KG) * 100

    ca_pct = max(ca_pct, 2.0 * p_pct)  # Ca:P ≥ 2:1 prevents urinary calculi

    # --- Roughage maximum (% of DMI) — PIECEWISE/EXPONENTIAL ---
    # Old linear ramp (-8 pp/lb ADG) was too low at backgrounding (35% vs
    # reality 50-80%) and too high at finishing (23% vs reality 8-15%).
    # Samuelson et al. (2016): finishing = 8-10%, backgrounding = 40.7%.
    # NASEM 2016: 5-10% forage in high-concentrate finishing diets.
    # Saskatchewan Ag: backgrounding at 60-70% forage.
    #
    # Piecewise function:
    #   ADG ≤ 1.5: 70% max (stocker/backgrounding programs)
    #   ADG 1.5–2.5: exponential decay 70% → 15%
    #   ADG 2.5–3.5: linear 15% → 10%
    #   ADG ≥ 3.5: cap at 8% (aggressive finishing; NASEM 5-10%)
    if adg_lb <= 1.5:
        roughage_max_pct = 70.0
    elif adg_lb <= 2.5:
        # Exponential decay from 70 to 15 over 1.0 lb/d range
        t = (adg_lb - 1.5) / 1.0  # 0 → 1
        roughage_max_pct = 15.0 + (70.0 - 15.0) * math.exp(-2.5 * t)
    elif adg_lb <= 3.5:
        # Linear taper from 15 to 10
        t = (adg_lb - 2.5) / 1.0
        roughage_max_pct = 15.0 - 5.0 * t
    else:
        roughage_max_pct = 8.0
    roughage_max_pct = max(8.0, min(roughage_max_pct, 70.0))

    return {
        # % of diet DM
        "cp_min": round(cp_min_pct, 2),
        "cp_max": round(cp_max_pct, 2),
        "tdn_min": round(tdn_min_pct, 2),
        "tdn_max": round(tdn_max_pct, 2),
        "ca_min": round(max(0.30, ca_pct), 3),
        "ca_max": 0.70,
        "p_min": round(max(0.18, p_pct), 3),
        "p_max": 0.35,
        "sulfur_min": 0.15,
        "sulfur_max": 0.30,   # NRC 2005 revised: 0.30% for <15% forage diets
        "ndf_min": 12.0,
        "ndf_max": 35.0,
        "ee_max": 6.5,
        # Mcal/day absolute (nem_max/neg_max = None means no upper bound)
        "nem_min": round(nem_min, 3),
        "nem_max": nem_max,
        "neg_min": round(neg_min, 3),
        "neg_max": neg_max,
        # Scalars for display/validation
        "dmi_lb": round(dmi, 2),
        "nem_req_mcal": round(nem_req_mcal, 3),
        "neg_req_mcal": round(neg_req_mcal, 3),
        # Roughage limits (% of DMI)
        "roughage_min": 8.0,
        "roughage_max": round(roughage_max_pct, 1),
    }


def get_ingredient_matrix() -> Tuple[Dict, Dict, Dict]:
    """
    Return three dicts for the LP:
      nutrients  – {nutrient_name: {ingredient: value_per_lb_DM}}
      costs      – {ingredient: $/lb DM}  (fallback prices)
      bounds     – {ingredient: (min_lb, max_lb)} as fractions of total DMI

    Unit conventions:
      Percentage nutrients (cp, tdn, ca, p, ndf, ee, sulfur):
        stored as fraction 0–1 (e.g. 9% CP → 0.09)
      Energy density nutrients (nem, neg):
        stored as Mcal/lb DM  (e.g. 0.98 for corn)
        These are NOT divided by 100.
    """
    pct_nutrients = ["cp", "tdn", "ca", "p", "ndf", "ee", "sulfur"]
    energy_nutrients = ["nem", "neg"]
    all_nuts = pct_nutrients + energy_nutrients

    nutrients: Dict[str, Dict[str, float]] = {k: {} for k in all_nuts}
    costs: Dict[str, float] = {}
    bounds: Dict[str, Tuple[float, float]] = {}

    for ing, data in INGREDIENTS.items():
        for nut in pct_nutrients:
            nutrients[nut][ing] = data[nut] / 100.0   # fraction per lb DM
        for nut in energy_nutrients:
            nutrients[nut][ing] = data[nut]            # Mcal per lb DM (no conversion)

        costs[ing] = data["cost_usd_per_ton_dm"] / 2000.0  # $/lb DM

        bounds[ing] = (
            data["min_pct"] / 100.0,   # min fraction of diet DM
            data["max_pct"] / 100.0,   # max fraction of diet DM
        )

    return nutrients, costs, bounds


# ---------------------------------------------------------------------------
# DMIR (DMI Required) — NASEM BCNRM preferred approach
# ---------------------------------------------------------------------------

def estimate_dmir(
    bw_lb: float,
    adg_lb: float,
    diet_nem_mcal_per_lb: float = 0.90,
    diet_neg_mcal_per_lb: float = 0.60,
) -> float:
    """Estimate DMI Required to achieve a target ADG (lb/day).

    The DMIR approach avoids the circular dependency inherent in the full
    NASEM DMI equation (DMI depends on diet NE, which depends on intake
    discount, which depends on DMI). Instead it directly calculates:

        DMIR = NEm_req / diet_NEm + RE / diet_NEg

    where RE = 0.0635 × EQSBW^0.75 × ADG_kg^1.097 (retained energy).

    Parameters
    ----------
    bw_lb               Body weight in pounds.
    adg_lb              Target ADG in pounds/day.
    diet_nem_mcal_per_lb  Diet NEm concentration (Mcal/lb DM). Default 0.90.
    diet_neg_mcal_per_lb  Diet NEg concentration (Mcal/lb DM). Default 0.60.

    Returns
    -------
    DMI required in lb/day.
    """
    bw_kg = bw_lb / LB_PER_KG
    adg_kg = adg_lb / LB_PER_KG
    ebw_kg = 0.891 * bw_kg

    nem_req_mcal = 0.077 * (bw_kg ** 0.75)  # maintenance Mcal/day
    re_mcal = 0.0635 * (ebw_kg ** 0.75) * max(adg_kg, 0.01) ** 1.097  # retained energy

    # Avoid division by zero
    nem_conc = max(diet_nem_mcal_per_lb, 0.01)
    neg_conc = max(diet_neg_mcal_per_lb, 0.01)

    dmir_lb = nem_req_mcal / nem_conc + re_mcal / neg_conc
    return round(dmir_lb, 2)


# ---------------------------------------------------------------------------
# Ionophore adjustment (monensin)
# ---------------------------------------------------------------------------

def apply_ionophore_adjustment(
    requirements: Dict[str, float],
    dmi_lb: float,
    use_ionophore: bool = True,
) -> Tuple[Dict[str, float], float]:
    """Apply ionophore (monensin) adjustment to requirements and DMI.

    97.3% of US feedlots use ionophores (Samuelson et al. 2016).
    NASEM 2016 equations are calibrated assuming monensin inclusion.

    Adjustments (NASEM Table 19-6, Duffield et al. 2012 meta-analysis):
      - DMI reduced 3%
      - Diet ME (and thus NEm/NEg) increased 2.3%

    When ionophore is NOT used, DMI should be increased by 3%.

    Parameters
    ----------
    requirements  Dict from compute_requirements().
    dmi_lb        Baseline DMI estimate (lb/day).
    use_ionophore If True (default), apply standard monensin adjustments.

    Returns
    -------
    (adjusted_requirements, adjusted_dmi_lb)
    """
    adj_req = dict(requirements)
    adj_dmi = dmi_lb

    if use_ionophore:
        # Reduce DMI by 3%
        adj_dmi = dmi_lb * 0.97
        # NE requirements can be met with 2.3% less intake because
        # ionophore improves feed efficiency. We lower the absolute
        # Mcal/day thresholds accordingly.
        ne_factor = 0.977  # 1.0 - 0.023
        for key in ("nem_min", "neg_min", "nem_req_mcal", "neg_req_mcal"):
            if key in adj_req and adj_req[key] is not None:
                adj_req[key] = round(adj_req[key] * ne_factor, 3)
    else:
        # Without ionophore, increase DMI by 3% (reverse adjustment)
        adj_dmi = dmi_lb * 1.03

    adj_req["dmi_lb"] = round(adj_dmi, 2)
    return adj_req, adj_dmi

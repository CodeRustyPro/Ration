"""
Confinement Economics — "Hotel Cost" Calculator.

Answers the core producer question: "Should I push this animal harder or
slow the gain and sell when the price is better?"

Computes:
  - Days on feed to reach target market weight
  - Total all-in cost (feed + yardage + interest)
  - Breakeven sale price ($/cwt)
  - Projected profit per head
  - Optimal ADG sweep: which ADG maximizes profit given market forecasts

Industry context:
  - Yardage: $0.50-1.50/hd/day (covers pen, labor, equipment)
  - Interest: 6-9% annual on cattle investment
  - Typical feeding period: 120-180 days
  - Breakeven is the price/cwt you must get to cover all costs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.optimizer import optimize_ration


@dataclass
class ConfinementResult:
    """Economics for a single ADG scenario."""
    adg_lb: float
    days_on_feed: int
    total_feed_cost: float
    total_yardage: float
    total_interest: float
    total_cost: float           # feed + yardage + interest
    cost_per_lb_gain_allin: float  # total_cost / lbs_gained
    purchase_cost: float        # what you paid for the animal
    revenue: float              # target_bw * sale_price
    profit_per_head: float      # revenue - purchase_cost - total_cost
    breakeven_cwt: float        # (purchase + total_cost) / (target_bw/100)
    feed_cost_per_day: float
    target_bw_lb: float
    current_bw_lb: float


def compute_confinement(
    current_bw_lb: float,
    target_bw_lb: float,
    adg_lb: float,
    feed_cost_per_day: float,
    purchase_price_cwt: float,
    sale_price_cwt: float,
    yardage_per_day: float = 0.75,
    interest_rate_pct: float = 7.5,
) -> ConfinementResult:
    """Compute full confinement economics for one ADG scenario.

    Parameters
    ----------
    current_bw_lb       Current body weight (lb).
    target_bw_lb        Target market weight (lb).
    adg_lb              Average daily gain (lb/day).
    feed_cost_per_day   From optimizer ($/head/day).
    purchase_price_cwt  What you paid for this feeder ($/cwt live).
    sale_price_cwt      Expected sale price at market weight ($/cwt live).
    yardage_per_day     Fixed overhead per head per day.
    interest_rate_pct   Annual interest rate on cattle investment (%).
    """
    lbs_to_gain = max(0, target_bw_lb - current_bw_lb)
    days = int(lbs_to_gain / max(adg_lb, 0.1))

    total_feed = feed_cost_per_day * days
    total_yardage = yardage_per_day * days

    purchase_cost = current_bw_lb / 100 * purchase_price_cwt
    interest = purchase_cost * (interest_rate_pct / 100) * (days / 365)

    total_cost = total_feed + total_yardage + interest
    cost_per_lb = total_cost / lbs_to_gain if lbs_to_gain > 0 else 0

    revenue = target_bw_lb / 100 * sale_price_cwt
    profit = revenue - purchase_cost - total_cost

    breakeven = (purchase_cost + total_cost) / (target_bw_lb / 100)

    return ConfinementResult(
        adg_lb=adg_lb,
        days_on_feed=days,
        total_feed_cost=round(total_feed, 2),
        total_yardage=round(total_yardage, 2),
        total_interest=round(interest, 2),
        total_cost=round(total_cost, 2),
        cost_per_lb_gain_allin=round(cost_per_lb, 3),
        purchase_cost=round(purchase_cost, 2),
        revenue=round(revenue, 2),
        profit_per_head=round(profit, 2),
        breakeven_cwt=round(breakeven, 2),
        feed_cost_per_day=round(feed_cost_per_day, 4),
        target_bw_lb=target_bw_lb,
        current_bw_lb=current_bw_lb,
    )


def sweep_adg_scenarios(
    current_bw_lb: float,
    target_bw_lb: float,
    purchase_price_cwt: float,
    sale_price_cwt: float,
    price_overrides: Optional[Dict[str, float]] = None,
    yardage_per_day: float = 0.75,
    interest_rate_pct: float = 7.5,
    use_ionophore: bool = True,
    excluded_ingredients: Optional[Tuple] = None,
    adg_range: Tuple[float, float] = (1.5, 4.5),
    n_points: int = 13,
) -> List[ConfinementResult]:
    """Sweep ADG targets and compute confinement economics for each.

    This answers: "At which ADG do I maximize profit?"
    Lower ADG = cheaper daily diet but more days on feed (more yardage/interest).
    Higher ADG = pricier diet but fewer days in the 'hotel'.
    """
    import numpy as np

    results = []
    for adg in np.linspace(adg_range[0], adg_range[1], n_points):
        adg = round(float(adg), 1)
        opt = optimize_ration(
            current_bw_lb, adg,
            price_overrides=price_overrides,
            use_ionophore=use_ionophore,
            excluded_ingredients=excluded_ingredients,
        )
        if opt is None:
            continue

        r = compute_confinement(
            current_bw_lb, target_bw_lb, adg,
            feed_cost_per_day=opt["total_cost_per_day"],
            purchase_price_cwt=purchase_price_cwt,
            sale_price_cwt=sale_price_cwt,
            yardage_per_day=yardage_per_day,
            interest_rate_pct=interest_rate_pct,
        )
        results.append(r)

    return results


def cumulative_cost_series(
    feed_cost_per_day: float,
    days_on_feed: int,
    yardage_per_day: float,
    purchase_cost: float,
    interest_rate_pct: float,
) -> List[Dict]:
    """Daily cumulative cost breakdown for charting.

    Returns list of {day, feed_cum, yardage_cum, interest_cum, total_cum}.
    """
    rows = []
    for d in range(1, days_on_feed + 1):
        feed = feed_cost_per_day * d
        yard = yardage_per_day * d
        interest = purchase_cost * (interest_rate_pct / 100) * (d / 365)
        rows.append({
            "day": d,
            "feed_cum": round(feed, 2),
            "yardage_cum": round(yard, 2),
            "interest_cum": round(interest, 2),
            "total_cum": round(feed + yard + interest, 2),
        })
    return rows

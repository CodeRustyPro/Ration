"""
Advanced economics and confinement cost modeling.

Calculates realistic feedlot profitability metrics including:
- Death loss risk allocation
- Freight and shipping
- Vet and processing fees
- Pencil and transit shrink
- Accurate interest (cattle vs operating/feed, scaled by equity)
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from src.units import cwt_to_per_lb

@dataclass
class CostResult:
    """Detailed breakdown of costs."""
    purchase_price_cwt: float
    purchase_cost_head: float
    laid_in_cost_head: float
    
    feed_cost_head: float
    yardage_cost_head: float
    # Interest
    cattle_interest: float
    operating_interest: float
    total_interest: float
    
    death_loss_cost: float
    vet_cost: float
    freight_cost: float
    
    total_all_in_cost: float
    
    # Revenue
    sale_weight_lb: float
    shrunk_sale_weight_lb: float
    breakeven_price_cwt: float
    projected_profit_head: float
    cost_of_gain_lb: float

def calculate_costs(
    profile: Dict[str, Any],
    feed_cost_per_day: float,
    days_on_feed: float,
    current_market_price_cwt: float = 190.0,
) -> CostResult:
    """
    Compute comprehensive feedlot economics based on user profile.
    
    Args:
        profile: Dictionary from the onboarding questionnaire containing:
            - operation_type: 'farmer', 'custom', 'backgrounder', etc.
            - start_weight: initial lb
            - target_weight: final lb
            - purchase_price_cwt: purchase price ($/cwt)
            - death_loss_pct: expected percent death loss
            - freight_dt_cwt: freight in $/cwt
            - transit_shrink_pct: shrink during shipping to lot
            - vet_cost: $/hd processing
            - pencil_shrink_pct: sale pencil shrink
            - yardage_cost: $/hd/day
            - interest_rate: annual rate (e.g., 8.5)
            - equity_pct: percent paid with cash
        feed_cost_per_day: daily ration cost
        days_on_feed: days in lot
        current_market_price_cwt: assumed sale price
    """
    # 1. Base weights
    start_wt = profile.get("start_weight", 800)
    target_wt = profile.get("target_weight", 1350)
    
    # 2. Purchase calculation
    purchase_cwt = profile.get("purchase_price_cwt", 245.0)
    # Transit shrink impacts true starting cost because you pay for the shrunk weight
    transit_shrink = profile.get("transit_shrink_pct", 0.0)
    freight_cwt = profile.get("freight_dt_cwt", 0.0)
    
    # True laid in cost includes freight and shrink
    # e.g., you pay for 800 lb but they arrive at 768 lb (4% shrink)
    raw_purchase_head = start_wt * cwt_to_per_lb(purchase_cwt)
    # The steer costs raw_purchase + freight. Shrink effectively increases the $/lb base.
    laid_in_cost = raw_purchase_head + (start_wt * cwt_to_per_lb(freight_cwt))
    
    # 3. Death loss allocation
    # Spread the cost of dead animals across the survivors.
    # Typically 1-2%. If 1.5% die, the 98.5% survivors absorb the cost of the raw laid-in animal.
    dl_pct = profile.get("death_loss_pct", 0.0)
    effective_laid_in = laid_in_cost / (1.0 - (dl_pct / 100.0)) if dl_pct < 100 else laid_in_cost
    death_loss_burden = effective_laid_in - laid_in_cost

    # 4. Operating costs
    vet = profile.get("vet_cost", 0.0)
    yardage_daily = profile.get("yardage_cost", 0.0)
    # Backgrounders typically don't count yardage the same way, but it's captured in the profile
    
    total_feed = feed_cost_per_day * days_on_feed
    total_yardage = yardage_daily * days_on_feed
    
    # 5. Interest (ISU model)
    # Cattle interest is full term.
    # Feed & operating interest averages to half term.
    rate = profile.get("interest_rate", 0.0) / 100.0
    equity = profile.get("equity_pct", 100.0) / 100.0
    financed_pct = max(0.0, 1.0 - equity)
    
    cattle_int = effective_laid_in * financed_pct * rate * (days_on_feed / 365.0)
    # ISU convention: charge operating interest on half the feeding period
    operating_int = (total_feed + total_yardage + vet) * financed_pct * rate * (days_on_feed / 365.0) * 0.5
    total_int = cattle_int + operating_int

    # 6. Total All-In Cost
    all_in = effective_laid_in + total_feed + total_yardage + vet + total_int
    
    # 7. Breakeven and Revenue
    pencil_shrink = profile.get("pencil_shrink_pct", 3.0)
    shrunk_sale_wt = target_wt * (1.0 - (pencil_shrink / 100.0))
    
    breakeven_cwt = (all_in / shrunk_sale_wt) * 100.0
    revenue = shrunk_sale_wt * (current_market_price_cwt / 100.0)
    profit = revenue - all_in
    
    # Cost of Gain (excluding animal purchase, including all other costs)
    # How much did it cost to put on the weight?
    total_gain = target_wt - (start_wt * (1.0 - transit_shrink / 100.0))
    if total_gain > 0:
         # COG = (Total cost - Initial animal cost) / Total Gain
         cog = (all_in - effective_laid_in) / total_gain
    else:
         cog = 0.0

    return CostResult(
        purchase_price_cwt=purchase_cwt,
        purchase_cost_head=raw_purchase_head,
        laid_in_cost_head=laid_in_cost,
        feed_cost_head=total_feed,
        yardage_cost_head=total_yardage,
        cattle_interest=cattle_int,
        operating_interest=operating_int,
        total_interest=total_int,
        death_loss_cost=death_loss_burden,
        vet_cost=vet,
        freight_cost=start_wt * cwt_to_per_lb(freight_cwt),
        total_all_in_cost=all_in,
        sale_weight_lb=target_wt,
        shrunk_sale_weight_lb=shrunk_sale_wt,
        breakeven_price_cwt=breakeven_cwt,
        projected_profit_head=profit,
        cost_of_gain_lb=cog
    )

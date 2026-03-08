"""
Unit conversions and display formatting rules.

Internal (Optimizer/Engine) Units:
- Weight: kg
- Energy: Mcal
- Prices: $/kg DM (dry matter)
- Concentrations: fraction (0.0 - 1.0)
- DMI (Dry Matter Intake): kg/day

External (Display/Farmer-Native) Units:
- Weight: lb (or cwt for prices)
- Prices: $/ton for feed, $/cwt for cattle, $/bu for grain
- DMI: lb/day
- Energy: Mcal/lb
- Concentrations: % (1-100)
"""

# Constants
LB_PER_KG = 2.20462
KG_PER_LB = 0.453592
LB_PER_TON = 2000
CWT_PER_LB = 0.01
CWT_LBS = 100
BU_CORN_LB = 56.0  # standard test weight for corn

def lb_to_kg(lb: float) -> float:
    return lb * KG_PER_LB

def kg_to_lb(kg: float) -> float:
    return kg * LB_PER_KG

def ton_to_lb(ton: float) -> float:
    return ton * LB_PER_TON

def cwt_to_per_lb(price_cwt: float) -> float:
    """$/cwt to $/lb"""
    return price_cwt / CWT_LBS

def per_lb_to_cwt(price_lb: float) -> float:
    """$/lb to $/cwt"""
    return price_lb * CWT_LBS

def per_ton_to_per_lb(price_ton: float) -> float:
    """$/ton to $/lb"""
    return price_ton / LB_PER_TON

def per_lb_to_per_ton(price_lb: float) -> float:
    """$/lb to $/ton"""
    return price_lb * LB_PER_TON

def corn_bu_to_ton(price_bu: float) -> float:
    """$/bushel of corn to $/ton"""
    return (price_bu / BU_CORN_LB) * LB_PER_TON

def as_fed_price_to_dm(as_fed_price_ton: float, dm_pct: float) -> float:
    """Convert $/ton as-fed to $/ton Dry Matter."""
    if dm_pct <= 0:
        return 0.0
    return as_fed_price_ton / (dm_pct / 100.0)

def format_price(value: float, unit: str = "$/ton") -> str:
    """Consistent price formatting for UI."""
    if unit == "$/cwt":
        return f"${value:,.2f}/cwt"
    elif unit == "$/ton":
        return f"${value:,.0f}/ton"
    elif unit == "$/bu":
        return f"${value:.2f}/bu"
    elif unit == "$/hd/day":
        if value < 0:
            return f"-${abs(value):.2f}/hd/day"
        return f"${value:.2f}/hd/day"
    elif unit == "$/lb":
        return f"${value:.3f}/lb"
    return f"${value:.2f}"

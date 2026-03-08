"""
AI Agent for generating farmer-friendly ration explanations.

Uses Anthropic's Claude to translate the LP optimization output, shadow prices, 
and ingredient choices into a simple, 3-sentence "coffee shop" explanation.
"""

import os
from typing import Dict, Any, List
import anthropic

def explain_ration(
    ration: Dict[str, Any],
    shadow_prices: List[Dict[str, Any]],
    target_adg: float,
    body_weight: float,
    cost_per_day: float,
) -> str:
    """
    Generate a concise plain-English explanation of the ration.
    
    Args:
        ration: The optimized ration dictionary.
        shadow_prices: The list of interpreted shadow prices.
        target_adg: Target Average Daily Gain.
        body_weight: Current animal weight.
        cost_per_day: Total daily feed cost.
        
    Returns:
        A Markdown-formatted string containing the explanation.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "⚠️ Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable to enable AI explanations."
    
    # Format the context for the LLM
    ingredients_str = ", ".join([f"{v['pct_of_dmi']:.1f}% {v['display_name'].split(',')[0]}" 
                                 for v in ration.values()])
    
    shadow_str = "\n".join([f"- {s['label']}: Costing {s['impact_per_day']:.3f}/day to meet. Insight: {s['insight']}" 
                            for s in shadow_prices[:3]])
    
    prompt = f"""
    You are an expert beef cattle nutrition consultant speaking directly to a farmer at a coffee shop. 
    Explain the following optimized feed ration in exactly 3 short, punchy sentences.
    
    Context:
    - Animal: {body_weight} lb steer targetting {target_adg} lb/d ADG.
    - Diet Cost: ${cost_per_day:.2f}/head/day.
    - Key Ingredients: {ingredients_str}
    - Biggest Cost Drivers (Shadow Prices):
    {shadow_str}
    
    Guidelines:
    1. Sentence 1: Summarize the main energy/protein strategy of the diet (e.g., "This is a hot corn-based finishing ration...").
    2. Sentence 2: Explain WHY the optimizer chose the ingredients it did, focusing on the cheapest sources of energy/protein.
    3. Sentence 3: Mention the biggest limiting factor (shadow price) and one concrete action the farmer could take to save money.
    
    Keep it extremely grounded, practical, and devoid of AI fluff. Speak like a Midwestern agronomist.
    """

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=150,
            temperature=0.3, # Keep it factual and grounded
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"⚠️ Error generating explanation: {e}"

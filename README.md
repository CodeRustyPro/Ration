# Bio-Economic Feed Ration Optimizer
**UIUC Precision Digital Agriculture Hackathon 2026**

A unified Decision Support System (DSS) that synchronizes NASEM-compliant biological growth modeling with CME Live Cattle Futures to solve the "Marginal Return Decay" problem in Midwest cattle feeding.

## 1. The Problem: The Invisible Margin Gap
In American cattle feeding, biology and economics are treated as disconnected silos. Small-to-mid-sized operations (under 200 head) lack the $5,000/year consulting nutritionists needed to navigate this complexity.

*   **Biological Inefficiency:** As cattle transition from muscle growth to fat deposition, the Cost of Gain (COG) nearly doubles ($5.20/lb at 700lb vs. $8.50/lb at 1,400lb).
*   **The "Static" Ration Trap:** Farmers often feed a single, non-optimized diet regardless of shifting corn prices or ethanol plant byproduct availability.
*   **The Marketing Blindspot:** There is a mathematically precise "Exit Day" where the cost of one more day of feed exceeds the value of the weight gained. Without real-time data integration, farmers lose $40–$80 per head in recoverable value.

## 2. Technical Approach: The "Bio-Economic" Engine
The prototype utilizes a multi-layered computational stack to find the global optimum for both the diet and the sale date.

### A. Linear Programming (LP) Solver
At the core is a custom simplex-based LP solver that processes 20+ simultaneous nutritional constraints in under 100ms.
*   **Constraints:** Crude Protein (CP), Metabolizable Energy (ME), NDF/ADF (Fiber) floors, and strict Mineral Ratios (Ca:P).
*   **Objective Function:** Minimize cost per unit of Dry Matter (DM) while meeting the target Average Daily Gain (ADG).

### B. Live API Integration
The system eliminates "manual entry lag" by pulling real-time data:
*   **USDA MARS API:** Ingests regional wholesale prices for corn, silage, and DDGS.
*   **CME Group API:** Pulls Live Cattle Futures to project revenue curves based on forecasted exit weights.
*   **NASEM/NRC Equations:** Implements the Nutrient Requirements of Beef Cattle (8th Revised Edition) to predict daily intake and metabolic efficiency.

### C. 4-Phase Step-Up Logic
To prevent rumen acidosis, the system auto-generates a 4-phase transition (Receiving → Intermediate 1 → Intermediate 2 → Finisher). Each phase is an independently solved LP solution, transitioning the animal from high-roughage to high-starch diets safely and profitably.

## 3. Key Decision Metrics

| Metric | Output | Significance |
| :--- | :--- | :--- |
| **Projected Profit** | $227.42 / Head | Net of yardage, interest, and death loss. |
| **Optimized Recipe** | "As-Fed" Pounds | Directly usable by the mixer-wagon operator. |
| **Optimal Exit** | Day 247 | The intersection of Marginal Revenue and Marginal Cost. |
| **Sensitivity Analysis** | Tornado Chart | Identifies if Profit is more sensitive to Feed vs. Sale Price. |



# 💰 Week 1 — Bayesian Price Optimizer

**Focus Area:** Reinforcement Learning, Stochastic Processes, Dynamic Pricing
**Algorithm:** Thompson Sampling (Profit-Maximizing Variant)
**Status:** ✅ Completed

---

## 🧐 Problem Overview

Pricing in a live market is a stochastic optimization problem:

* **Unknown Demand Curve:** The customer’s willingness-to-pay is not known.
* **Exploration Cost:** Overpricing kills conversions; underpricing kills revenue.
* **A/B Testing Waste:** Classic A/B tests waste ~50% traffic on inferior prices.

**Objective:**
Develop an online-learning agent that discovers the profit-maximizing price **while selling**, minimizing cumulative *Regret*.

---

## 🧠 Solution: Profit-Aware Thompson Sampling

This project implements a **Bayesian Multi-Armed Bandit** where each price is an arm with an uncertain conversion probability.

### Conversion Model

Each price has a Beta-distributed belief:

[
P(\theta) \sim \text{Beta}(\alpha, \beta)
]

* **α (alpha)** = number of sales
* **β (beta)** = number of no-sales

These evolve with every customer interaction.

### Decision Logic (Profit Maximization)

Standard Thompson Sampling maximizes conversion rate.
This implementation maximizes **Expected Profit**:

1. **Sample**:
   Draw ( p \sim \text{Beta}(\alpha, \beta) ) for each price.

2. **Calculate** expected profit:

[
\text{EP} = p \times (\text{Price} - \text{Cost})
]

3. **Select** the price with highest ( \text{EP} ).
4. **Update** α or β based on sale outcome.

This allows dynamic pricing under uncertainty with Bayesian updating.

---

## 📂 Files in This Project

| File                     | Description                                                                                                                                   |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `bandit_pricing.py`      | Core simulation. Runs the Thompson Sampling agent against a stochastic market (sigmoid demand curve) for ~2,000 iterations.                   |
| `bandit_simulation.html` | Interactive visualization (React, single-file). Animates evolving Beta distributions. Includes an optional “AI Consultant” powered by Gemini. |

---

## 🚀 How to Run

### 1. Python Simulation

Run the pricing bandit simulation:

```bash
python bandit_pricing.py
```

**Outputs:**

* Price-selection scatter plot showing convergence.
* Final Beta PDFs for each price, showing learned beliefs.

---

### 2. Interactive Visualization

Open:

```
bandit_simulation.html
```

Features:

* Live animation of belief updates
* “God Mode”: adjust true customer value to simulate market shocks
* “AI Consultant”: LLM critique of the current strategy

---

## 📊 Key Results & Insights

* **Fast Convergence:** Optimal price (e.g., $40) typically found within **100–200 steps**.
* **Rational Risk Handling:** High-margin but low-probability prices (e.g., $70) are avoided because sampled profit is usually low.
* **Shock Recovery:** When “true value” changes, the agent briefly explores then stabilizes again.
* **Low Regret Curve:** Performance strongly outperforms any static pricing strategy.

---

*This project is part of the Stochastic AI Lab weekly challenge.*

---

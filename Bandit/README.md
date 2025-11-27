💰 Week 1: Bayesian Price Optimizer

Focus Area: Reinforcement Learning, Stochastic Processes, Dynamic Pricing

Algorithm: Thompson Sampling with Profit Maximization

Status: ✅ Completed

🧐 The Problem

In a live market, finding the optimal price for a product is difficult:

Unknown Demand: We don't know what customers are willing to pay (the "True Value").

Exploration Cost: Testing a high price might scare away customers, but testing a low price leaves money on the table.

A/B Testing Flaws: Standard A/B testing wastes 50% of traffic on "losing" prices for weeks.

The Goal: Build an AI agent that learns the optimal price while selling, minimizing "Regret" (lost revenue) by balancing exploration and exploitation.

🧠 The Solution: Multi-Armed Bandit (Thompson Sampling)

Instead of a static model, I implemented a Bayesian Reinforcement Learning agent.

The Math

The agent treats the conversion rate of every price as a Beta Distribution ($\alpha, \beta$) rather than a fixed number.

$\alpha$ (Alpha): Count of Sales.

$\beta$ (Beta): Count of No-Sales.

$$P(\theta) \sim \text{Beta}(\alpha, \beta)$$

The Decision Logic (Profit-Aware)

Unlike standard implementations that maximize Conversion Rate, this agent maximizes Expected Profit.

Sample: Draw a random conversion probability $p$ from the Beta distribution for each price (Thompson Sampling).

Calculate: Multiply by the margin.


$$\text{Expected Profit} = p \times (\text{Price} - \text{Cost})$$

Select: Choose the price with the highest expected profit for this customer.

Update: Observe the outcome (Sale/No Sale) and update $\alpha$ or $\beta$.

📂 Files in this Project

File

Description

bandit_pricing.py

The Core Logic. A Python simulation running the agent against a stochastic market environment (Sigmoid Demand Curve) for 2,000 days.

bandit_simulation.html

The Visualizer. A single-file React application that animates the "Belief Distributions" evolving in real-time. Includes an "AI Consultant" powered by Gemini.

🚀 How to Run

1. Python Simulation

Run the script to see the agent learn over 2,000 iterations.

python bandit_pricing.py


Output: You will see a scatter plot of prices chosen over time (converging to the optimal) and the final Probability Density Functions (PDFs) of the agent's beliefs.

2. Interactive Web Visualization

Simply open bandit_simulation.html in any modern web browser.

Start: Click to watch the agent learn.

God Mode: Adjust "True Customer Value" in real-time to watch the agent react to market shocks.

AI Consultant: Click the "Analyze" button to get an LLM critique of the agent's current strategy.

📊 Results & Insights

Convergence: The agent typically identifies the optimal price (e.g., $40) within ~100-200 iterations.

Risk Aversion: The algorithm correctly avoids the highest price ($70) even if it has a high margin, because the probability of sale (sampled from the Beta distribution) is too low.

Recovery: When market conditions change (e.g., "True Value" drops), the agent briefly explores lower prices before stabilizing again.

Part of the Stochastic AI Lab challenge.

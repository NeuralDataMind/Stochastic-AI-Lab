import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- CONFIGURATION ---
TRUE_VALUE = 45        # The "hidden" maximum price customers tolerate
PRICE_SENSITIVITY = 0.5 # How fast demand drops if price > value
PRODUCT_COST = 30      # It costs us $30 to make the product
DAYS = 2000            # How long we run the simulation

# --- 1. THE HIDDEN MARKET (The Environment) ---
class Market:
    def __init__(self, true_value, sensitivity):
        self.true_value = true_value
        self.sensitivity = sensitivity
    
    def get_customer_decision(self, price):
        """
        Simulates a real customer.
        Returns: 1 (Sale) or 0 (No Sale)
        """
        # Sigmoid Demand Curve
        # If Price < True Value, probability is high.
        # If Price > True Value, probability drops fast.
        prob_buy = 1 / (1 + np.exp(self.sensitivity * (price - self.true_value)))
        
        # Random decision based on that probability
        return 1 if np.random.random() < prob_buy else 0

# --- 2. THE BRAIN (Thompson Sampling Agent) ---
class BanditAgent:
    def __init__(self, possible_prices):
        self.prices = possible_prices
        # Initialize priors: Alpha=1, Beta=1 (Uniform Distribution - "We know nothing")
        self.alphas = np.ones(len(possible_prices))
        self.betas = np.ones(len(possible_prices))
        
    def select_price(self):
        """
        The Core Logic:
        1. Sample from Beta Distribution for each price.
        2. Calculate Expected PROFIT (Price - Cost) * Probability.
        3. Pick the price with highest Expected Profit.
        """
        best_price_idx = -1
        max_expected_profit = -float('inf')
        
        for i in range(len(self.prices)):
            # SAMPLE the probability (The "Wildcard" factor)
            sampled_prob = np.random.beta(self.alphas[i], self.betas[i])
            
            # CALCULATE PROFIT (Your Query #1)
            # We optimize for Profit, not just Revenue
            margin = self.prices[i] - PRODUCT_COST
            expected_profit = sampled_prob * margin
            
            if expected_profit > max_expected_profit:
                max_expected_profit = expected_profit
                best_price_idx = i
                
        return best_price_idx

    def update(self, price_idx, outcome):
        """
        The Learning Step:
        Outcome = 1 (Sale) -> Increase Alpha
        Outcome = 0 (No Sale) -> Increase Beta
        """
        if outcome == 1:
            self.alphas[price_idx] += 1
        else:
            self.betas[price_idx] += 1

# --- 3. THE SIMULATION LOOP ---
def run_simulation():
    # We test prices from $30 to $60
    prices = [30, 35, 40, 45, 50, 55, 60]
    
    market = Market(TRUE_VALUE, PRICE_SENSITIVITY)
    agent = BanditAgent(prices)
    
    history_prices = []
    history_rewards = []
    
    print(f"🚀 Starting Simulation: {DAYS} Days")
    print(f"💰 Product Cost: ${PRODUCT_COST}")
    print("-" * 30)

    for day in range(DAYS):
        # 1. Agent picks a price
        chosen_idx = agent.select_price()
        chosen_price = prices[chosen_idx]
        
        # 2. Market reacts
        sale = market.get_customer_decision(chosen_price)
        
        # 3. Agent learns
        agent.update(chosen_idx, sale)
        
        # 4. Log data
        profit = (chosen_price - PRODUCT_COST) if sale == 1 else 0
        history_prices.append(chosen_price)
        history_rewards.append(profit)
        
        # Optional: Print progress every 500 days
        if (day+1) % 500 == 0:
            print(f"Day {day+1}: Most chosen price recently is ${chosen_price}")

    # --- 4. VISUALIZATION ---
    plot_results(prices, agent, history_prices)

def plot_results(prices, agent, history_prices):
    # Plot 1: Which price did it pick over time?
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_prices, alpha=0.3, marker='o', markersize=2)
    plt.title("Price Selection Over Time")
    plt.xlabel("Customer #")
    plt.ylabel("Price Offered ($)")
    
    # Plot 2: The Final Beliefs (The "Brain")
    plt.subplot(1, 2, 2)
    x = np.linspace(0, 1, 100)
    for i, price in enumerate(prices):
        # Draw the Beta curve for each price
        y = (x**(agent.alphas[i]-1)) * ((1-x)**(agent.betas[i]-1))
        # Normalize for display
        y = y / y.max()
        plt.plot(x, y, label=f"${price}", linewidth=2)
        
    plt.title("Final Belief Distributions (PDF)")
    plt.xlabel("Conversion Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print Final Stats
    print("\n📊 Final Logic State:")
    for i, p in enumerate(prices):
        mean_prob = agent.alphas[i] / (agent.alphas[i] + agent.betas[i])
        print(f"Price ${p}: Est. Conv Rate: {mean_prob:.2f} | Samples: {int(agent.alphas[i]+agent.betas[i])}")

if __name__ == "__main__":
    run_simulation()
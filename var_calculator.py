import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt


# Get historical stock data
stock = input("Enter stock ticker: ")
data = yf.download(stock, start='2020-01-01', end='2025-01-01')

# Print columns to check what's available
print(data.columns)  # This will print the column names of the downloaded data

# Use 'Close' if 'Adj Close' is not available
returns = data['Close'].pct_change().dropna()

# Calculate Historical VaR
confidence_level = 0.95
var_historical = np.percentile(returns, (1 - confidence_level) * 100)
print(f"Historical VaR at {confidence_level * 100}% confidence: {var_historical:.2%}")

# Monte Carlo Simulation VaR
simulations = 10000
mean, std_dev = returns.mean(), returns.std()
simulated_returns = np.random.normal(mean, std_dev, simulations)
var_monte_carlo = np.percentile(simulated_returns, (1 - confidence_level) * 100)
print(f"Monte Carlo VaR at {confidence_level * 100}% confidence: {var_monte_carlo:.2%}")

# Plot
plt.hist(simulated_returns, bins=50, edgecolor='black')
plt.axvline(var_monte_carlo, color='red', linestyle='dashed', linewidth=2)
plt.title('Monte Carlo Simulated Returns')
plt.show()

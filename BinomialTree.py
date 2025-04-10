"""
Author: Sai Navyanth
SID: 
"""


import numpy as np

def american_option_pricing(S0, K, r, T, sigma, N, option_type):
    """
    Prices an American option using the binomial tree method.

    Args:
        S0 (float): The spot price of the asset.
        K (float): The strike price.
        r (float): The risk-free interest rate.
        T (float): The time to maturity (in years).
        sigma (float): The volatility of the asset.
        N (int): The number of steps in the binomial tree.
        option_type (str): The type of option ('call' or 'put').

    Returns:
        tuple: (option_price, stock_prices, option_prices)
            - option_price: The price of the American option
            - stock_prices: The binomial tree of stock prices
            - option_prices: The binomial tree of option prices
    """

    # Calculate the up and down factors
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u

    # Calculate the risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialize the stock price tree
    stock_prices = np.zeros((N + 1, N + 1))
    stock_prices[0, 0] = S0
    
    for i in range(1, N + 1):
        stock_prices[i, 0] = stock_prices[i - 1, 0] * u
        for j in range(1, i + 1):
            stock_prices[i, j] = stock_prices[i - 1, j - 1] * d

    # Initialize the option price tree
    option_prices = np.zeros((N + 1, N + 1))
    
    # Calculate terminal option values
    for j in range(N + 1):
        if option_type == 'call':
            option_prices[N, j] = max(0, stock_prices[N, j] - K)
        else:  # put option
            option_prices[N, j] = max(0, K - stock_prices[N, j])

    # Work backwards through the tree
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            # Calculate the discounted expected value
            discounted_expected_value = np.exp(-r * dt) * (p * option_prices[i + 1, j] + (1 - p) * option_prices[i + 1, j + 1])
            
            # Calculate the intrinsic value
            if option_type == 'call':
                intrinsic_value = max(0, stock_prices[i, j] - K)
            else:  # put option
                intrinsic_value = max(0, K - stock_prices[i, j])
            
            # Take the maximum of the discounted expected value and intrinsic value
            option_prices[i, j] = max(discounted_expected_value, intrinsic_value)

    return option_prices[0, 0], stock_prices, option_prices


print(american_option_pricing(50, 40, 0.1, 2, 0.4, 200, 'put'))
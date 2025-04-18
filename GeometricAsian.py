"""
Author: Sai Navyanth
SID: 3036384139
"""

import numpy as np
from scipy.stats import norm


def geometric_asian_option_price(S0, sigma, r, T, K, n, option_type):
    """
    Calculates the price of a geometric Asian option.
    Args:
        S0 (float): The initial stock price.
        sigma (float): The volatility of the stock price.
        r (float): The risk-free interest rate.
        T (float): The time to maturity (in years).
        K (float): The strike price.
        n (int): The number of observation times for the geometric average.
    """

    dt = T / n
    sigma_cap = sigma * np.sqrt((n+1) * (2 * n + 1) / (6 * n ** 2))
    mu_cap = 0.5 * sigma_cap ** 2 + (r - 0.5 * sigma ** 2) * (n + 1) / (2 * n)
    d1 = (np.log(S0 / K) + (mu_cap + 0.5 * sigma_cap ** 2) * T) / (sigma_cap * np.sqrt(T))
    d2 = d1 - sigma_cap * np.sqrt(T)
    if option_type == 'call':
        return np.exp(-r * T) * (S0 * np.exp(mu_cap * T) * norm.cdf(d1) - K * norm.cdf(d2))
    elif option_type == 'put':
        return np.exp(-r * T) * (K * norm.cdf(-d2) - S0 * np.exp(mu_cap * T) * norm.cdf(-d1))
    else:
        raise ValueError("Invalid option type.")
    

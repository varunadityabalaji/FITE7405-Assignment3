import numpy as np
from scipy.stats import norm

print("Geometric Basket Asian Option Price: ")
def geometric_asian_option_price(S0, sigma, r, T, K, n, option_type):
    """
    Calculates the price of a geometric Asian option.

    Args:
        S0 (float): The initial stock price.
        sigma (float): The volatility of the stock price.
        r (float): The risk-free interest rate.
        T (float): The time to maturity.
        K (float): The strike price.
        n (int): The number of observation times for the geometric average.
        option_type (str): 'call' or 'put'

    Returns:
        float: The price of the geometric Asian option.
    """
    dt = T / n
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S0 * np.exp(-r * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-r * T) * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type.")
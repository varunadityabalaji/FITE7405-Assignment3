"""
Author: Sai Navyanth
SID: 
"""

import numpy as np
from scipy.stats import norm


def geometric_basket_option_price(S1_0, S2_0, sigma1, sigma2, r, T, K, rho, type):
    """
    Calculates the price of a geometric basket Asian option using the closed-form solution.

    Parameters:
    S1_0 (float): Initial price of the first underlying asset.
    S2_0 (float): Initial price of the second underlying asset.
    sigma1 (float): Volatility of the first underlying asset.
    sigma2 (float): Volatility of the second underlying asset.
    r (float): Risk-free interest rate.
    T (float): Time to maturity (in years).
    K (float): Strike price of the option.
    rho (float): Correlation coefficient between the two assets.
    type (str): Type of the option ('call' or 'put').

    Returns:
    float: Price of the geometric basket Asian option.
    """


    # Calculate the average volatility
    sigma_avg = np.sqrt(sigma1**2 + sigma2**2 + 2*(sigma1*sigma2)*rho)/2

    # Calculate the average discount factor
    df = np.exp(-r * T)

    # Calculate the geometric average of the initial prices
    B = np.sqrt(S1_0 * S2_0)

  # calculate the average drift
    mew = r - 0.25*(sigma1**2 + sigma2**2) + 0.5*sigma_avg**2

    # Calculate the d1 and d2 terms
    d1 = (np.log(B / K) + (mew + 0.5 * sigma_avg**2) * T) / (sigma_avg * np.sqrt(T))
    d2 = d1 - sigma_avg * np.sqrt(T)

    # Calculate the option price based on the type
    if type == 'call':
        price = df * (B * np.exp(mew * T) * norm.cdf(d1) - K * norm.cdf(d2))
    elif type == 'put':
        price = df * (K * norm.cdf(-d2) - B * np.exp(mew * T) * norm.cdf(-d1))
    else:
        raise ValueError('Invalid option type')

    return price




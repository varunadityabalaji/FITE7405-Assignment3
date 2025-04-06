import numpy as np
from scipy.stats import norm

print("Geometric Basket Asian Option Price: ")
def geometric_basket_asian_option_price(S1_0, S2_0, sigma1, sigma2, r, T, K, rho, type):
    """
    Calculates the price of a geometric basket Asian option using the closed-form solution.
    """
    # Calculate the average volatility
    sigma_avg = np.sqrt(sigma1**2 + sigma2**2 + 2*(sigma1*sigma2)*rho)/2

    # Calculate the average discount factor
    df = np.exp(-r * T)

    # Calculate the geometric average of the initial prices
    B = np.sqrt(S1_0 * S2_0)

  # calculate the upper updated interest rate term  
    u_b = r - 0.25*(sigma1**2 + sigma2**2) + 0.5*sigma_avg**2

    # Calculate the d1 and d2 terms
    d1 = (np.log(B / K) + (u_b + 0.5 * sigma_avg**2) * T) / (sigma_avg * np.sqrt(T))
    d2 = d1 - sigma_avg * np.sqrt(T)

    # Calculate the option price based on the type
    if type == 'call':
        price = df * (B * np.exp(u_b * T) * norm.cdf(d1) - K * norm.cdf(d2))
    elif type == 'put':
        price = df * (K * norm.cdf(-d2) - B * np.exp(u_b * T) * norm.cdf(-d1))
    else:
        raise ValueError('Invalid option type')

    return price

print(geometric_basket_asian_option_price(100, 100, 0.2, 0.2, 0.05, 1, 100, 0.5, 'call'))


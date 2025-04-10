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
        float: The price of the American option.
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
        for j in range(i + 1):
            stock_prices[i, j] = stock_prices[i - 1, j] * u
            stock_prices[i, j + 1] = stock_prices[i - 1, j] * d

    # Initialize the option price tree
    option_prices = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        if option_type == 'call':
            option_prices[N, i] = max(0, stock_prices[N, i] - K)
        elif option_type == 'put':
            option_prices[N, i] = max(0, K - stock_prices[N, i])

    # Work backwards through the tree
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            # Calculate the discounted expected value of the option at the next time step
            discounted_expected_value = np.exp(-r * dt) * (p * option_prices[i + 1, j] + (1 - p) * option_prices[i + 1, j + 1])

            # Calculate the intrinsic value of the option
            if option_type == 'call':
                intrinsic_value = max(0, stock_prices[i, j] - K)
            else:
                intrinsic_value = max(0, K - stock_prices[i, j])

            # Take the maximum of the discounted expected value and the intrinsic value
            option_prices[i, j] = max(discounted_expected_value, intrinsic_value)

    # Return the price of the option
    return option_prices[0, 0]

if __name__ == "__main__":
    # Example parameters
    S0 = 100  # Spot price
    K = 100   # Strike price
    r = 0.05  # Risk-free interest rate
    T = 1     # Time to maturity (in years)
    sigma = 0.2  # Volatility
    N = 100   # Number of steps in the binomial tree
    option_type = 'call'  # Option type ('call' or 'put')

    # Calculate the price of the American option
    price = american_option_pricing(S0, K, r, T, sigma, N, option_type)
    print(f"The price of the American {option_type} option is: {price:.2f}")
"""
Author: Varun Balaji
SID: 3036383355
"""

import numpy as np
from scipy.stats import norm
from typing import List, Tuple, Dict

class AsianOptionPricer:
    """
    AsianOptionPricer is a class for pricing Asian options using Monte Carlo simulations 
    and control variates. It supports both arithmetic and geometric Asian options for 
    'call' and 'put' types.

    Attributes:
        S (float): Initial stock price.
        k (float): Strike price.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility of the stock.
        N (int): Number of time steps.
        M (int): Number of Monte Carlo simulations.
        option_type (str): Option type ('call' or 'put').

    Methods:
        geometric_asian_exact(): Computes the exact price of a geometric Asian option.
        simulate_paths(): Simulates stock price paths using Monte Carlo.
        compute_payoffs(paths): Computes payoffs for arithmetic and geometric Asian options.
        monte_carlo_pricing(): Estimates the option price using Monte Carlo and control variates.
    """
    def __init__(self, S: float, k: float, T: float, r: float, sigma: float, N: int, M: int, option_type: str, control_variate: bool = True) -> None:
        try:
            if S <= 0 or k <= 0 or T <= 0 or sigma <= 0 or N <= 0 or M <= 0:
                raise ValueError("All input parameters must be positive.")
            if not (0 <= r <= 1):
                raise ValueError("Risk-free rate (r) must be between 0 and 1.")
            if option_type not in ["call", "put"]:
                raise ValueError("Invalid option type. Must be 'call' or 'put'.")
            self.S = S          # Initial stock price
            self.k = k          # Strike price
            self.T = T          # Time to maturity (in years)
            self.r = r          # Risk-free interest rate
            self.sigma = sigma  # Volatility of the stock
            self.N = N          # Number of time steps 
            self.M = M          # Number of Monte Carlo simulations
            self.dt = T / N     # Time step size 
            self.option_type = option_type  # Option type
            self.control_variate = control_variate  # Default to using control variate
        except ValueError as e:
            raise ValueError(f"Initialization error: {e}")

    def __repr__(self) -> str:
        return (f"AsianOptionPricer(S={self.S}, k={self.k}, T={self.T}, r={self.r}, "
                f"sigma={self.sigma}, N={self.N}, M={self.M}, option_type={self.option_type})")

    def __str__(self) -> str:
        return (f"Asian Option Pricer with initial stock price {self.S}, strike price {self.k}, "
                f"time to maturity {self.T} years, risk-free rate {self.r}, volatility {self.sigma}, "
                f"{self.N} time steps, {self.M} Monte Carlo simulations, and option type '{self.option_type}'.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AsianOptionPricer):
            return NotImplemented
        return (self.S == other.S and self.k == other.k and self.T == other.T and
                self.r == other.r and self.sigma == other.sigma and
                self.N == other.N and self.M == other.M and self.option_type == other.option_type)

    def __hash__(self) -> int:
        return hash((self.S, self.k, self.T, self.r, self.sigma, self.N, self.M, self.option_type))

    def geometric_asian_exact(self) -> float:
        try:
            sigsqT = self.sigma**2 * self.T * (self.N + 1) * (2 * self.N + 1) / (6 * self.N**2)
            muT = 0.5 * sigsqT + (self.r - 0.5 * self.sigma**2) * self.T * (self.N + 1) / (2 * self.N)
            d1 = (np.log(self.S / self.k) + (muT + 0.5 * sigsqT)) / np.sqrt(sigsqT)
            d2 = d1 - np.sqrt(sigsqT)
            N1 = norm.cdf(d1)
            N2 = norm.cdf(d2)
            if self.option_type == "call":
                geo_price = np.exp(-self.r * self.T) * (self.S * np.exp(muT) * N1 - self.k * N2)
            elif self.option_type == "put":
                N1 = norm.cdf(-d1)
                N2 = norm.cdf(-d2)
                geo_price = np.exp(-self.r * self.T) * (self.k * N2 - self.S * np.exp(muT) * N1)
            return geo_price
        except Exception as e:
            raise RuntimeError(f"Error in geometric_asian_exact: {e}")

    def simulate_paths(self) -> np.ndarray:
        try:
            np.random.seed(42)  # Set the seed for reproducibility
            all_paths = np.zeros((self.M, self.N))  # rows is simulations, columns is time steps
            for i in range(self.M):
                path = [self.S]
                for j in range(1, self.N):
                    drift = (self.r - 0.5 * self.sigma**2) * self.dt
                    diffusion = self.sigma * np.sqrt(self.dt) * np.random.randn()
                    next_price = path[-1] * np.exp(drift + diffusion)
                    path.append(next_price)
                all_paths[i] = path
            return all_paths
        except Exception as e:
            raise RuntimeError(f"Error in simulate_paths: {e}")

    def compute_payoffs(self, paths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            arith_payoffs = []
            geo_payoffs = []
            for path in paths:
                arith_mean = np.mean(path)
                geo_mean = np.exp(np.mean(np.log(path)))
                if self.option_type == "call":
                    arith_payoff = np.exp(-self.r * self.T) * max(arith_mean - self.k, 0)
                    geo_payoff = np.exp(-self.r * self.T) * max(geo_mean - self.k, 0)
                elif self.option_type == "put":
                    arith_payoff = np.exp(-self.r * self.T) * max(self.k - arith_mean, 0)
                    geo_payoff = np.exp(-self.r * self.T) * max(self.k - geo_mean, 0)
                arith_payoffs.append(arith_payoff)
                geo_payoffs.append(geo_payoff)
            return np.array(arith_payoffs), np.array(geo_payoffs)
        except Exception as e:
            raise RuntimeError(f"Error in compute_payoffs: {e}")

    def monte_carlo_pricing(self) -> Dict[str, List[float]]:
        geo_exact = self.geometric_asian_exact()
        paths = self.simulate_paths()
        arith_payoffs, geo_payoffs = self.compute_payoffs(paths)

        if self.control_variate:
            # Calculate theta
            covXY = np.cov(arith_payoffs, geo_payoffs)[0, 1]
            theta = covXY / np.var(geo_payoffs)

            # Control Variate
            Z = arith_payoffs + theta * (geo_exact - geo_payoffs)
            Zmean = np.mean(Z)
            Zstd = np.std(Z)
            confcv = [Zmean - 1.96 * Zstd / np.sqrt(self.M),
                      Zmean + 1.96 * Zstd / np.sqrt(self.M)]

            return {
                'cv_estimate': str(Zmean),
                'cv_ci': [str(confcv[0]), str(confcv[1])]
            }
        
        else:
            # Without control variate
            arith_mean = np.mean(arith_payoffs)
            arith_std = np.std(arith_payoffs)
            conf = [arith_mean - 1.96 * arith_std / np.sqrt(self.M),
                    arith_mean + 1.96 * arith_std / np.sqrt(self.M)]

            return {
                'cv_estimate': str(arith_mean),
                'cv_ci': [str(conf[0]), str(conf[1])]
            }

if __name__ == "__main__":
    pricer = AsianOptionPricer(S = 100, k = 100, sigma = 0.2, r = 0.05, T = 1, N=50, M=100000, option_type='put',control_variate=True)
    results = pricer.monte_carlo_pricing()
    for key, value in results.items():
        print(f"{key}: {value}")
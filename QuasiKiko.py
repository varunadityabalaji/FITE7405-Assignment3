"""
Author: Varun Balaji
SID: 3036383355
"""

import numpy as np
from scipy.stats import norm, qmc
from typing import List, Dict, Union
import numpy.typing as npt

class KIKOOptionPricerQMC:
    """
    KIKOOptionPricerQMC is a class for pricing Knock-In Knock-Out (KIKO) options using 
    Quasi-Monte Carlo (QMC) simulation with Sobol sequences.

    Attributes:
        S (float): Initial stock price.
        K (float): Strike price of the option.
        T (float): Time to maturity (in years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility of the underlying asset.
        L (float): Knock-in barrier (lower barrier).
        U (float): Knock-out barrier (upper barrier).
        N (int): Number of time steps in the simulation.
        M (int): Number of simulated paths (default is 100,000).
        option_type (str): Type of option ('put' or 'call').
        dt (float): Time step size, calculated as T / N.

    Methods:
        __repr__(): Returns a string representation of the object.
        simulate_paths(S0, Z): Simulates asset price paths using QMC-generated random numbers.
        get_payoffs(paths): Computes the payoffs for the simulated paths based on the option type.
        delta(h): Calculates the delta (sensitivity of the option price to changes in the underlying asset price).
        monte_carlo_pricing(): Prices the KIKO option and calculates its delta using QMC simulation.
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, L: float, U: float, N: int, option_type: str , M: int = 100000) -> None:
        try:
            if option_type not in ['put', 'call']:
                raise ValueError("option_type must be either 'put' or 'call'")
            self.S = S
            self.K = K
            self.T = T
            self.r = r
            self.sigma = sigma
            self.L = L
            self.U = U
            self.M = M
            self.N = N
            self.dt = T / N
            self.option_type = option_type
        except Exception as e:
            raise RuntimeError(f"Error in __init__: {e}")

    def __repr__(self) -> str:
        return (f"KIKOOptionPricerQMC(S={self.S}, K={self.K}, T={self.T}, r={self.r}, sigma={self.sigma}, "
                f"L={self.L}, U={self.U}, M={self.M}, N={self.N}, option_type='{self.option_type}')")

    def simulate_paths(self, S0: float, Z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        try:
            paths = np.zeros((self.M, self.N))
            paths[:, 0] = S0
            for t in range(1, self.N):
                drift = (self.r - 0.5 * self.sigma ** 2) * self.dt
                diffusion = self.sigma * np.sqrt(self.dt) * Z[:, t]
                paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion)
            return paths
        except Exception as e:
            raise RuntimeError(f"Error in simulate_paths: {e}")

    def get_payoffs(self, paths: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        try:
            payoffs = []
            for path in paths:
                min_price = np.min(path)
                max_price = np.max(path)

                knocked_out = max_price >= self.U
                knocked_in = min_price <= self.L

                if self.option_type == 'put':
                    if knocked_in and not knocked_out:
                        payoff = max(self.K - path[-1], 0)
                    else:
                        payoff = 0
                elif self.option_type == 'call':
                    if knocked_in and not knocked_out:
                        payoff = max(path[-1] - self.K, 0)
                    else:
                        payoff = 0
                payoffs.append(payoff)
            return np.array(payoffs)
        except Exception as e:
            raise RuntimeError(f"Error in get_payoffs: {e}")

    def delta(self, h: float = 1.0) -> float:
        try:
            sobol = qmc.Sobol(d=self.N, scramble=True)
            U = sobol.random(n=self.M)
            Z = norm.ppf(U)

            # Simulate for S+h
            paths_up = self.simulate_paths(self.S + h, Z)
            payoffs_up = self.get_payoffs(paths_up)
            EV_up = np.mean(payoffs_up)

            # Simulate for S
            paths = self.simulate_paths(self.S, Z)
            payoffs = self.get_payoffs(paths)
            EV = np.mean(payoffs)

            delta = np.exp(-self.r * self.T) * (EV_up - EV) / h
            return delta
        except Exception as e:
            raise RuntimeError(f"Error in delta calculation: {e}")

    def monte_carlo_pricing(self) -> Dict[str, str]:
        try:
            sobol = qmc.Sobol(d=self.N, scramble=True)
            U = sobol.random(n=self.M)
            Z = norm.ppf(U)
            paths = self.simulate_paths(self.S, Z)
            payoffs = self.get_payoffs(paths)
            price = np.exp(-self.r * self.T) * np.mean(payoffs)

            # Calculate delta
            delta = self.delta()

            # Return results as a dictionary with string values
            return {
                "price": str(price),
                "delta": str(delta)
            }
        except Exception as e:
            raise RuntimeError(f"Error in monte_carlo_pricing: {e}")


if __name__ == "__main__":
    # Example usage
    try:
        pricer = KIKOOptionPricerQMC(S=100, K=100, T=1, r=0.05, sigma=0.2, L=90, U=110, M=10000, N=252, option_type='call')
        result = pricer.monte_carlo_pricing()
        print(f"Pricing Result: {result}")
    except Exception as e:
        print(f"An error occurred: {e}")
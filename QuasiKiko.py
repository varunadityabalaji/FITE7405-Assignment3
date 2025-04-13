"""
Author: Varun Balaji
SID: 3036383355
"""

import numpy as np
from scipy.stats import norm, qmc
from typing import Dict, Tuple, Union
import numpy.typing as npt

class KIKOOptionPricerQMC:
    def __init__(
        self, S: float, K: float, T: float, r: float, sigma: float, L: float, U: float, 
        N: int, option_type: str, rebate: float = 0.0, M: int = 100000,
    ) -> None:
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
        self.rebate = rebate
        self.seed = 49  # Default seed for reproducibility

    def __repr__(self) -> str:
        return (f"KIKOOptionPricerQMC(S={self.S}, K={self.K}, T={self.T}, r={self.r}, sigma={self.sigma}, "
                f"L={self.L}, U={self.U}, M={self.M}, N={self.N}, option_type='{self.option_type}', rebate={self.rebate}, seed={self.seed})")

    def simulate_paths(self, S0: float, Z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        paths = np.zeros((self.M, self.N + 1)) 
        paths[:, 0] = S0
        for t in range(1, self.N + 1): 
            drift = (self.r - 0.5 * self.sigma ** 2) * self.dt
            diffusion = self.sigma * np.sqrt(self.dt) * Z[:, t-1] 
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion)
        return paths[:, 1:]  

    def get_payoffs(self, paths: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        payoffs = []
        for path in paths:
            min_price = np.min(path)
            max_price = np.max(path)

            knocked_out = max_price >= self.U
            knocked_in = min_price <= self.L

            if knocked_out:
                knock_out_indices = np.where(path >= self.U)[0]
                if knock_out_indices.size > 0:
                    knock_out_time = knock_out_indices[0] + 1  # 1-based index
                    discount_factor = np.exp(-self.r * knock_out_time * self.dt)
                    payoff = self.rebate * discount_factor
            elif not knocked_in:
                payoff = 0  
            else:
                if self.option_type == 'put':
                    payoff = max(self.K - path[-1], 0)
                elif self.option_type == 'call':
                    payoff = max(path[-1] - self.K, 0)
                discount_factor = np.exp(-self.r * self.T)
                payoff *= discount_factor
            payoffs.append(payoff)
        return np.array(payoffs)

    def delta(self, Z: npt.NDArray[np.float64], h: float = 0.01) -> float:
        paths_up = self.simulate_paths(self.S + h, Z)
        payoffs_up = self.get_payoffs(paths_up)
        EV_up = np.mean(payoffs_up)

        paths = self.simulate_paths(self.S, Z)
        payoffs = self.get_payoffs(paths)
        EV = np.mean(payoffs)

        delta = np.exp(-self.r * self.T) * (EV_up - EV) / h
        return delta

    def monte_carlo_pricing(self) -> Dict[str, Union[str, Tuple[float, float]]]:
        sobol = qmc.Sobol(d=self.N, scramble=True, seed=self.seed)
        U = sobol.random(n=self.M)
        Z = norm.ppf(U)

        paths = self.simulate_paths(self.S, Z)
        payoffs = self.get_payoffs(paths)
        
        price = np.mean(payoffs)
        std_error = np.std(payoffs) / np.sqrt(self.M)

        delta = self.delta(Z)

        return {
            "price": f"{price:.4f}",
            "confidence_interval": (round(price - 1.96 * std_error, 4), round(price + 1.96 * std_error, 4)),
            "delta": f"{delta:.4f}"
        }

if __name__ == "__main__":

    # Example 1: Varying the volatility
    print("Effect of Volatility:")
    for sigma in [0.1, 0.2, 0.3, 0.4, 0.5]:
        pricer = KIKOOptionPricerQMC(S=100, K=100, T=3, r=0.05, sigma=sigma, L=90, U=110, N=50, M=100000, option_type='put')
        result = pricer.monte_carlo_pricing()
        print(f"Volatility: {sigma}, Price: {result['price']}, Confidence Interval: {result['confidence_interval']}, Delta: {result['delta']}")

    # Example 2: Varying the risk-free rate
    print("\nEffect of Risk-Free Rate:")
    for r in [0.01, 0.03, 0.05, 0.07, 0.1]:
        pricer = KIKOOptionPricerQMC(S=100, K=100, T=3, r=r, sigma=0.3, L=90, U=110, N=50, M=100000, option_type='put')
        result = pricer.monte_carlo_pricing()
        print(f"Risk-Free Rate: {r}, Price: {result['price']}, Confidence Interval: {result['confidence_interval']}, Delta: {result['delta']}")

    # Example 3: Varying the time to maturity
    print("\nEffect of Time to Maturity:")
    for T in [1, 2, 3, 4, 5]:
        pricer = KIKOOptionPricerQMC(S=100, K=100, T=T, r=0.05, sigma=0.3, L=90, U=110, N=50, M=100000, option_type='put')
        result = pricer.monte_carlo_pricing()
        print(f"Time to Maturity: {T}, Price: {result['price']}, Confidence Interval: {result['confidence_interval']}, Delta: {result['delta']}")

    # Example 4: Varying the number of time steps
    print("\nEffect of Number of Time Steps:")
    for N in [10, 20, 50, 100, 200]:
        pricer = KIKOOptionPricerQMC(S=100, K=100, T=3, r=0.05, sigma=0.3, L=90, U=110, N=N, M=100000, option_type='put')
        result = pricer.monte_carlo_pricing()
        print(f"Number of Time Steps: {N}, Price: {result['price']}, Confidence Interval: {result['confidence_interval']}, Delta: {result['delta']}")

    # Example 5: Varying the number of simulations
    print("\nEffect of Number of Simulations:")
    for M in [1000, 10000, 50000, 100000, 200000]:
        pricer = KIKOOptionPricerQMC(S=100, K=100, T=3, r=0.05, sigma=0.3, L=90, U=110, N=50, M=M, option_type='put')
        result = pricer.monte_carlo_pricing()
        print(f"Number of Simulations: {M}, Price: {result['price']}, Confidence Interval: {result['confidence_interval']}, Delta: {result['delta']}")

    # Example 6: Varying the strike price
    print("\nEffect of Strike Price:")
    for K in [80, 90, 100, 110, 120]:
        pricer = KIKOOptionPricerQMC(S=100, K=K, T=3, r=0.05, sigma=0.3, L=90, U=110, N=50, M=100000, option_type='put')
        result = pricer.monte_carlo_pricing()
        print(f"Strike Price: {K}, Price: {result['price']}, Confidence Interval: {result['confidence_interval']}, Delta: {result['delta']}")
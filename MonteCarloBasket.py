"""
Author: Varun Balaji
SID: 3036383355
"""

import numpy as np
from scipy.stats import norm
from typing import Literal, Tuple, Dict, List

class BasketOptionPricer:
    """
    A class to price basket options using Monte Carlo simulation.

    Attributes:
        S1, S2 (float): Initial prices of the two assets.
        K (float): Strike price.
        T (float): Time to maturity.
        r (float): Risk-free rate.
        sigma1, sigma2 (float): Volatilities of the two assets.
        rho (float): Correlation between the assets.
        M (int): Number of simulations.
        option_type (Literal["call", "put"]): Option type.
        use_control_variate (bool): Use control variate for variance reduction.

    Methods:
        monte_carlo_pricing(): Prices the basket option and returns estimates and confidence intervals.
    """
    
    def __init__(
        self, 
        S1: float, 
        S2: float, 
        K: float, 
        T: float, 
        r: float, 
        sigma1: float, 
        sigma2: float, 
        rho: float, 
        M: int, 
        option_type: Literal["call", "put"],
        use_control_variate: bool = True  
    ) -> None:
        try:
            if option_type not in ["call", "put"]:
                raise ValueError("option_type must be 'call' or 'put'")

            self.S1 = S1          # Initial price of asset 1
            self.S2 = S2          # Initial price of asset 2
            self.K = K            # Strike price
            self.T = T            # Time to maturity
            self.r = r            # Risk-free rate
            self.sigma1 = sigma1  # Volatility of asset 1
            self.sigma2 = sigma2  # Volatility of asset 2
            self.rho = rho        # Correlation between assets
            self.M = M            # Number of simulations
            self.option_type = option_type
            self.use_control_variate = use_control_variate  # Toggle for control variate
        except Exception as e:
            raise ValueError(f"Error initializing BasketOptionPricer: {e}")

    def __repr__(self) -> str:
        try:
            return (
                f"BasketOptionPricer(S1={self.S1}, S2={self.S2}, K={self.K}, T={self.T}, "
                f"r={self.r}, sigma1={self.sigma1}, sigma2={self.sigma2}, rho={self.rho}, "
                f"M={self.M}, option_type='{self.option_type}', use_control_variate={self.use_control_variate})"
            )
        except Exception as e:
            return f"Error in __repr__: {e}"
        
    def simulate_paths(self) -> Tuple[np.ndarray, np.ndarray]:
        try:
            np.random.seed(95)  # For reproducibility
            
            #single timestep
            dt = self.T  
            
            Z1 = np.random.normal(0, 1, self.M)
            Z2 = np.random.normal(0, 1, self.M)
            
            #from assignment 2
            Z2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2  # Correlated Z2

            ST1 = self.S1 * np.exp((self.r - 0.5 * self.sigma1**2) * dt + self.sigma1 * np.sqrt(dt) * Z1)
            ST2 = self.S2 * np.exp((self.r - 0.5 * self.sigma2**2) * dt + self.sigma2 * np.sqrt(dt) * Z2)

            return ST1, ST2
        except Exception as e:
            raise ValueError(f"Error in simulate_paths: {e}")
    
    def geometric_basket_price(self) -> float:
        try:
            sigma_avg = np.sqrt(self.sigma1**2 + self.sigma2**2 + 2*(self.sigma1*self.sigma2)*self.rho)/2
            df = np.exp(-self.r * self.T)

            # Calculate the geometric average of the initial prices
            B = np.sqrt(self.S1 * self.S2)

        # calculate the average drift
            mew = self.r - 0.25*(self.sigma1**2 + self.sigma2**2) + 0.5*sigma_avg**2

            # Calculate the d1 and d2 terms
            d1 = (np.log(B / self.K) + (mew + 0.5 * sigma_avg**2) * self.T) / (sigma_avg * np.sqrt(self.T))
            d2 = d1 - sigma_avg * np.sqrt(self.T)

            # Calculate the option price based on the type
            if self.option_type == 'call':
                price = df * (B * np.exp(mew * self.T) * norm.cdf(d1) - self.K * norm.cdf(d2))
            elif self.option_type == 'put':
                price = df * (self.K * norm.cdf(-d2) - B * np.exp(mew * self.T) * norm.cdf(-d1))
            return price
        except Exception as e:
            raise ValueError(f"Error in geometric_basket_price: {e}")

    def monte_carlo_pricing(self) -> Dict[str, List[float]]:
        try:
            ST1, ST2 = self.simulate_paths()

            arith_mean = 0.5 * (ST1 + ST2)

            if self.option_type == 'call':
                arith_payoff = np.exp(-self.r * self.T) * np.maximum(arith_mean - self.K, 0)
            else:
                arith_payoff = np.exp(-self.r * self.T) * np.maximum(self.K - arith_mean, 0)

            if not self.use_control_variate:
                # Return plain Monte Carlo pricing if control variate is not used
                mean_price = np.mean(arith_payoff)
                std_price = np.std(arith_payoff)
                conf_interval = [
                    mean_price - 1.96 * std_price / np.sqrt(self.M),
                    mean_price + 1.96 * std_price / np.sqrt(self.M)
                ]
                return {
                    'cv_estimate': str(mean_price),
                    'cv_ci': [str(conf_interval[0]), str(conf_interval[1])],
                }

            # Control variate logic
            geo_mean = np.sqrt(ST1 * ST2)
            if self.option_type == 'call':
                geo_payoff = np.exp(-self.r * self.T) * np.maximum(geo_mean - self.K, 0)
            else:
                geo_payoff = np.exp(-self.r * self.T) * np.maximum(self.K - geo_mean, 0)

            geo_exact = self.geometric_basket_price()

            cov_xy = np.cov(arith_payoff, geo_payoff)[0, 1]
            theta = cov_xy / np.var(geo_payoff)
            Z = arith_payoff + theta * (geo_exact - geo_payoff)

            Zmean = np.mean(Z)
            Zstd = np.std(Z)

            conf_interval = [
                Zmean - 1.96 * Zstd / np.sqrt(self.M),
                Zmean + 1.96 * Zstd / np.sqrt(self.M)
            ]

            return {
                'cv_estimate': str(Zmean),
                'cv_ci': [str(conf_interval[0]), str(conf_interval[1])],
            }
        
        except Exception as e:
            raise ValueError(f"Error in Monte Carlo pricing: {e}")

if __name__ == "__main__":
    test_cases = [
        (100, 100, 100, 0.3, 0.3, 0.5, "call", 22.10),
        (100, 100, 100, 0.3, 0.3, 0.9, "call", 25.88),
        (100, 100, 100, 0.1, 0.3, 0.5, "call", 17.92),
        (100, 100, 80, 0.3, 0.3, 0.5, "call", 32.54),
        (100, 100, 120, 0.3, 0.3, 0.5, "call", 14.69),
        (100, 100, 100, 0.5, 0.5, 0.5, "call", 28.45),
        (100, 100, 100, 0.3, 0.3, 0.5, "put", 11.49),
        (100, 100, 100, 0.3, 0.3, 0.9, "put", 12.62),
        (100, 100, 100, 0.1, 0.3, 0.5, "put", 6.59),
        (100, 100, 80, 0.3, 0.3, 0.5, "put", 4.71),
        (100, 100, 120, 0.3, 0.3, 0.5, "put", 21.29),
        (100, 100, 100, 0.5, 0.5, 0.5, "put", 23.47),
    ]

    r = 0.05
    T = 3
    M = 100000

    for S1, S2, K, sigma1, sigma2, rho, option_type, closed_form_price in test_cases:
        pricer = BasketOptionPricer(S1, S2, K, T, r, sigma1, sigma2, rho, M, option_type)
        result = pricer.monte_carlo_pricing()
        print(f"Test Case: S1={S1}, S2={S2}, K={K}, sigma1={sigma1}, sigma2={sigma2}, rho={rho}, option_type={option_type}")
        print(f"Our Closed Form Price: {closed_form_price}")
        print(f"Monte Carlo Estimate: {result['cv_estimate']}")
        print(f"95% Confidence Interval: {result['cv_ci']}")
        print("-" * 80)
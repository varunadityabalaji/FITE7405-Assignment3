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
            sigma_b = np.sqrt((self.sigma1**2 + self.sigma2**2 + 2 * self.rho * self.sigma1 * self.sigma2) / 4 )
            mu_b = (np.log(self.S1 * self.S2) / 2 + (self.r - 0.5 * (self.sigma1**2 + self.sigma2**2) / 2 + 0.5 * sigma_b**2) * self.T)

            d1 = (np.log(np.exp(mu_b) / self.K) + (sigma_b**2) * self.T) / (sigma_b * np.sqrt(self.T))
            d2 = d1 - sigma_b * np.sqrt(self.T)

            if self.option_type == "call":
                price = np.exp(-self.r * self.T) * (np.exp(mu_b) * norm.cdf(d1) - self.K * norm.cdf(d2))
            else:
                price = np.exp(-self.r * self.T) * (self.K * norm.cdf(-d2) - np.exp(mu_b) * norm.cdf(-d1))
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
                    'plain_mc_estimate': str(mean_price),
                    'plain_mc_ci': [str(conf_interval[0]), str(conf_interval[1])],
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
    try:
        pricer = BasketOptionPricer(
            S1=100, S2=100, K=100, T=1, r=0.05, 
            sigma1=0.2, sigma2=0.2, rho=0.5, M=10000, option_type="put", use_control_variate=True
        )
        print(pricer)
        print("Monte Carlo Pricing:", pricer.monte_carlo_pricing())
    except Exception as e:
        print(f"Error in main execution: {e}")

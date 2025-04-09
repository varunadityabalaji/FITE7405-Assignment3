"""
Author: Sai Navyanth
SID: 
"""

from math import sqrt, log,exp,pi
from scipy.stats import norm

class BlackScholes(object):
    """
    BlackScholes class for option pricing and implied volatility calculation.
    This class implements the Black-Scholes model for European options. It provides
    methods to calculate option prices (call and put), implied volatility, and other
    related metrics.
    Attributes:
        S (float): Spot price of the underlying asset.
        K (float): Strike price of the option.
        T (float): Time to maturity (in years).
        t (float): Current time (in years).
        r (float): Risk-free interest rate.
        q (float): Dividend yield (default is 0).
    Methods:
        calculate_d1_d2(sigma): Computes the d1 and d2 parameters for the Black-Scholes formula.
        call(sigma): Calculates the price of a call option given volatility (sigma).
        put(sigma): Calculates the price of a put option given volatility (sigma).
        vega(sigma): Computes the vega of the option, representing sensitivity to volatility.
        verify_bounds(price, optionType): Verifies if the option price is within theoretical bounds.
        implied_volatility(Ctrue, OptionType): Estimates the implied volatility for a given option price.
    """
       
    def __init__(self, S, K, T, t, r, q=0):
        self.S = S  # Spot price
        self.K = K  # Strike price
        self.T = T  # Time to maturity
        self.t = t  # Current time
        self.r = r  # Risk-free rate
        self.q = q  # Dividend yield

    def __repr__(self):
        return (f"BlackScholes(S={self.S}, K={self.K}, T={self.T}, t={self.t}, "
                f"r={self.r}, q={self.q})")

    def __eq__(self, other):
        if not isinstance(other, BlackScholes):
            return NotImplemented
        return (self.S == other.S and self.K == other.K and self.T == other.T and
                self.t == other.t and self.r == other.r and self.q == other.q)

    def __hash__(self):
        return hash((self.S, self.K, self.T, self.t, self.r, self.q))

    def calculate_d1_d2(self, sigma):
        d1 = ((log(self.S / self.K) + (self.r - self.q) * (self.T - self.t)) / (sigma * sqrt(self.T - self.t))) + 0.5 * sigma * sqrt(self.T - self.t)
        d2 = d1 - (sigma * sqrt(self.T - self.t))
        return d1, d2

    def call(self, sigma):
        d1, d2 = self.calculate_d1_d2(sigma)
        C = self.S * norm.cdf(d1) * exp(-1 * self.q * (self.T - self.t)) - self.K * exp(-1 * self.r * (self.T - self.t)) * norm.cdf(d2)
        return C

    def put(self, sigma):
        d1, d2 = self.calculate_d1_d2(sigma)
        P = self.K * exp(-1 * self.r * (self.T - self.t)) * norm.cdf(-1 * d2) - self.S * norm.cdf(-d1) * exp(-1 * self.q * (self.T - self.t))
        return P

    def vega(self, sigma):
        d1, _ = self.calculate_d1_d2(sigma)
        vega = self.S * exp(-1 * self.q * (self.T - self.t)) * sqrt(self.T - self.t) * exp(-1 * 0.5 * d1 * sigma**2) / sqrt(2 * pi)
        return vega

    def verify_bounds(self, price, optionType):
        if optionType == 'C':
            call_lower_bound = max(self.S * exp(-1 * self.q * self.T) - self.K * exp(-1 * self.r * self.T), 0)
            call_upper_bound = self.S * exp(-1 * self.q * self.T)
            return call_lower_bound <= price < call_upper_bound

        if optionType == 'P':
            put_lower_bound = max(self.K * exp(-1 * self.r * self.T) - self.S * exp(-1 * self.q * self.T), 0)
            put_upper_bound = self.K * exp(-1 * self.r * self.T)
            return put_lower_bound <= price < put_upper_bound

    def implied_volatility(self, Ctrue, OptionType='C'):
        sigmahat = sqrt(2 * abs((log(self.S / self.K) + (self.r - self.q) * (self.T - self.t)) / (self.T - self.t)))  # Initial guess
        tol = 1e-8  # Tolerance
        nmax = 1000
        sigmadiff = 1
        n = 1
        sigma = sigmahat

        while sigmadiff >= tol and nmax > n:
            if OptionType == 'C':
                C = self.call(sigma)
            else:
                C = self.put(sigma)

            Cvega = self.vega(sigma)
            increment = (C - Ctrue) / Cvega
            sigma = sigma - increment
            n += 1
            sigmadiff = abs(increment)

        return sigma

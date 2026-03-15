"""
bs_pricer.py — Black-Scholes pricer extracted from Phase 4.
Minimal version for use in the Streamlit app.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass

RISK_FREE_RATE = 0.065
DIVIDEND_YIELD = 0.013


@dataclass
class BSOption:
    S: float
    K: float
    T: float
    r: float = RISK_FREE_RATE
    q: float = DIVIDEND_YIELD
    sigma: float = 0.15
    option_type: str = "call"

    @property
    def d1(self):
        if self.T <= 0 or self.sigma <= 0:
            return np.inf if self.S >= self.K else -np.inf
        return (np.log(self.S / self.K) +
                (self.r - self.q + 0.5 * self.sigma**2) * self.T) / \
               (self.sigma * np.sqrt(self.T))

    @property
    def d2(self):
        return self.d1 - self.sigma * np.sqrt(self.T)

    def price(self):
        if self.T <= 0:
            return max(self.S - self.K, 0) if self.option_type == "call" \
                   else max(self.K - self.S, 0)
        d1, d2 = self.d1, self.d2
        disc_S = self.S * np.exp(-self.q * self.T)
        disc_K = self.K * np.exp(-self.r * self.T)
        if self.option_type == "call":
            return disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
        return disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)

    def delta(self):
        if self.T <= 0:
            return (1.0 if self.S > self.K else 0.0) if self.option_type == "call" \
                   else (-1.0 if self.S < self.K else 0.0)
        f = np.exp(-self.q * self.T)
        return f * norm.cdf(self.d1) if self.option_type == "call" \
               else -f * norm.cdf(-self.d1)

    def gamma(self):
        if self.T <= 0 or self.sigma <= 0: return 0.0
        return (np.exp(-self.q * self.T) * norm.pdf(self.d1)) / \
               (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        if self.T <= 0 or self.sigma <= 0: return 0.0
        return self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1) * \
               np.sqrt(self.T) * 0.01

    def theta(self):
        if self.T <= 0 or self.sigma <= 0: return 0.0
        d1, d2 = self.d1, self.d2
        t1 = -(self.S * np.exp(-self.q * self.T) * norm.pdf(d1) * self.sigma) / \
              (2 * np.sqrt(self.T))
        if self.option_type == "call":
            return (t1 - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2) +
                    self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(d1)) / 365
        return (t1 + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) -
                self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-d1)) / 365


def implied_volatility(market_price, S, K, T, r=RISK_FREE_RATE,
                       q=DIVIDEND_YIELD, option_type="call"):
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    if market_price < intrinsic - 1e-4: return np.nan
    if market_price <= intrinsic + 1e-6: return 1e-6

    def obj(sigma):
        return BSOption(S, K, T, r, q, sigma, option_type).price() - market_price

    try:
        if obj(1e-6) * obj(20.0) > 0: return np.nan
        return brentq(obj, 1e-6, 20.0, xtol=1e-8, maxiter=500)
    except Exception:
        return np.nan

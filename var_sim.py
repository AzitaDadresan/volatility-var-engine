# Monte Carlo VaR simulation using GARCH forecasted volatility
# Implements GBM under stochastic volatility (CQF Module 4)

import numpy as np
import pandas as pd
from garch_fit import fetch_data, fit_garch

def monte_carlo_var(S0, mu, sigma, T=1/252, paths=10000, confidence=0.95):
    """
    Monte Carlo simulation for VaR estimation.
    Args:
        S0: Initial price
        mu: Expected return (annualized)
        sigma: Volatility (from GARCH forecast, annualized %)
        T: Time horizon (1 trading day)
        paths: Number of MC paths
        confidence: VaR confidence level
    Returns:
        VaR, CVaR at specified confidence
    """
    sigma = sigma / 100
    
    Z = np.random.standard_normal(paths)
    S_T = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    pnl = S_T - S0
    
    var = np.percentile(pnl, (1 - confidence) * 100)
    cvar = pnl[pnl <= var].mean()
    
    return var, cvar, pnl

if __name__ == '__main__':
    returns = fetch_data('SPY')
    _, forecast_vol = fit_garch(returns)
    
    S0 = 450.0
    mu = returns.mean() * 252 / 100
    
    var_95, cvar_95, pnl_dist = monte_carlo_var(
        S0=S0, 
        mu=mu, 
        sigma=forecast_vol * np.sqrt(252),
        paths=10000
    )
    
    print(f"\n=== Monte Carlo VaR (10K paths, 1-day horizon) ===")
    print(f"Initial Price: ${S0:.2f}")
    print(f"GARCH Volatility (annualized): {forecast_vol * np.sqrt(252):.2f}%")
    print(f"95% VaR: ${var_95:.2f}")
    print(f"95% CVaR: ${cvar_95:.2f}")
    
    pd.DataFrame({'PnL': pnl_dist}).to_csv('var_simulation.csv', index=False)
    print("\nResults saved to var_simulation.csv")

# GARCH(1,1) volatility forecasting for VaR estimation
# Author: Azita Dadresan | CQF (Module 3: Volatility Modeling)
# Implements ARCH effects: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
import matplotlib.pyplot as plt

def fetch_data(ticker='SPY', period='1y'):
    """Download historical price data from Yahoo Finance."""
    data = yf.download(ticker, period=period, progress=False)
    returns = 100 * data['Adj Close'].pct_change().dropna()
    return returns

def ewma_volatility(returns, lambda_param=0.94):
    """Exponentially weighted moving average volatility."""
    var = returns.var()
    ewma_var = [var]
    for r in returns[1:]:
        var = lambda_param * var + (1 - lambda_param) * r**2
        ewma_var.append(var)
    return np.sqrt(np.array(ewma_var))

def fit_garch(returns):
    """Fit GARCH(1,1) and forecast 1-step ahead volatility."""
    model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
    fitted = model.fit(disp='off')
    
    print("\n=== GARCH(1,1) Fit ===")
    print(fitted.summary())
    
    forecast = fitted.forecast(horizon=1)
    forecasted_vol = np.sqrt(forecast.variance.values[-1, 0])
    
    return fitted, forecasted_vol

if __name__ == '__main__':
    returns = fetch_data('SPY')
    print(f"Loaded {len(returns)} daily returns for SPY")
    
    ewma_vol = ewma_volatility(returns.values)
    garch_fit, forecast_vol = fit_garch(returns)
    
    print(f"\n1-day ahead volatility forecast: {forecast_vol:.4f}%")
    
    plt.figure(figsize=(12, 6))
    plt.plot(returns.index, ewma_vol, label='EWMA (λ=0.94)', alpha=0.7)
    plt.plot(returns.index, np.sqrt(garch_fit.conditional_volatility), 
             label='GARCH(1,1)', alpha=0.7)
    plt.title('SPY Volatility: EWMA vs GARCH(1,1)')
    plt.xlabel('Date')
    plt.ylabel('Volatility (%)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('volatility_comparison.png', dpi=150)
    plt.show() 

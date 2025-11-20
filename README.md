# Volatility VaR Engine

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**EWMA/GARCH volatility models + Monte Carlo VaR for intraday risk forecasting.**

## Features
- **EWMA (λ=0.94)**: Exponentially weighted moving average volatility
- **GARCH(1,1)**: Generalized AutoRegressive Conditional Heteroskedasticity
- **Monte Carlo VaR**: 10K paths, 95% confidence interval
- **yfinance integration**: Real-time S&P 500 data pull
- **Pytest suite**: Unit tests for model accuracy

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Fit GARCH model and forecast volatility
python garch_fit.py

# Run Monte Carlo VaR simulation
python var_sim.py

# Run tests
pytest test_var.py -v
```

## Use Case
Risk management for portfolio managers at hedge funds (Bridgewater, AQR). Extends my Fidelity anomaly detection work to volatility regime detection. Aligns with CQF Module 3 (volatility modeling).

## Author
Azita Dadresan | CQF, JHU TA (Stochastic Processes)

## License
MIT 

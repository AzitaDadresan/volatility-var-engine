# Unit tests for volatility and VaR calculations

import pytest
import numpy as np
from garch_fit import ewma_volatility
from var_sim import monte_carlo_var

def test_ewma_decay():
    """Test EWMA gives more weight to recent observations."""
    returns = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    vol = ewma_volatility(returns, lambda_param=0.9)
    
    assert vol[-1] > vol[0], "EWMA should increase with rising returns"
    assert len(vol) == len(returns), "Output length mismatch"

def test_var_negative():
    """VaR should be negative (loss) for 95% confidence."""
    var, cvar, _ = monte_carlo_var(S0=100, mu=0.05, sigma=20, paths=1000)
    
    assert var < 0, "95% VaR should represent a loss"
    assert cvar < var, "CVaR should be more extreme than VaR"

def test_var_convergence():
    """VaR should converge with more paths."""
    np.random.seed(42)
    var_1k, _, _ = monte_carlo_var(S0=100, mu=0.05, sigma=20, paths=1000)
    np.random.seed(42)
    var_10k, _, _ = monte_carlo_var(S0=100, mu=0.05, sigma=20, paths=10000)
    
    assert abs(var_1k - var_10k) < 5, "VaR should converge with more paths"

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

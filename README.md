# Regime-Aware Dual Momentum Allocator (RADM)

This project explores a **regime-aware alternative to the classic 60/40 portfolio**.

---

## Why not 60/40?
The traditional 60% equities / 40% bonds portfolio has been a cornerstone of portfolio construction for decades. It works well in periods when:
- **Equities trend upward** with manageable volatility, and
- **Bonds diversify** by providing positive real returns and offsetting equity drawdowns.

But history shows that in **negative real yield regimes** (when inflation > interest rates) or during **volatility shocks**, bonds often fail to diversify. Following 60/40 blindly in these environments can mean:
- Losing purchasing power in “safe” assets,  
- Larger drawdowns than expected,  
- No protection when diversification is needed most.  

---

## What this code does
This repository implements a **simple, transparent allocator** that adapts weights based on **macro-financial regimes**.  

### Regime detection
- **NEG regime:** if real yields < 0 or volatility (VIX) exceeds a threshold.  
- **POS regime:** otherwise.  

### Portfolio construction
- In each regime, select **2–3 assets** from a predefined pool using **momentum (12-month total return)**.  
- Equal-weight selected assets for simplicity.  
- Always include **cash** (`BIL`) as an anchor:
  - Acts as a volatility buffer.  
  - Provides flexibility for volatility targeting.  

### Risk controls
- **Volatility targeting:** scale exposures toward a desired annualized vol.  
- **Cash floors:** ensure a minimum allocation to cash in riskier regimes.  
- **Transaction costs:** included to reflect real-world frictions.  

### Benchmark
- Results are compared against a **classic 60/40 (SPY/AGG)** rebalanced monthly.  

---

## Why it matters
This allocator is intentionally simple:
- Uses **momentum ranking** instead of complex ML or Bayesian optimizers.  
- Equal-weights chosen assets to avoid overfitting.  
- Keeps risk management transparent and easy to interpret.  

The point is not to maximize backtest Sharpe ratios, but to **demonstrate how regime awareness changes portfolio behavior**:
- Lower drawdowns in NEG regimes.  
- Real-return resilience when bonds bleed under inflation.  
- A reminder that “safe” assets are only safe in the right environment.  

---

## Usage
Example run:

```bash
python regime_dual_momentum.py \
  --neg_pool GLD DBC VNQ AGG TIP \
  --pos_pool SPY QUAL VLUE AGG LQD \
  --start 2005-01-01 \
  --top_k 2 --cost_bps 10 \
  --vol_setpoint 0.10 \
  --cash_floor 0.03 \
  --vix_threshold 20
```

---

## Outputs
- Nominal and real-return performance (CAGR, Vol, Sharpe, MaxDD).  
- Conditional performance by regime (NEG vs POS).  
- Cumulative return, drawdown, and rolling Sharpe plots.  
- CSV file with portfolio & benchmark returns.  

---

## Example Results

### Nominal Performance
```
== 60/40 (Benchmark) ==
CAGR   7.9% | Vol   9.7% | Sharpe  0.84 | MaxDD -32%

== RADM (Portfolio) ==
CAGR   15.1% | Vol   9.4% | Sharpe  1.55 | MaxDD -11%
```

### Real Performance (inflation-adjusted)
```
== 60/40 (REAL) ==
CAGR   5.3% | Vol   9.8% | Sharpe  0.58 | MaxDD -33%

== RADM (REAL) ==
CAGR   12.3% | Vol   9.3% | Sharpe  1.29 | MaxDD -11%
```

These results highlight:
- **60/40 loses purchasing power** in inflationary NEG regimes.  
- **RADM adapts** by shifting into assets like commodities, gold, or real estate when real yields are negative or volatility is high.  
- Substantial **drawdown reduction** vs 60/40.  

---

## Dependencies
Install with:

```
pip install -r requirements.txt
```

---

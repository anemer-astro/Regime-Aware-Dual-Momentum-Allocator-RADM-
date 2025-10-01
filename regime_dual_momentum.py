#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regime-Aware Dual Momentum (RADM) — auto regime detection + real-return overlay
-------------------------------------------------------------------------------
What this file does (end-to-end):
1) Downloads daily prices for all tickers in your NEG/POS pools + SPY, AGG, DBC, GLD, etc.
2) Fetches CPI (CPIAUCSL) and 10Y yield (DGS10) from FRED to compute:
      • CPI YoY inflation  (annual, %)
      • Real yield = DGS10 - CPI YoY
3) Auto-detects regimes monthly:
      NEG if (real yield < 0) OR (VIX > threshold); else POS.
4) Builds monthly returns, runs RADM:
      • In each month, choose the regime pool
      • Pick top-K by 12m momentum; apply 10m absolute momentum gate
      • Equal-weight selected assets; vol-target via cash to hit setpoint
      • Keep a small cash floor
5) Builds monthly-rebalanced 60/40 benchmark (SPY/AGG).
6) Computes nominal metrics AND **real-return overlay** (deflating by CPI).
7) Saves: joined CSV + plots (equity curves nominal & real, rolling vol, cash weight, selection ribbon).
"""

from __future__ import annotations
import argparse
import math
import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# FRED (for CPI & DGS10)
from pandas_datareader import data as pdr


# ----------------------------- Utilities ------------------------------------ #

def month_end_index(ix: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Coerce any timestamp index to month-end timestamps."""
    return ix.to_period("M").to_timestamp("M")

def to_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert daily Adj Close to month-end total returns."""
    m = prices.resample("M").last()
    rets = m.pct_change().dropna(how="all")
    rets.index = month_end_index(rets.index)
    return rets

def cagr_vol_sharpe_dd(monthly_rets: pd.Series, rf_annual: float = 0.0) -> Dict[str, float]:
    """CAGR, annualized vol, Sharpe (vs RF), Max Drawdown from monthly returns."""
    r = monthly_rets.dropna().values
    if len(r) == 0:
        return dict(CAGR=np.nan, Vol=np.nan, Sharpe=np.nan, MaxDD=np.nan)
    yrs = len(r) / 12.0
    cagr = (1.0 + r).prod() ** (1.0 / yrs) - 1.0 if yrs > 0 else np.nan
    vol = np.std(r, ddof=1) * np.sqrt(12) if len(r) > 1 else np.nan
    rf_m = (1 + rf_annual) ** (1 / 12) - 1
    ex_r = r - rf_m
    sharpe = (ex_r.mean() / (np.std(ex_r, ddof=1) + 1e-12)) * np.sqrt(12) if len(r) > 1 else np.nan
    curve = np.cumprod(1 + r)
    peaks = np.maximum.accumulate(curve)
    maxdd = float(np.min(curve / (peaks + 1e-12) - 1)) if len(curve) else np.nan
    return dict(CAGR=cagr, Vol=vol, Sharpe=sharpe, MaxDD=maxdd)

def print_metrics(title: str, rets: pd.Series, rf_annual: float = 0.0):
    m = cagr_vol_sharpe_dd(rets, rf_annual=rf_annual)
    print(f"\n== {title} ==")
    print(f"Nominal | CAGR {m['CAGR']*100:6.2f}% | Vol {m['Vol']*100:6.2f}% | Sharpe {m['Sharpe']:5.2f} | MaxDD {m['MaxDD']*100:6.2f}%")

def print_metrics_real(title: str, nominal_rets: pd.Series, cpi_index: pd.Series):
    """Compute real returns by deflating the equity curve with CPI index."""
    # Build nominal equity curve, then deflate by CPI index normalized to 1
    nom_curve = (1 + nominal_rets.fillna(0)).cumprod()
    cpi = cpi_index.reindex(nom_curve.index).ffill()
    cpi_norm = cpi / cpi.iloc[0]
    real_curve = nom_curve / cpi_norm
    # Convert back to monthly 'real' returns
    real_rets = real_curve.pct_change().fillna(0.0)
    m = cagr_vol_sharpe_dd(real_rets, rf_annual=0.0)
    print(f"\n== {title} (REAL) ==")
    print(f"Real    | CAGR {m['CAGR']*100:6.2f}% | Vol {m['Vol']*100:6.2f}% | Sharpe {m['Sharpe']:5.2f} | MaxDD {m['MaxDD']*100:6.2f}%")
    return real_rets, real_curve

# ----------------------------- Data Download -------------------------------- #

def download_prices(tickers: List[str], start: str = "2005-01-01") -> pd.DataFrame:
    """Download daily Adj Close for tickers via yfinance."""
    tickers = sorted(set([t.strip().upper() for t in tickers if t.strip()]))
    if not tickers:
        raise ValueError("No tickers provided to download.")
    px = yf.download(tickers, start=start, auto_adjust=False, progress=False)["Adj Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame()
    px = px.dropna(how="all")
    if px.empty:
        raise ValueError("Empty price data. Check tickers/date range.")
    return px.reindex(columns=tickers)

def fetch_cpi_and_10y(start: str = "2005-01-01"):
    """
    Fetch CPIAUCSL (index, monthly) and DGS10 (%, daily) from FRED.
    Returns:
        cpi_m  : monthly CPI index (level)
        infl_yoy: monthly YoY inflation rate (%) aligned to month-end
        dgs10_m: monthly 10y yield (%) aligned to month-end
        real_yield: dgs10_m - infl_yoy
    Graceful fallback:
        If FRED fails, we derive a CPI proxy from GLD/DBC (very rough) and skip real-yield regimes.
    """
    try:
        cpi = pdr.DataReader("CPIAUCSL", "fred", start=start)  # monthly level
        dgs10 = pdr.DataReader("DGS10", "fred", start=start)   # daily %, 10y constant maturity
        cpi.index = month_end_index(cpi.index)
        infl_yoy = cpi.pct_change(12) * 100.0
        dgs10_m = dgs10.resample("M").last()
        dgs10_m.index = month_end_index(dgs10_m.index)
        real_yield = dgs10_m["DGS10"] - infl_yoy["CPIAUCSL"]
        # Rename for clarity
        cpi.columns = ["CPIAUCSL"]
        infl_yoy.name = "CPI_YoY_pct"
        dgs10_m.columns = ["DGS10"]
        real_yield.name = "Real_Yield_pct"
        return cpi["CPIAUCSL"], infl_yoy, dgs10_m["DGS10"], real_yield
    except Exception as e:
        print(f"[WARN] FRED fetch failed ({e}). Falling back to heuristic CPI proxy.")
        # CPI proxy: accumulate monthly returns of a commodity basket (DBC) + small lag
        # This is only for the REAL overlay fallback; regime detection will revert to VIX-only.
        return None, None, None, None

# --------------------------- Auto Regime Detection --------------------------- #

def detect_regimes_auto(monthly_rets: pd.DataFrame,
                        vix_daily: pd.Series,
                        real_yield_pct: Optional[pd.Series],
                        vix_threshold: float = 20.0) -> pd.Series:
    """
    NEG if (real_yield < 0) OR (VIX > threshold); else POS.
    Returns a 1-D Series indexed by monthly_rets.index.
    """
    # Target monthly index (strict month-end)
    idx = monthly_rets.index.to_period("M").to_timestamp("M")

    # --- VIX: month-end Series aligned to idx (force 1-D) ---
    if isinstance(vix_daily, pd.DataFrame):
        # if it sneaks in as a 1-col frame, squeeze it
        if "VIX" in vix_daily.columns:
            vix_daily = vix_daily["VIX"]
        else:
            vix_daily = vix_daily.squeeze()
    vix_m = vix_daily.resample("M").last()
    vix_m.index = vix_m.index.to_period("M").to_timestamp("M")
    vix_m = vix_m.reindex(idx).ffill()

    # Build individual 1-D boolean conditions as numpy vectors
    vix_cond = (vix_m > vix_threshold).reindex(idx).fillna(False).to_numpy()

    if real_yield_pct is not None:
        ry = real_yield_pct.copy()
        # ensure it's a Series (not DF); if DF, take first column
        if isinstance(ry, pd.DataFrame):
            ry = ry.squeeze()
        ry.index = ry.index.to_period("M").to_timestamp("M")
        ry = ry.reindex(idx).ffill()
        ry_cond = (ry < 0).fillna(False).to_numpy()
        neg_cond = np.logical_or(ry_cond, vix_cond)
    else:
        neg_cond = vix_cond

    # Build labels from a 1-D boolean mask
    labels = pd.Series(np.where(neg_cond, "RY-NEG", "RY-POS"), index=idx)
    return labels


# ------------------ Regime-Aware Dual Momentum Backtest --------------------- #

def backtest_dual_momentum_regime(
    monthly_returns: pd.DataFrame,
    regime_labels_m: pd.Series,
    cost_bps: float = 5.0,
    neg_pool: Optional[List[str]] = None,
    pos_pool: Optional[List[str]] = None,
    cash_tkr: str = "BIL",
    top_k: int = 2,
    vol_setpoint: float = 0.10,
    cash_floor: float = 0.05,
) -> pd.DataFrame:
    """
    RADM engine:
      • Pick pool by regime (NEG vs POS)
      • Top-K by 12m momentum, then 10m MA absolute gate
      • Equal-weight selected assets; vol-target via cash
      • Cash floor for tail control
      • Returns portfolio + realized weights + monitors
    """
    rets = monthly_returns.copy().dropna(how="all")
    rets.index = month_end_index(rets.index)
    labels = regime_labels_m.copy()
    labels.index = month_end_index(labels.index)
    idx = rets.index.intersection(labels.index)
    rets, labels = rets.loc[idx], labels.loc[idx]

    if cash_tkr not in rets.columns:
        rets[cash_tkr] = 0.0

    if neg_pool is None: neg_pool = ["GLD", "DBC", "VNQ", "SPY", "AGG"]
    if pos_pool is None: pos_pool = ["SPY", "QUAL", "VLUE", "AGG", "LQD"]

    def top_k_momentum(pool_cols: List[str], up_to_t: pd.Timestamp) -> List[str]:
        lb = rets.loc[:up_to_t].iloc[-13:-1]  # 12m lookback ending t-1
        cols = [c for c in pool_cols if c in lb.columns]
        if len(lb) < 6 or len(cols) == 0:
            return []
        mom = (1 + lb[cols]).prod() - 1.0
        return mom.sort_values(ascending=False).index.tolist()[:top_k]

    def above_10m_ma(series: pd.Series) -> bool:
        if len(series) < 11:
            return True
        idxlvl = (1 + series).cumprod()
        ma = idxlvl.rolling(10).mean().iloc[-1]
        return bool(idxlvl.iloc[-1] > ma)

    weights_hist, port_rets, mons = [], [], []
    prev_w = pd.Series(0.0, index=rets.columns)

    # Realized-vol controller
    realized_window: List[float] = []
    target_vol = vol_setpoint
    ctrl_alpha = 0.15
    max_leverage = 1.00

    for t in idx:
        lab = labels.loc[t]
        neg_real = ("RY-" in str(lab))
        pool = neg_pool if neg_real else pos_pool
        pool = [c for c in pool if c in rets.columns and c != cash_tkr]

        # 1) Relative momentum
        top_sel = top_k_momentum(pool, t)

        # 2) Absolute momentum gate
        selected = []
        for a in top_sel:
            hist = rets[a].loc[:t].iloc[-11:]  # last ~11 months
            if above_10m_ma(hist):
                selected.append(a)

        # 3) Equal-weight risky sleeve (or cash if none pass)
        w = pd.Series(0.0, index=rets.columns)
        if len(selected) == 0:
            w[cash_tkr] = 1.0
        else:
            w[selected] = 1.0 / len(selected)

        # 4) Cash floor
        if w[cash_tkr] < cash_floor:
            need = cash_floor - w[cash_tkr]
            others = [c for c in w.index if c != cash_tkr and w[c] > 0]
            if others:
                pro = w[others] / max(w[others].sum(), 1e-12)
                w[others] = (w[others] - pro * need).clip(lower=0.0)
                w[cash_tkr] = cash_floor
                if w.sum() > 1.0:
                    w = w / w.sum()

        # 5) Vol targeting via realized 12m portfolio vol
        realized_window = realized_window[-11:]
        if port_rets:
            realized_window.append(port_rets[-1])
        if len(realized_window) >= 6:
            realized_ann = np.std(realized_window, ddof=1) * np.sqrt(12)
            ratio = vol_setpoint / max(1e-6, realized_ann)
            target_vol = float(np.clip(
                (1 - ctrl_alpha) * target_vol + ctrl_alpha * (target_vol * ratio), 0.06, 0.18
            ))
        else:
            realized_ann = np.nan

        # 6) Approx ex-ante vol from trailing 12m cov of active sleeve
        lb = rets.loc[:t].iloc[-13:-1]
        act = [c for c in w.index if w[c] > 0 and c != cash_tkr]
        if len(lb) >= 6 and len(act) >= 1:
            cov = lb[act].cov() * 12.0
            if len(act) == 1:
                ex_ante = float(np.sqrt(max(cov.values[0, 0], 0.0)))
            else:
                w_act = (w[act] / w[act].sum()).values
                ex_ante = float(np.sqrt(max(w_act @ cov.values @ w_act, 0.0)))
        else:
            ex_ante = 0.0

        gross_risky = min(max_leverage, (target_vol / ex_ante)) if ex_ante > 1e-9 else 0.0
        w_target = pd.Series(0.0, index=rets.columns)
        if len(act) > 0:
            w_target[act] = gross_risky * (w[act] / max(w[act].sum(), 1e-12))
        w_target[cash_tkr] += max(0.0, 1.0 - w_target.sum())
        if w_target.sum() > 1.0:
            w_target = w_target / w_target.sum()

        # 7) Costs & PnL
        prev_w = prev_w.reindex(w_target.index).fillna(0.0)
        turnover = (w_target - prev_w).abs().sum()
        tc = turnover * (cost_bps / 10000.0)
        r = float((w_target * rets.loc[t]).sum() - tc)

        port_rets.append(r)
        weights_hist.append(w_target.rename(t))
        prev_w = w_target

        mons.append({
            "date": t, "regime": lab, "neg_real": bool(neg_real),
            "selected": ",".join(selected) if selected else "",
            "ex_ante_vol": ex_ante, "realized_vol_ann_proxy": realized_ann,
            "target_vol": target_vol, "cash_w": w_target.get(cash_tkr, 0.0),
            "turnover": turnover, "net_r": r
        })

    port = pd.Series(port_rets, index=rets.index, name="Portfolio")
    W = pd.DataFrame(weights_hist).reindex(columns=rets.columns).fillna(0.0)
    M = pd.DataFrame(mons).set_index("date")
    M.columns = [f"m:{c}" for c in M.columns]
    return pd.concat([port, W], axis=1).join(M, how="left")

# --------------------------- 60/40 Benchmark -------------------------------- #

def backtest_60_40(monthly_rets: pd.DataFrame, spy: str = "SPY", agg: str = "AGG") -> pd.Series:
    """
    Monthly-rebalanced 60/40 benchmark using SPY/AGG monthly total returns.
    Equivalent to 0.6*SPY + 0.4*AGG each month.
    """
    if spy not in monthly_rets.columns or agg not in monthly_rets.columns:
        raise ValueError("60/40 needs both SPY and AGG in monthly_rets.")

    # Align and drop months where either leg is missing
    pair = monthly_rets[[spy, agg]].dropna(how="any").copy()

    port = 0.6 * pair[spy] + 0.4 * pair[agg]
    port.name = "Benchmark_60_40"
    return port
# --------------------------------- Plots ------------------------------------ #

def plot_equity_curves(nom_radm: pd.Series, nom_bench: pd.Series,
                       real_radm: pd.Series, real_bench: pd.Series, outdir: str):
    """Plot nominal and real equity curves."""
    eq_nom_radm = (1 + nom_radm).cumprod()
    eq_nom_bench = (1 + nom_bench).cumprod()
    eq_real_radm = (1 + real_radm).cumprod()
    eq_real_bench = (1 + real_bench).cumprod()

    plt.figure(figsize=(10, 6))
    plt.plot(eq_nom_radm.index, eq_nom_radm.values, label="RADM (Nominal)")
    plt.plot(eq_nom_bench.index, eq_nom_bench.values, label="60/40 (Nominal)")
    plt.title("Equity Curves — Nominal")
    plt.xlabel("Date"); plt.ylabel("Growth of $1"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "equity_nominal.png"), dpi=160); plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(eq_real_radm.index, eq_real_radm.values, label="RADM (Real)")
    plt.plot(eq_real_bench.index, eq_real_bench.values, label="60/40 (Real)")
    plt.title("Equity Curves — Real (CPI Deflated)")
    plt.xlabel("Date"); plt.ylabel("Real Growth of $1"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "equity_real.png"), dpi=160); plt.close()

def plot_rolling_vol(nom_radm: pd.Series, nom_bench: pd.Series, outdir: str):
    """Rolling 12m realized vol (annualized) on nominal returns."""
    rv_radm  = nom_radm.rolling(12).std() * np.sqrt(12)
    rv_bench = nom_bench.rolling(12).std() * np.sqrt(12)
    plt.figure(figsize=(10, 4))
    plt.plot(rv_radm.index, rv_radm.values, label="RADM")
    plt.plot(rv_bench.index, rv_bench.values, label="60/40")
    plt.title("Rolling 12m Realized Vol (Ann.)")
    plt.xlabel("Date"); plt.ylabel("Vol"); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "rolling_vol_12m.png"), dpi=160); plt.close()

def plot_cash_weight(out_df: pd.DataFrame, cash_tkr: str, outdir: str):
    """Cash allocation through time."""
    cw = out_df.get(cash_tkr, pd.Series(index=out_df.index, data=np.nan))
    plt.figure(figsize=(10, 3))
    plt.plot(cw.index, cw.values)
    plt.title(f"Cash Weight ({cash_tkr})")
    plt.xlabel("Date"); plt.ylabel("Weight"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "cash_weight.png"), dpi=160); plt.close()

def plot_selection_ribbon(out_df: pd.DataFrame, outdir: str):
    """Number of selected assets passing the absolute momentum gate."""
    sel = out_df["m:selected"].fillna("").apply(lambda s: len(s.split(",")) if s else 0)
    plt.figure(figsize=(10, 2.6))
    plt.plot(sel.index, sel.values)
    plt.title("Selected Assets Count (Top-K Passing Gate)")
    plt.xlabel("Date"); plt.ylabel("Count"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "selection_count.png"), dpi=160); plt.close()

# --------------------------------- Main ------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="RADM — auto regimes (real-yield & VIX) + real-return overlay")
    ap.add_argument("--neg_pool", type=str, required=True,
                    help="NEG-real regime pool as comma-separated tickers (e.g., 'GLD,DBC,VNQ,SPY,AGG')")
    ap.add_argument("--pos_pool", type=str, required=True,
                    help="POS-real regime pool as comma-separated tickers (e.g., 'SPY,QUAL,VLUE,AGG,LQD')")
    ap.add_argument("--cash_tkr", type=str, default="BIL", help="Cash ticker (default BIL)")
    ap.add_argument("--start", type=str, default="2005-01-01", help="Start date for downloads (YYYY-MM-DD)")
    ap.add_argument("--cost_bps", type=float, default=5.0, help="Round-trip transaction costs (bps)")
    ap.add_argument("--top_k", type=int, default=2, help="Top-K assets by 12m momentum")
    ap.add_argument("--vol_setpoint", type=float, default=0.10, help="Target realized vol (ann.)")
    ap.add_argument("--cash_floor", type=float, default=0.05, help="Min cash allocation")
    ap.add_argument("--vix_threshold", type=float, default=20.0, help="VIX threshold for 'high vol'")
    ap.add_argument("--rf_annual", type=float, default=0.0, help="Annual risk-free for Sharpe")
    ap.add_argument("--out_csv", type=str, default="radm_output.csv", help="Output CSV path")
    ap.add_argument("--out_dir", type=str, default="out", help="Plots directory")
    args = ap.parse_args()

    # 1) Parse pools
    neg_pool = [t.strip().upper() for t in args.neg_pool.split(",") if t.strip()]
    pos_pool = [t.strip().upper() for t in args.pos_pool.split(",") if t.strip()]
    cash_tkr = args.cash_tkr.strip().upper()

    # 2) Build download universe
    base_needed = ["SPY", "AGG", "GLD", "DBC", "QUAL", "VLUE", "VNQ", "LQD", "TIP", cash_tkr]
    universe = sorted(set(neg_pool + pos_pool + base_needed))
    vix_tkr = "^VIX"

    print("Downloading prices...")
    px = download_prices(universe, start=args.start)

    # FIX: obtain VIX as a Series safely (no rename bug)
    vix_df = yf.download(vix_tkr, start=args.start, auto_adjust=False, progress=False)[["Adj Close"]]
    vix_df = vix_df.rename(columns={"Adj Close": "VIX"}).dropna()
    vix = vix_df["VIX"]

    # Monthly returns
    rets_m = to_monthly_returns(px)
    rets_m.index = rets_m.index.to_period("M").to_timestamp("M")


    # 3) Fetch CPI & 10y; compute real yield (DGS10 - CPI YoY)
    cpi_m, infl_yoy, dgs10_m, real_yield_pct = fetch_cpi_and_10y(start=args.start)
    if real_yield_pct is not None:
        real_yield_pct.index = real_yield_pct.index.to_period("M").to_timestamp("M")

    # 4) Auto regime detection: NEG if real_yield<0 or VIX>threshold
    print("Detecting regimes (NEG = Real Yield < 0 OR VIX > threshold)...")
    labels_m = detect_regimes_auto(rets_m, vix, real_yield_pct, vix_threshold=args.vix_threshold)

    # 5) Run RADM backtest
    print("Running RADM backtest...")
    out = backtest_dual_momentum_regime(
        monthly_returns=rets_m,
        regime_labels_m=labels_m,
        cost_bps=args.cost_bps,
        neg_pool=neg_pool,
        pos_pool=pos_pool,
        cash_tkr=cash_tkr,
        top_k=args.top_k,
        vol_setpoint=args.vol_setpoint,
        cash_floor=args.cash_floor,
    )

    # 6) 60/40 benchmark
    bench_6040 = backtest_60_40(rets_m, "SPY", "AGG").reindex(out.index).dropna()

    # 7) Save joined output
    out.to_csv(args.out_csv, float_format="%.8f")
    print(f"Saved: {args.out_csv}")

    # 8) Nominal metrics
    print_metrics("RADM (Portfolio)", out["Portfolio"], rf_annual=args.rf_annual)
    print_metrics("60/40 (Benchmark)", bench_6040, rf_annual=args.rf_annual)

    # 9) Real-return overlay (if CPI available)
    os.makedirs(args.out_dir, exist_ok=True)
    if cpi_m is not None:
        real_radm, real_curve_radm = print_metrics_real("RADM (Portfolio)", out["Portfolio"], cpi_m)
        real_bench, real_curve_bench = print_metrics_real("60/40 (Benchmark)", bench_6040, cpi_m)
        # Plots: nominal & real curves
        plot_equity_curves(out["Portfolio"], bench_6040, real_radm, real_bench, args.out_dir)
    else:
        # Fallback: plot nominal only if CPI missing
        real_radm = out["Portfolio"] * 0.0
        real_bench = bench_6040 * 0.0
        plot_equity_curves(out["Portfolio"], bench_6040, real_radm, real_bench, args.out_dir)

    # Rolling vol (nominal only) + diagnostics
    plot_rolling_vol(out["Portfolio"], bench_6040, args.out_dir)
    plot_cash_weight(out, cash_tkr, args.out_dir)
    plot_selection_ribbon(out, args.out_dir)

    print(f"Saved plots to: {args.out_dir}/")
    print("Done.")


if __name__ == "__main__":
    # Example:
    # python regime_dual_momentum.py --neg_pool "GLD,DBC,VNQ,SPY,AGG" --pos_pool "SPY,QUAL,VLUE,AGG,LQD"
    main()

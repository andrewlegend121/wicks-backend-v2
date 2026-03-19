"""
backtest_runner.py
──────────────────
Main backtest execution engine for Wicks.

Accepts a strategy config (confluence params + risk rules),
fetches candle data from Polygon.io, runs the confluence scorer,
simulates trades with the risk model, and returns results.

POST /backtest
  {
    "symbol":       "C:EURUSD",
    "window":       "3mo",
    "params":       { <confluence_id>: { <param>: <value> } },
    "min_conf":     2,
    "risk": {
      "rr":           2.0,
      "risk_pct":     1.0,
      "sl_type":      "ob_low",   // ob_low | swing_low | fixed_pips
      "sl_pips":      15,
      "max_trades":   null,
      "session":      null
    }
  }

Returns:
  {
    "symbol":       ...,
    "window":       ...,
    "total_setups": ...,
    "total_trades": ...,
    "wins":         ...,
    "losses":       ...,
    "winrate":      ...,
    "total_r":      ...,
    "avg_r":        ...,
    "max_drawdown": ...,
    "equity_curve": [...],
    "trades":       [...],
    "stats":        {...}
  }
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone

from confluence_scorer import score_confluences, summarise, ConfluentSetup


# ── Polygon.io data fetcher ───────────────────────────────────────────────────

POLYGON_KEY = os.getenv("POLYGON_API_KEY", "")

WINDOW_MAP = {
    "1d":  1, "3d":  3, "1w":  7, "2w": 14,
    "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365,
}

def _fetch_candles(symbol: str, window: str,
                   multiplier: int = 1, span: str = "minute") -> pd.DataFrame:
    """
    Fetch OHLCV candles from Polygon.io.
    Returns a 1-min base DataFrame with DatetimeIndex UTC.
    """
    days  = WINDOW_MAP.get(window, 30)
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range"
        f"/{multiplier}/{span}"
        f"/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        f"?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_KEY}"
    )

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])
    if not results:
        raise ValueError(f"No data returned for {symbol} / {window}")

    df = pd.DataFrame(results)
    df = df.rename(columns={
        "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume", "t": "timestamp_ms"
    })
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
    df = df.sort_index().dropna()
    return df


# ── Trade simulation ──────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_ts:      str
    direction:     str
    entry_price:   float
    sl_price:      float
    tp_price:      float
    exit_price:    float = 0.0
    exit_ts:       str = ""
    result:        str = "open"   # win | loss | be
    r_multiple:    float = 0.0
    pnl_pips:      float = 0.0
    confluences:   List[str] = field(default_factory=list)
    conf_score:    float = 0.0


def _simulate_trade(
    setup: ConfluentSetup,
    df_1min: pd.DataFrame,
    risk: Dict[str, Any],
) -> Trade:
    """
    Forward-simulate a single trade from entry to SL or TP hit.
    """
    rr          = float(risk.get("rr", 2.0))
    sl_pips     = float(risk.get("sl_pips", 15))
    pip_size    = 0.0001   # default forex; adjust for JPY etc.

    ep  = setup.entry_price
    sl_dist = sl_pips * pip_size

    if setup.direction == "bullish":
        sl = ep - sl_dist
        tp = ep + sl_dist * rr
    else:
        sl = ep + sl_dist
        tp = ep - sl_dist * rr

    trade = Trade(
        entry_ts    = str(setup.timestamp),
        direction   = setup.direction,
        entry_price = round(ep, 5),
        sl_price    = round(sl, 5),
        tp_price    = round(tp, 5),
        confluences = setup.confluences_hit,
        conf_score  = setup.confluence_score,
    )

    # Forward scan for SL or TP hit
    future = df_1min.loc[df_1min.index > setup.timestamp]
    for ts, row in future.iterrows():
        if setup.direction == "bullish":
            if row["low"] <= sl:
                trade.exit_price  = sl
                trade.exit_ts     = str(ts)
                trade.result      = "loss"
                trade.r_multiple  = -1.0
                trade.pnl_pips    = round((sl - ep) / pip_size, 1)
                break
            if row["high"] >= tp:
                trade.exit_price  = tp
                trade.exit_ts     = str(ts)
                trade.result      = "win"
                trade.r_multiple  = rr
                trade.pnl_pips    = round((tp - ep) / pip_size, 1)
                break
        else:
            if row["high"] >= sl:
                trade.exit_price  = sl
                trade.exit_ts     = str(ts)
                trade.result      = "loss"
                trade.r_multiple  = -1.0
                trade.pnl_pips    = round((ep - sl) / pip_size, 1)
                break
            if row["low"] <= tp:
                trade.exit_price  = tp
                trade.exit_ts     = str(ts)
                trade.result      = "win"
                trade.r_multiple  = rr
                trade.pnl_pips    = round((ep - tp) / pip_size, 1)
                break

    return trade


def _equity_curve(trades: List[Trade], start: float = 10_000.0,
                  risk_pct: float = 1.0) -> List[float]:
    eq    = start
    curve = [eq]
    for t in trades:
        if t.result == "win":
            eq += eq * (risk_pct / 100) * abs(t.r_multiple)
        elif t.result == "loss":
            eq -= eq * (risk_pct / 100)
        curve.append(round(eq, 2))
    return curve


def _max_drawdown(curve: List[float]) -> float:
    peak = curve[0]
    dd   = 0.0
    for v in curve:
        if v > peak:
            peak = v
        dd = max(dd, (peak - v) / peak)
    return round(dd * 100, 2)


# ── FastAPI route handler ─────────────────────────────────────────────────────

async def run_backtest(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Called from main FastAPI app.  Orchestrates the full pipeline.
    """
    symbol       = body.get("symbol", "C:EURUSD")
    window       = body.get("window", "1mo")
    params       = body.get("params", {})
    min_conf     = int(body.get("min_conf", 2))
    risk         = body.get("risk", {})
    max_trades   = risk.get("max_trades", None)

    # 1. Fetch data
    df = _fetch_candles(symbol, window)

    # 2. Score confluences
    setups = score_confluences(df, params, min_confluences=min_conf)

    if max_trades:
        setups = setups[:int(max_trades)]

    # 3. Simulate trades
    trades = [_simulate_trade(s, df, risk) for s in setups]

    # 4. Stats
    closed = [t for t in trades if t.result != "open"]
    wins   = [t for t in closed if t.result == "win"]
    losses = [t for t in closed if t.result == "loss"]

    risk_pct  = float(risk.get("risk_pct", 1.0))
    eq_curve  = _equity_curve(closed, risk_pct=risk_pct)
    max_dd    = _max_drawdown(eq_curve)
    total_r   = round(sum(t.r_multiple for t in closed), 2)
    avg_r     = round(total_r / max(len(closed), 1), 3)
    winrate   = round(len(wins) / max(len(closed), 1) * 100, 1)

    # Confluence frequency
    conf_stats = summarise(setups)

    return {
        "symbol":       symbol,
        "window":       window,
        "total_setups": len(setups),
        "total_trades": len(closed),
        "wins":         len(wins),
        "losses":       len(losses),
        "winrate":      winrate,
        "total_r":      total_r,
        "avg_r":        avg_r,
        "max_drawdown": max_dd,
        "equity_curve": eq_curve,
        "trades": [asdict(t) for t in closed],
        "stats":  conf_stats,
    }


# ── Standalone test (no Polygon key needed — uses synthetic data) ─────────────

if __name__ == "__main__":
    import asyncio, random

    random.seed(42)
    n   = 8000
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 1.0800
    rows  = []
    for i in range(n):
        close += random.gauss(0.00008, 0.0009)
        o = close - abs(random.gauss(0, 0.0003))
        h = max(o, close) + abs(random.gauss(0, 0.0005))
        l = min(o, close) - abs(random.gauss(0, 0.0005))
        rows.append({"open": o, "high": h, "low": l,
                     "close": close, "volume": random.randint(100, 2000)})
    synthetic_df = pd.DataFrame(rows, index=idx)

    # Patch fetch to use synthetic data
    def _fake_fetch(symbol, window, **kw):
        return synthetic_df
    import backtest_runner as br
    br._fetch_candles = _fake_fetch

    body = {
        "symbol":   "C:EURUSD",
        "window":   "1mo",
        "min_conf": 2,
        "params": {
            "trend": {"Bias": "Both",      "Timeframe": "4H"},
            "bos":   {"Direction": "Both", "TF": "1H"},
            "fvg":   {"Type": "Both",      "TF": "15m",
                      "Status": "Any",     "CE (50%)": "Any"},
            "ob":    {"Type": "Both",      "TF": "15m",
                      "Mitigation": "Wick","Volume": "Any"},
        },
        "risk": {"rr": 2.0, "sl_pips": 15, "risk_pct": 1.0},
    }

    result = asyncio.run(run_backtest(body))
    print(f"\n{'─'*40}")
    print(f"Symbol      : {result['symbol']}")
    print(f"Setups      : {result['total_setups']}")
    print(f"Trades      : {result['total_trades']}")
    print(f"Win Rate    : {result['winrate']}%")
    print(f"Total R     : {result['total_r']}R")
    print(f"Max DD      : {result['max_drawdown']}%")
    print(f"Avg Score   : {result['stats'].get('avg_score', 'N/A')}")

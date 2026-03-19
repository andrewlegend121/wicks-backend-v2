"""
trend_detector.py
─────────────────
ICT Market Trend Bias — HTF directional analysis.

Logic:
  Swing highs / lows are identified via a rolling pivot window.
  Bullish trend  = series of Higher Highs + Higher Lows (HH/HL).
  Bearish trend  = series of Lower Highs + Lower Lows (LH/LL).
  Neutral/ranging = mixed structure.

Supported timeframes: 1H | 4H | Daily
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, List
import pandas as pd
import numpy as np


TrendBias  = Literal["bullish", "bearish", "neutral"]
TrendTF    = Literal["1H", "4H", "Daily"]


@dataclass
class SwingPoint:
    timestamp: pd.Timestamp
    price: float
    kind: Literal["HH", "HL", "LH", "LL"]


@dataclass
class TrendSignal:
    timestamp: pd.Timestamp
    timeframe: TrendTF
    bias: TrendBias
    swing_points: List[SwingPoint] = field(default_factory=list)
    confidence: float = 0.0          # 0–1 based on how many consecutive pivots align
    confirmed: bool = False


# ── helpers ──────────────────────────────────────────────────────────────────

def _resample(df: pd.DataFrame, tf: TrendTF) -> pd.DataFrame:
    rule_map = {"1H": "1h", "4H": "4h", "Daily": "1D"}
    rule = rule_map[tf]
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    return df.resample(rule).agg(agg).dropna()


def _find_swings(df: pd.DataFrame, pivot_window: int = 3) -> pd.DataFrame:
    """Tag each candle as pivot high / pivot low / none."""
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)
    pivot_high = np.zeros(n, dtype=bool)
    pivot_low  = np.zeros(n, dtype=bool)

    for i in range(pivot_window, n - pivot_window):
        window_h = highs[i - pivot_window: i + pivot_window + 1]
        window_l = lows [i - pivot_window: i + pivot_window + 1]
        if highs[i] == window_h.max():
            pivot_high[i] = True
        if lows[i] == window_l.min():
            pivot_low[i] = True

    df = df.copy()
    df["pivot_high"] = pivot_high
    df["pivot_low"]  = pivot_low
    return df


def _classify_structure(pivot_highs: List[float],
                         pivot_lows: List[float]) -> tuple[TrendBias, float]:
    """
    Given the last N pivot highs and lows, return (bias, confidence 0-1).
    confidence = fraction of pivot-to-pivot moves that agree with bias.
    """
    if len(pivot_highs) < 2 or len(pivot_lows) < 2:
        return "neutral", 0.0

    # Compare consecutive pivot highs and lows
    hh_count = sum(1 for a, b in zip(pivot_highs, pivot_highs[1:]) if b > a)
    hl_count = sum(1 for a, b in zip(pivot_lows,  pivot_lows[1:])  if b > a)
    lh_count = sum(1 for a, b in zip(pivot_highs, pivot_highs[1:]) if b < a)
    ll_count = sum(1 for a, b in zip(pivot_lows,  pivot_lows[1:])  if b < a)

    bull_score = hh_count + hl_count
    bear_score = lh_count + ll_count
    total      = max(bull_score + bear_score, 1)

    if bull_score > bear_score:
        return "bullish", round(bull_score / total, 2)
    elif bear_score > bull_score:
        return "bearish", round(bear_score / total, 2)
    return "neutral", 0.5


# ── public API ────────────────────────────────────────────────────────────────

def detect_trend(
    df: pd.DataFrame,          # 1-min base OHLCV, DatetimeIndex UTC
    timeframe: TrendTF = "4H",
    bias_filter: Literal["Bullish", "Bearish", "Both"] = "Both",
    pivot_window: int = 3,
    lookback_pivots: int = 6,  # how many recent pivots to score
) -> List[TrendSignal]:
    """
    Returns a list of TrendSignal objects, one per timeframe candle where
    the structural bias matches bias_filter.
    """
    tf_df   = _resample(df, timeframe)
    tf_df   = _find_swings(tf_df, pivot_window)
    signals: List[TrendSignal] = []

    ph_prices: List[float] = []
    pl_prices: List[float] = []
    ph_times:  List[pd.Timestamp] = []
    pl_times:  List[pd.Timestamp] = []

    for ts, row in tf_df.iterrows():
        if row["pivot_high"]:
            ph_prices.append(row["high"])
            ph_times.append(ts)
        if row["pivot_low"]:
            pl_prices.append(row["low"])
            pl_times.append(ts)

        # Only classify once we have enough pivots
        if len(ph_prices) < 2 or len(pl_prices) < 2:
            continue

        recent_highs = ph_prices[-lookback_pivots:]
        recent_lows  = pl_prices[-lookback_pivots:]
        bias, conf   = _classify_structure(recent_highs, recent_lows)

        # Apply bias filter
        wanted = {"bullish", "bearish"} if bias_filter == "Both" else {bias_filter.lower()}
        if bias not in wanted:
            continue

        # Build swing points for context
        swings: List[SwingPoint] = []
        for t, p in zip(ph_times[-3:], ph_prices[-3:]):
            kind = "HH" if len(ph_prices) > 1 and p > ph_prices[-2] else "LH"
            swings.append(SwingPoint(t, p, kind))
        for t, p in zip(pl_times[-3:], pl_prices[-3:]):
            kind = "HL" if len(pl_prices) > 1 and p > pl_prices[-2] else "LL"
            swings.append(SwingPoint(t, p, kind))

        signals.append(TrendSignal(
            timestamp  = ts,
            timeframe  = timeframe,
            bias       = bias,
            swing_points = sorted(swings, key=lambda s: s.timestamp),
            confidence = conf,
            confirmed  = conf >= 0.6,
        ))

    return signals


def get_current_bias(
    df: pd.DataFrame,
    timeframe: TrendTF = "4H",
    pivot_window: int = 3,
) -> tuple[TrendBias, float]:
    """Convenience: return current bias + confidence for the latest candle."""
    sigs = detect_trend(df, timeframe, "Both", pivot_window)
    if not sigs:
        return "neutral", 0.0
    last = sigs[-1]
    return last.bias, last.confidence


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random, datetime
    random.seed(42)

    # Generate synthetic 1-min OHLCV trending upward
    n   = 1000
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 1.0800
    rows  = []
    for i in range(n):
        close += random.gauss(0.0002, 0.0008)   # slight upward drift
        o = close - abs(random.gauss(0, 0.0003))
        h = max(o, close) + abs(random.gauss(0, 0.0003))
        l = min(o, close) - abs(random.gauss(0, 0.0003))
        rows.append({"open": o, "high": h, "low": l,
                     "close": close, "volume": random.randint(100, 1000)})

    df = pd.DataFrame(rows, index=idx)

    for tf in ("1H", "4H", "Daily"):
        bias, conf = get_current_bias(df, tf)
        print(f"[{tf}] bias={bias}  confidence={conf:.0%}")

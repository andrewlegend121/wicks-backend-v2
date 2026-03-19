"""
liquidity_detector.py
─────────────────────
ICT Liquidity confluence detection:

  1. Liquidity Sweep
     Price spikes beyond equal highs / equal lows (EQH/EQL) or
     a prior swing high/low, then sharply reverses on the same candle.
     Buy-side  sweep — spike above buy-side liquidity (prior highs)
     Sell-side sweep — spike below sell-side liquidity (prior lows)

  2. Engineered Liquidity
     Price creates a DELIBERATE inducement move to lure in traders,
     then sweeps the liquidity pool and reverses.

     Subtypes:
       Inducement   — a small swing (lower high / higher low) forms after
                      a BOS; price hunts it before continuing.
       False Break  — price closes BEYOND a level then immediately reverses
                      (one-candle false break).
       Stop Hunt    — price spikes into a known stop cluster
                      (equal highs / lows ± tolerance) then reverses.

Timeframes: 15m | 1H | 4H
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, List, Optional
import pandas as pd
import numpy as np


LiqType  = Literal["Buy-side", "Sell-side", "Both"]
EngType  = Literal["Inducement", "False break", "Stop hunt"]
LiqTF    = Literal["15m", "1H", "4H"]


# ── data models ───────────────────────────────────────────────────────────────

@dataclass
class LiquiditySweep:
    timestamp: pd.Timestamp
    timeframe: LiqTF
    sweep_type: Literal["buy-side", "sell-side"]
    swept_level: float          # the liquidity pool price
    wick_extent: float          # how far price went beyond the level
    reversal_magnitude: float   # close-to-level distance (strength of reversal)
    confirmed: bool = True


@dataclass
class EngineeredLiquidity:
    timestamp: pd.Timestamp
    timeframe: LiqTF
    eng_type: Literal["inducement", "false_break", "stop_hunt"]
    direction: Literal["bullish", "bearish"]
    trap_level: float
    sweep_candle: pd.Timestamp
    confirmed: bool = True


# ── helpers ───────────────────────────────────────────────────────────────────

def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = {"15m": "15min", "1H": "1h", "4H": "4h"}[tf]
    return df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna()


def _equal_levels(series: pd.Series,
                  tolerance: float) -> List[float]:
    """Find clusters of 'equal' values within tolerance → return cluster mean."""
    vals   = series.dropna().values
    groups: List[list] = []
    for v in vals:
        placed = False
        for g in groups:
            if abs(v - np.mean(g)) <= tolerance:
                g.append(v); placed = True; break
        if not placed:
            groups.append([v])
    return [float(np.mean(g)) for g in groups if len(g) >= 2]


# ════════════════════════════════════════════════════════════════════════════
# LIQUIDITY SWEEP
# ════════════════════════════════════════════════════════════════════════════

def detect_liquidity_sweeps(
    df: pd.DataFrame,
    timeframe: LiqTF = "1H",
    sweep_type: LiqType = "Both",
    pivot_window: int = 3,
    min_reversal_ratio: float = 0.5,  # reversal must recover ≥50% of sweep wick
) -> List[LiquiditySweep]:
    """
    Identify candles where price sweeps a prior swing high/low then reverses.
    Classic ICT sweep: wick pokes through the level; body closes BACK inside.
    """
    tf_df  = _resample(df, timeframe)
    n      = len(tf_df)
    signals: List[LiquiditySweep] = []

    highs  = tf_df["high"].values
    lows   = tf_df["low"].values
    opens  = tf_df["open"].values
    closes = tf_df["close"].values
    idx    = tf_df.index

    # Rolling swing highs / lows (simple lookback)
    for i in range(pivot_window * 2, n):
        atr_approx = np.mean(highs[i - 14:i] - lows[i - 14:i]) if i >= 14 else 0.001

        # Prior swing high = max of window before current
        prior_high = highs[i - pivot_window * 2: i - pivot_window].max()
        prior_low  = lows [i - pivot_window * 2: i - pivot_window].min()

        # ── Buy-side sweep (spike above prior high, close below it) ─────
        if sweep_type in ("Buy-side", "Both"):
            if highs[i] > prior_high and closes[i] < prior_high:
                wick      = highs[i] - prior_high
                reversal  = prior_high - closes[i]
                if wick > 0 and reversal / (wick + reversal) >= min_reversal_ratio:
                    signals.append(LiquiditySweep(
                        timestamp           = idx[i],
                        timeframe           = timeframe,
                        sweep_type          = "buy-side",
                        swept_level         = prior_high,
                        wick_extent         = wick,
                        reversal_magnitude  = reversal,
                    ))

        # ── Sell-side sweep (spike below prior low, close above it) ─────
        if sweep_type in ("Sell-side", "Both"):
            if lows[i] < prior_low and closes[i] > prior_low:
                wick      = prior_low - lows[i]
                reversal  = closes[i] - prior_low
                if wick > 0 and reversal / (wick + reversal) >= min_reversal_ratio:
                    signals.append(LiquiditySweep(
                        timestamp           = idx[i],
                        timeframe           = timeframe,
                        sweep_type          = "sell-side",
                        swept_level         = prior_low,
                        wick_extent         = wick,
                        reversal_magnitude  = reversal,
                    ))

    return signals


# ════════════════════════════════════════════════════════════════════════════
# ENGINEERED LIQUIDITY
# ════════════════════════════════════════════════════════════════════════════

def detect_engineered_liquidity(
    df: pd.DataFrame,
    timeframe: LiqTF = "1H",
    eng_type: EngType = "Inducement",
    pivot_window: int = 3,
    eql_tolerance_mult: float = 0.15,  # × ATR to consider levels "equal"
    atr_period: int = 14,
) -> List[EngineeredLiquidity]:
    """
    Detect engineered liquidity traps.

    Inducement:
      After a BOS, price creates a small opposing swing.
      We detect the inducement when price sweeps that opposing swing
      and then closes back in the original BOS direction.

    False Break:
      Candle closes beyond a recent extreme then next candle
      immediately closes back inside.

    Stop Hunt:
      Price spike into equal highs/lows (EQH/EQL) followed by reversal.
    """
    tf_df = _resample(df, timeframe).copy()
    n     = len(tf_df)
    if n < atr_period + pivot_window * 2 + 4:
        return []

    tr = pd.concat([
        tf_df["high"] - tf_df["low"],
        (tf_df["high"] - tf_df["close"].shift()).abs(),
        (tf_df["low"]  - tf_df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    tf_df["atr"] = tr.rolling(atr_period, min_periods=1).mean()

    signals: List[EngineeredLiquidity] = []
    idx = tf_df.index

    # ── Inducement ────────────────────────────────────────────────────────
    if eng_type == "Inducement":
        # After each swing break, look for a re-test inducement swing
        for i in range(pivot_window * 2 + 2, n - 2):
            atr_i  = tf_df["atr"].iloc[i]
            w_high = tf_df["high"].iloc[i - pivot_window * 2: i].max()
            w_low  = tf_df["low"].iloc[i - pivot_window * 2: i].min()

            cl = tf_df["close"].iloc[i]
            # Bullish BOS zone
            if cl > w_high:
                # Look for a lower high forming after BOS (inducement)
                future_2 = tf_df.iloc[i + 1: i + 3]
                if len(future_2) < 2:
                    continue
                induced_low = future_2["low"].min()
                induced_ts  = future_2["low"].idxmin()
                # Price then sweeps that induced low and reverses
                if future_2["close"].iloc[-1] > cl * 0.9999:   # still bullish close
                    signals.append(EngineeredLiquidity(
                        timestamp   = idx[i],
                        timeframe   = timeframe,
                        eng_type    = "inducement",
                        direction   = "bullish",
                        trap_level  = induced_low,
                        sweep_candle = induced_ts,
                    ))
            elif cl < w_low:
                future_2 = tf_df.iloc[i + 1: i + 3]
                if len(future_2) < 2:
                    continue
                induced_high = future_2["high"].max()
                induced_ts   = future_2["high"].idxmax()
                if future_2["close"].iloc[-1] < cl * 1.0001:
                    signals.append(EngineeredLiquidity(
                        timestamp   = idx[i],
                        timeframe   = timeframe,
                        eng_type    = "inducement",
                        direction   = "bearish",
                        trap_level  = induced_high,
                        sweep_candle = induced_ts,
                    ))

    # ── False Break ───────────────────────────────────────────────────────
    elif eng_type == "False break":
        for i in range(pivot_window + 1, n - 1):
            prior_high = tf_df["high"].iloc[i - pivot_window: i].max()
            prior_low  = tf_df["low"].iloc[i - pivot_window: i].min()
            row        = tf_df.iloc[i]
            nxt        = tf_df.iloc[i + 1]
            atr_i      = tf_df["atr"].iloc[i]

            # Bearish false break: closes above, next closes back below
            if (row["close"] > prior_high and
                    nxt["close"] < prior_high and
                    nxt["close"] < row["close"] - atr_i * 0.3):
                signals.append(EngineeredLiquidity(
                    timestamp   = nxt.name,
                    timeframe   = timeframe,
                    eng_type    = "false_break",
                    direction   = "bearish",
                    trap_level  = prior_high,
                    sweep_candle = row.name,
                ))

            # Bullish false break: closes below, next closes back above
            if (row["close"] < prior_low and
                    nxt["close"] > prior_low and
                    nxt["close"] > row["close"] + atr_i * 0.3):
                signals.append(EngineeredLiquidity(
                    timestamp   = nxt.name,
                    timeframe   = timeframe,
                    eng_type    = "false_break",
                    direction   = "bullish",
                    trap_level  = prior_low,
                    sweep_candle = row.name,
                ))

    # ── Stop Hunt (Equal Highs / Equal Lows) ─────────────────────────────
    elif eng_type == "Stop hunt":
        atr_now = tf_df["atr"].mean()
        tol     = eql_tolerance_mult * atr_now

        eqh = _equal_levels(tf_df["high"], tol)
        eql = _equal_levels(tf_df["low"],  tol)

        for i in range(1, n):
            row   = tf_df.iloc[i]
            atr_i = tf_df["atr"].iloc[i]

            for level in eqh:
                if (row["high"] > level and
                        row["close"] < level and
                        row["high"] - level > atr_i * 0.1):
                    signals.append(EngineeredLiquidity(
                        timestamp    = idx[i],
                        timeframe    = timeframe,
                        eng_type     = "stop_hunt",
                        direction    = "bearish",
                        trap_level   = level,
                        sweep_candle = idx[i],
                    ))
                    break

            for level in eql:
                if (row["low"] < level and
                        row["close"] > level and
                        level - row["low"] > atr_i * 0.1):
                    signals.append(EngineeredLiquidity(
                        timestamp    = idx[i],
                        timeframe    = timeframe,
                        eng_type     = "stop_hunt",
                        direction    = "bullish",
                        trap_level   = level,
                        sweep_candle = idx[i],
                    ))
                    break

    return signals


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(21)
    n   = 3000
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 1.0800
    rows  = []
    for i in range(n):
        close += random.gauss(0.0001, 0.0009)
        o = close - abs(random.gauss(0, 0.0003))
        h = max(o, close) + abs(random.gauss(0, 0.0005))
        l = min(o, close) - abs(random.gauss(0, 0.0005))
        rows.append({"open": o, "high": h, "low": l,
                     "close": close, "volume": random.randint(100, 2000)})
    df = pd.DataFrame(rows, index=idx)

    for tf in ("15m", "1H"):
        sw  = detect_liquidity_sweeps(df, tf, "Both")
        eng = detect_engineered_liquidity(df, tf, "Stop hunt")
        print(f"[{tf}] Sweeps={len(sw)}  Engineered={len(eng)}")

"""
structure_detector.py
─────────────────────
ICT Market Structure confluences:

  1. Break of Structure (BOS)
     Bullish : price closes above the most recent swing high
     Bearish : price closes below the most recent swing low

  2. Change of Character (CHoCH)
     The FIRST BOS against the prevailing trend —
     signals a possible trend reversal.
     Bullish CHoCH : bearish trend broken upward
     Bearish CHoCH : bullish trend broken downward

  3. Fail to Displace (FTD)
     Candle wicks beyond a key level (High / Low / OB / FVG)
     but body does NOT close beyond it — displacement failed.

BOS timeframes : 15m | 1H | 4H
CHoCH timeframes: 5m  | 15m | 1H
FTD timeframes :  5m  | 15m | 1H
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, List, Optional
import pandas as pd
import numpy as np


Direction  = Literal["bullish", "bearish", "both"]
BOSTF      = Literal["15m", "1H", "4H"]
CHoCHTF    = Literal["5m", "15m", "1H"]
FTDTF      = Literal["5m", "15m", "1H"]
FTDLevel   = Literal["High", "Low", "OB", "FVG"]


# ── data models ───────────────────────────────────────────────────────────────

@dataclass
class BOSSignal:
    timestamp: pd.Timestamp
    timeframe: BOSTF
    direction: Literal["bullish", "bearish"]
    broken_level: float          # the swing high/low that was broken
    close_price: float
    confirmed: bool = True


@dataclass
class CHoCHSignal:
    timestamp: pd.Timestamp
    timeframe: CHoCHTF
    direction: Literal["bullish", "bearish"]
    broken_level: float
    prior_trend: Literal["bullish", "bearish"]
    confirmed: bool = True


@dataclass
class FTDSignal:
    timestamp: pd.Timestamp
    timeframe: FTDTF
    direction: Literal["bullish", "bearish"]   # which way price tried to displace
    level_type: FTDLevel
    level_price: float
    wick_beyond: float    # how far the wick went past the level
    confirmed: bool = True


# ── helpers ───────────────────────────────────────────────────────────────────

def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule_map = {"5m": "5min", "15m": "15min", "1H": "1h", "4H": "4h"}
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    return df.resample(rule_map[tf]).agg(agg).dropna()


def _swing_highs_lows(series_high: pd.Series, series_low: pd.Series,
                      window: int = 3):
    """Return index arrays of pivot highs and lows."""
    n  = len(series_high)
    ph = []   # (iloc, price)
    pl = []
    for i in range(window, n - window):
        h_win = series_high.iloc[i - window: i + window + 1].values
        l_win = series_low.iloc[i - window: i + window + 1].values
        if series_high.iloc[i] == h_win.max():
            ph.append((i, series_high.iloc[i]))
        if series_low.iloc[i] == l_win.min():
            pl.append((i, series_low.iloc[i]))
    return ph, pl


# ── BOS ───────────────────────────────────────────────────────────────────────

def detect_bos(
    df: pd.DataFrame,
    timeframe: BOSTF = "1H",
    direction: Literal["Bullish", "Bearish", "Both"] = "Both",
    pivot_window: int = 3,
) -> List[BOSSignal]:
    """
    Scan resampled OHLCV for Break of Structure events.
    A BOS is confirmed on the CLOSE of the breaking candle.
    """
    tf_df   = _resample(df, timeframe)
    ph, pl  = _swing_highs_lows(tf_df["high"], tf_df["low"], pivot_window)
    signals: List[BOSSignal] = []

    idx  = tf_df.index
    cl   = tf_df["close"].values
    high = tf_df["high"].values
    low  = tf_df["low"].values

    # Bullish BOS: close above last swing high
    if direction in ("Bullish", "Both"):
        last_ph_idx, last_ph_price = None, None
        ph_ptr = 0
        for i in range(len(tf_df)):
            # advance swing high pointer to all pivots up to (but not including) i
            while ph_ptr < len(ph) and ph[ph_ptr][0] < i:
                last_ph_idx, last_ph_price = ph[ph_ptr]
                ph_ptr += 1
            if last_ph_price is not None and cl[i] > last_ph_price:
                signals.append(BOSSignal(
                    timestamp     = idx[i],
                    timeframe     = timeframe,
                    direction     = "bullish",
                    broken_level  = last_ph_price,
                    close_price   = cl[i],
                ))
                last_ph_price = None   # reset — BOS consumed this level

    # Bearish BOS: close below last swing low
    if direction in ("Bearish", "Both"):
        last_pl_idx, last_pl_price = None, None
        pl_ptr = 0
        for i in range(len(tf_df)):
            while pl_ptr < len(pl) and pl[pl_ptr][0] < i:
                last_pl_idx, last_pl_price = pl[pl_ptr]
                pl_ptr += 1
            if last_pl_price is not None and cl[i] < last_pl_price:
                signals.append(BOSSignal(
                    timestamp    = idx[i],
                    timeframe    = timeframe,
                    direction    = "bearish",
                    broken_level = last_pl_price,
                    close_price  = cl[i],
                ))
                last_pl_price = None

    signals.sort(key=lambda s: s.timestamp)
    return signals


# ── CHoCH ─────────────────────────────────────────────────────────────────────

def detect_choch(
    df: pd.DataFrame,
    timeframe: CHoCHTF = "15m",
    direction: Literal["Bullish", "Bearish", "Both"] = "Both",
    pivot_window: int = 3,
) -> List[CHoCHSignal]:
    """
    CHoCH = the FIRST BOS that contradicts the prior trend.
    We track the running trend; when a BOS fires in the opposite direction,
    it is a CHoCH.
    """
    bos_all = detect_bos(df, timeframe, "Both", pivot_window)  # type: ignore
    if not bos_all:
        return []

    signals: List[CHoCHSignal] = []
    prior_trend: Optional[Literal["bullish", "bearish"]] = None

    for bos in bos_all:
        if prior_trend is None:
            prior_trend = bos.direction
            continue

        is_choch = bos.direction != prior_trend
        if is_choch:
            wanted = {"bullish", "bearish"} if direction == "Both" else {direction.lower()}
            if bos.direction in wanted:
                signals.append(CHoCHSignal(
                    timestamp    = bos.timestamp,
                    timeframe    = timeframe,   # type: ignore
                    direction    = bos.direction,
                    broken_level = bos.broken_level,
                    prior_trend  = prior_trend,
                ))
        prior_trend = bos.direction

    return signals


# ── Fail to Displace ──────────────────────────────────────────────────────────

def detect_ftd(
    df: pd.DataFrame,
    timeframe: FTDTF = "15m",
    level_type: FTDLevel = "High",
    direction: Literal["Bullish", "Bearish", "Both"] = "Both",
    pivot_window: int = 3,
    min_wick_pips: float = 0.0002,   # minimum wick extension beyond level
) -> List[FTDSignal]:
    """
    Fail to Displace:
    For High/Low: wick pierces the swing high/low but candle closes back on
                  the other side — absorption / failed displacement.
    For OB/FVG:   pass those zone boundaries in externally; this module
                  uses the nearest swing high/low as a proxy when OB/FVG
                  zones are not provided directly.
    """
    tf_df  = _resample(df, timeframe)
    ph, pl = _swing_highs_lows(tf_df["high"], tf_df["low"], pivot_window)
    signals: List[FTDSignal] = []

    idx   = tf_df.index
    op    = tf_df["open"].values
    hi    = tf_df["high"].values
    lo    = tf_df["low"].values
    cl    = tf_df["close"].values

    # Build lookup: for each candle i, what was the most recent swing high/low
    # before it?
    ph_dict: dict[int, float] = {}
    pl_dict: dict[int, float] = {}
    last_h = last_l = None
    ph_ptr = pl_ptr = 0
    for i in range(len(tf_df)):
        while ph_ptr < len(ph) and ph[ph_ptr][0] < i:
            last_h = ph[ph_ptr][1]; ph_ptr += 1
        while pl_ptr < len(pl) and pl[pl_ptr][0] < i:
            last_l = pl[pl_ptr][1]; pl_ptr += 1
        ph_dict[i] = last_h
        pl_dict[i] = last_l

    for i in range(1, len(tf_df)):
        # --- Bearish FTD (price wicked above swing high but closed below it) ---
        if direction in ("Bearish", "Both") and ph_dict[i] is not None:
            level = ph_dict[i]
            wick_above = hi[i] - level
            if wick_above > min_wick_pips and cl[i] < level:
                signals.append(FTDSignal(
                    timestamp   = idx[i],
                    timeframe   = timeframe,
                    direction   = "bearish",
                    level_type  = level_type,
                    level_price = level,
                    wick_beyond = wick_above,
                ))

        # --- Bullish FTD (price wicked below swing low but closed above it) ---
        if direction in ("Bullish", "Both") and pl_dict[i] is not None:
            level = pl_dict[i]
            wick_below = level - lo[i]
            if wick_below > min_wick_pips and cl[i] > level:
                signals.append(FTDSignal(
                    timestamp   = idx[i],
                    timeframe   = timeframe,
                    direction   = "bullish",
                    level_type  = level_type,
                    level_price = level,
                    wick_beyond = wick_below,
                ))

    return signals


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(7)
    n   = 2000
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 1.0800
    rows  = []
    for i in range(n):
        drift = 0.0003 if i < 1000 else -0.0003
        close += random.gauss(drift, 0.0008)
        o = close - abs(random.gauss(0, 0.0003))
        h = max(o, close) + abs(random.gauss(0, 0.0003))
        l = min(o, close) - abs(random.gauss(0, 0.0003))
        rows.append({"open": o, "high": h, "low": l,
                     "close": close, "volume": random.randint(100, 1000)})
    df = pd.DataFrame(rows, index=idx)

    bos   = detect_bos(df, "15m", "Both")
    choch = detect_choch(df, "15m", "Both")
    ftd   = detect_ftd(df, "15m", "High", "Both")

    print(f"BOS signals   : {len(bos)}")
    print(f"CHoCH signals : {len(choch)}")
    print(f"FTD signals   : {len(ftd)}")

    if bos:
        print(f"  Last BOS  → {bos[-1].direction} @ {bos[-1].timestamp}")
    if choch:
        print(f"  Last CHoCH→ {choch[-1].direction} @ {choch[-1].timestamp}")

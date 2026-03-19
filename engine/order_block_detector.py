"""
order_block_detector.py
───────────────────────
ICT Order Block detection with full mitigation + volume classification.

Definition:
  Bullish OB  — the last BEARISH candle before a bullish impulse leg
                that breaks structure (BOS) upward.
  Bearish OB  — the last BULLISH candle before a bearish impulse leg
                that breaks structure (BOS) downward.

Mitigation modes (when price returns to the OB zone):
  Wick    — any wick penetrates the OB zone
  Body50% — candle body reaches at least 50 % into the OB
  Close   — candle closes within the OB zone

Volume classification:
  High    — OB candle volume > 1.5× 20-bar SMA
  Normal  — OB candle volume 0.8–1.5× SMA
  Any     — no filter

Timeframes: 5m | 15m | 1H | 4H
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, List, Optional, Tuple
import pandas as pd
import numpy as np

OBType      = Literal["Bullish", "Bearish", "Both"]
OBTimeframe = Literal["5m", "15m", "1H", "4H"]
Mitigation  = Literal["Wick", "Body 50%", "Close"]
VolumeClass = Literal["High", "Normal", "Any"]


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class OrderBlock:
    timestamp: pd.Timestamp          # candle that formed the OB
    timeframe: OBTimeframe
    ob_type: Literal["bullish", "bearish"]
    zone_high: float                  # OB upper edge
    zone_low: float                   # OB lower edge
    zone_50pct: float                 # midpoint
    volume: float
    volume_class: Literal["high", "normal"]
    mitigated: bool = False
    mitigation_timestamp: Optional[pd.Timestamp] = None
    mitigation_method: Optional[str] = None


# ── helpers ───────────────────────────────────────────────────────────────────

def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = {"5m": "5min", "15m": "15min", "1H": "1h", "4H": "4h"}[tf]
    return df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna()


def _classify_volume(vol: float, sma_vol: float) -> Literal["high", "normal"]:
    return "high" if vol > sma_vol * 1.5 else "normal"


def _is_bullish(row) -> bool:
    return row["close"] > row["open"]


def _is_bearish(row) -> bool:
    return row["close"] < row["open"]


def _impulse_strength(df: pd.DataFrame, start_i: int, end_i: int) -> float:
    """Net price change of an impulse segment (positive = bullish)."""
    return df["close"].iloc[end_i] - df["close"].iloc[start_i]


# ── mitigation check ──────────────────────────────────────────────────────────

def _check_mitigation(ob: OrderBlock, row: pd.Series,
                      mode: Mitigation) -> bool:
    if ob.ob_type == "bullish":
        # Price returning DOWN into bullish OB zone
        if mode == "Wick":
            return row["low"] <= ob.zone_high and row["low"] >= ob.zone_low - (ob.zone_high - ob.zone_low)
        elif mode == "Body 50%":
            body_low = min(row["open"], row["close"])
            return body_low <= ob.zone_50pct
        else:  # Close
            return row["close"] >= ob.zone_low and row["close"] <= ob.zone_high
    else:
        # Price returning UP into bearish OB zone
        if mode == "Wick":
            return row["high"] >= ob.zone_low and row["high"] <= ob.zone_high + (ob.zone_high - ob.zone_low)
        elif mode == "Body 50%":
            body_high = max(row["open"], row["close"])
            return body_high >= ob.zone_50pct
        else:  # Close
            return row["close"] >= ob.zone_low and row["close"] <= ob.zone_high


# ── main detector ─────────────────────────────────────────────────────────────

def detect_order_blocks(
    df: pd.DataFrame,
    timeframe: OBTimeframe = "1H",
    ob_type: OBType = "Both",
    mitigation: Mitigation = "Wick",
    volume_filter: VolumeClass = "Any",
    impulse_candles: int = 3,     # how many candles define the impulse leg
    vol_sma_period: int = 20,
) -> List[OrderBlock]:
    """
    Detect order blocks and tag mitigation events on the same TF data.

    Returns all detected OBs (mitigated and unmitigated).
    """
    tf_df  = _resample(df, timeframe).reset_index()
    n      = len(tf_df)
    if n < vol_sma_period + impulse_candles + 2:
        return []

    # Volume SMA
    tf_df["vol_sma"] = (
        tf_df["volume"]
        .rolling(vol_sma_period, min_periods=1)
        .mean()
    )

    obs: List[OrderBlock] = []

    for i in range(1, n - impulse_candles):
        row     = tf_df.iloc[i]
        vol_sma = tf_df["vol_sma"].iloc[i]
        v_class = _classify_volume(row["volume"], vol_sma)

        # Volume gate
        if volume_filter == "High"   and v_class != "high":   continue
        if volume_filter == "Normal" and v_class != "normal":  continue

        # ── Bullish OB: last bearish candle before bullish impulse ───
        if ob_type in ("Bullish", "Both") and _is_bearish(row):
            future = tf_df.iloc[i + 1: i + 1 + impulse_candles]
            # Impulse: majority bullish, total net positive
            bull_count = sum(1 for _, r in future.iterrows() if _is_bullish(r))
            net_move   = future["close"].iloc[-1] - future["open"].iloc[0]
            if bull_count >= max(1, impulse_candles // 2) and net_move > 0:
                obs.append(OrderBlock(
                    timestamp    = row["timestamp"],
                    timeframe    = timeframe,
                    ob_type      = "bullish",
                    zone_high    = row["high"],
                    zone_low     = row["low"],
                    zone_50pct   = (row["high"] + row["low"]) / 2,
                    volume       = row["volume"],
                    volume_class = v_class,
                ))

        # ── Bearish OB: last bullish candle before bearish impulse ───
        if ob_type in ("Bearish", "Both") and _is_bullish(row):
            future = tf_df.iloc[i + 1: i + 1 + impulse_candles]
            bear_count = sum(1 for _, r in future.iterrows() if _is_bearish(r))
            net_move   = future["close"].iloc[-1] - future["open"].iloc[0]
            if bear_count >= max(1, impulse_candles // 2) and net_move < 0:
                obs.append(OrderBlock(
                    timestamp    = row["timestamp"],
                    timeframe    = timeframe,
                    ob_type      = "bearish",
                    zone_high    = row["high"],
                    zone_low     = row["low"],
                    zone_50pct   = (row["high"] + row["low"]) / 2,
                    volume       = row["volume"],
                    volume_class = v_class,
                ))

    # ── Mitigation pass ──────────────────────────────────────────────
    tf_df_indexed = tf_df.set_index("timestamp")
    for ob in obs:
        try:
            fut = tf_df_indexed.loc[tf_df_indexed.index > ob.timestamp]
        except Exception:
            continue
        for ts, row in fut.iterrows():
            if _check_mitigation(ob, row, mitigation):
                ob.mitigated             = True
                ob.mitigation_timestamp  = ts
                ob.mitigation_method     = mitigation
                break

    return obs


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(11)
    n   = 3000
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 1.0800
    rows  = []
    for i in range(n):
        close += random.gauss(0.0001, 0.0009)
        o = close - abs(random.gauss(0, 0.0003))
        h = max(o, close) + abs(random.gauss(0, 0.0004))
        l = min(o, close) - abs(random.gauss(0, 0.0004))
        rows.append({"open": o, "high": h, "low": l,
                     "close": close, "volume": random.randint(100, 2000)})
    df = pd.DataFrame(rows, index=idx)

    for tf in ("5m", "15m", "1H"):
        obs = detect_order_blocks(df, tf, "Both", "Wick", "Any")
        mit = [o for o in obs if o.mitigated]
        print(f"[{tf}] OBs={len(obs)}  mitigated={len(mit)}")

"""
blocks_detector.py
──────────────────
Breaker Block, Propulsion Block, and Rejection Block detectors.

━━ BREAKER BLOCK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
An Order Block that was FULLY mitigated (price closed through it completely).
The zone FLIPS:
  Bullish Breaker — was a bearish OB; price broke through it upward;
                    zone now acts as support on re-test.
  Bearish Breaker — was a bullish OB; price broke through it downward;
                    zone now acts as resistance on re-test.

Mitigation modes: Wick | Body 50% | Close
Timeframes: 15m | 1H | 4H

━━ PROPULSION BLOCK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The consolidation / "base" range that forms just BEFORE a strong impulse.
When price returns to that base it often re-accelerates in the original direction.

Logic:
  1. Detect a tight-range consolidation (ATR-based — bars whose range is
     < ATR_mult × ATR).
  2. Confirm an impulse exits the consolidation (body closes beyond range).
  3. The consolidation range = propulsion zone.

Timeframes: 15m | 1H | 4H

━━ REJECTION BLOCK ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Repeated wick rejections at the SAME price zone (within a tolerance band).
  Min wicks: 2 | 3 | 4+

Timeframes: 5m | 15m | 1H | 4H
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, List, Optional
import pandas as pd
import numpy as np

from order_block_detector import (
    detect_order_blocks, OrderBlock, Mitigation, VolumeClass, _resample
)


BlockType = Literal["Bullish", "Bearish", "Both"]
BBTF      = Literal["15m", "1H", "4H"]
PBTF      = Literal["15m", "1H", "4H"]
RBTF      = Literal["5m", "15m", "1H", "4H"]
MinWicks  = Literal["2", "3", "4+"]


# ── data models ───────────────────────────────────────────────────────────────

@dataclass
class BreakerBlock:
    timestamp: pd.Timestamp
    timeframe: BBTF
    bb_type: Literal["bullish", "bearish"]
    zone_high: float
    zone_low: float
    zone_50pct: float
    origin_ob_timestamp: pd.Timestamp
    mitigated: bool = False
    mitigation_timestamp: Optional[pd.Timestamp] = None
    mitigation_method: Optional[str] = None


@dataclass
class PropulsionBlock:
    timestamp: pd.Timestamp          # first candle of consolidation range
    timeframe: PBTF
    pb_type: Literal["bullish", "bearish"]    # direction of the ensuing impulse
    zone_high: float
    zone_low: float
    zone_mid: float
    impulse_start: pd.Timestamp
    impulse_magnitude: float          # % move of the impulse leg


@dataclass
class RejectionBlock:
    timestamp: pd.Timestamp           # timestamp of first wick rejection
    timeframe: RBTF
    rb_type: Literal["bullish", "bearish"]
    zone_price: float                  # approximate rejection level
    wick_count: int
    first_wick: pd.Timestamp
    last_wick: pd.Timestamp


# ════════════════════════════════════════════════════════════════════════════
# BREAKER BLOCK
# ════════════════════════════════════════════════════════════════════════════

def detect_breaker_blocks(
    df: pd.DataFrame,
    timeframe: BBTF = "1H",
    bb_type: BlockType = "Both",
    mitigation: Mitigation = "Wick",
    volume_filter: VolumeClass = "Any",
    impulse_candles: int = 3,
) -> List[BreakerBlock]:
    """
    Detect breaker blocks by finding OBs that were FULLY broken through.

    A bearish OB (high/low) is 'fully broken' when price closes ABOVE
    the OB high => flips to bullish breaker.
    A bullish OB (high/low) is 'fully broken' when price closes BELOW
    the OB low => flips to bearish breaker.
    """
    # First get all OBs (no mitigation filter — we want to check breakage)
    obs = detect_order_blocks(df, timeframe, "Both",  # type: ignore
                               "Wick", volume_filter, impulse_candles)
    if not obs:
        return []

    tf_df = _resample(df, timeframe)
    breakers: List[BreakerBlock] = []

    for ob in obs:
        try:
            fut = tf_df.loc[tf_df.index > ob.timestamp]
        except Exception:
            continue

        broken = False
        break_ts = None
        for ts, row in fut.iterrows():
            if ob.ob_type == "bearish" and row["close"] > ob.zone_high:
                broken   = True
                break_ts = ts
                break
            elif ob.ob_type == "bullish" and row["close"] < ob.zone_low:
                broken   = True
                break_ts = ts
                break

        if not broken:
            continue

        # Flip type
        new_type: Literal["bullish", "bearish"] = (
            "bullish" if ob.ob_type == "bearish" else "bearish"
        )

        wanted = {"bullish", "bearish"} if bb_type == "Both" else {bb_type.lower()}
        if new_type not in wanted:
            continue

        bb = BreakerBlock(
            timestamp           = break_ts,
            timeframe           = timeframe,
            bb_type             = new_type,
            zone_high           = ob.zone_high,
            zone_low            = ob.zone_low,
            zone_50pct          = ob.zone_50pct,
            origin_ob_timestamp = ob.timestamp,
        )

        # Mitigation pass (price returns to zone after break)
        try:
            after_break = tf_df.loc[tf_df.index > break_ts]
        except Exception:
            breakers.append(bb)
            continue

        for ts2, row2 in after_break.iterrows():
            in_zone = (row2["low"] <= bb.zone_high and
                       row2["high"] >= bb.zone_low)
            if in_zone:
                if mitigation == "Wick":
                    bb.mitigated            = True
                    bb.mitigation_timestamp = ts2
                    bb.mitigation_method    = "Wick"
                    break
                elif mitigation == "Body 50%":
                    body_high = max(row2["open"], row2["close"])
                    body_low  = min(row2["open"], row2["close"])
                    if new_type == "bullish" and body_low <= bb.zone_50pct:
                        bb.mitigated = True; bb.mitigation_timestamp = ts2; break
                    if new_type == "bearish" and body_high >= bb.zone_50pct:
                        bb.mitigated = True; bb.mitigation_timestamp = ts2; break
                else:  # Close
                    if bb.zone_low <= row2["close"] <= bb.zone_high:
                        bb.mitigated = True; bb.mitigation_timestamp = ts2; break

        breakers.append(bb)

    return breakers


# ════════════════════════════════════════════════════════════════════════════
# PROPULSION BLOCK
# ════════════════════════════════════════════════════════════════════════════

def detect_propulsion_blocks(
    df: pd.DataFrame,
    timeframe: PBTF = "1H",
    pb_type: BlockType = "Both",
    atr_period: int = 14,
    consolidation_mult: float = 0.6,   # range must be < mult × ATR
    consolidation_bars: int = 3,       # min bars in tight range
    impulse_mult: float = 1.5,         # impulse bar range must be > mult × ATR
) -> List[PropulsionBlock]:
    """
    Find consolidation zones that preceded a strong impulse.
    """
    tf_df = _resample(df, timeframe).copy()
    if len(tf_df) < atr_period + consolidation_bars + 2:
        return []

    # ATR
    tr = pd.concat([
        tf_df["high"] - tf_df["low"],
        (tf_df["high"] - tf_df["close"].shift()).abs(),
        (tf_df["low"]  - tf_df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    tf_df["atr"] = tr.rolling(atr_period, min_periods=1).mean()

    signals: List[PropulsionBlock] = []
    n = len(tf_df)

    for i in range(consolidation_bars, n - 1):
        atr_i = tf_df["atr"].iloc[i]
        if atr_i == 0:
            continue

        # Look-back window for consolidation
        window = tf_df.iloc[i - consolidation_bars: i + 1]
        w_high = window["high"].max()
        w_low  = window["low"].min()
        w_range = w_high - w_low

        if w_range > consolidation_mult * atr_i:
            continue   # range too wide — not a consolidation

        # Next bar must be an impulse
        nxt = tf_df.iloc[i + 1]
        nxt_range = nxt["high"] - nxt["low"]
        if nxt_range < impulse_mult * atr_i:
            continue   # not a strong enough impulse

        # Determine direction of impulse
        nxt_bullish = nxt["close"] > nxt["open"]
        nxt_bearish = nxt["close"] < nxt["open"]

        wanted = {"bullish", "bearish"} if pb_type == "Both" else {pb_type.lower()}
        pb_dir: Optional[Literal["bullish", "bearish"]] = None
        if nxt_bullish and "bullish" in wanted:
            pb_dir = "bullish"
        elif nxt_bearish and "bearish" in wanted:
            pb_dir = "bearish"

        if pb_dir is None:
            continue

        impulse_mag = abs(nxt["close"] - nxt["open"]) / max(w_low, 0.0001) * 100

        signals.append(PropulsionBlock(
            timestamp         = window.index[0],
            timeframe         = timeframe,
            pb_type           = pb_dir,
            zone_high         = w_high,
            zone_low          = w_low,
            zone_mid          = (w_high + w_low) / 2,
            impulse_start     = tf_df.index[i + 1],
            impulse_magnitude = round(impulse_mag, 4),
        ))

    return signals


# ════════════════════════════════════════════════════════════════════════════
# REJECTION BLOCK
# ════════════════════════════════════════════════════════════════════════════

def detect_rejection_blocks(
    df: pd.DataFrame,
    timeframe: RBTF = "15m",
    min_wicks: MinWicks = "3",
    wick_ratio: float = 0.5,    # wick must be > ratio × total candle range
    zone_tolerance_mult: float = 0.3,  # wicks within ± tolerance × ATR of zone
    atr_period: int = 14,
) -> List[RejectionBlock]:
    """
    Identify price zones where repeated wick rejections occurred.
    Bullish RB — repeated lower wicks (demand zone).
    Bearish RB — repeated upper wicks (supply zone).
    """
    tf_df = _resample(df, timeframe).copy()
    if len(tf_df) < atr_period + 2:
        return []

    tr = pd.concat([
        tf_df["high"] - tf_df["low"],
        (tf_df["high"] - tf_df["close"].shift()).abs(),
        (tf_df["low"]  - tf_df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    tf_df["atr"] = tr.rolling(atr_period, min_periods=1).mean()

    min_w = {"2": 2, "3": 3, "4+": 4}[min_wicks]

    # Identify candles with significant upper / lower wicks
    tf_df["body"]       = (tf_df["close"] - tf_df["open"]).abs()
    tf_df["total_rng"]  = tf_df["high"] - tf_df["low"]
    tf_df["upper_wick"] = tf_df["high"] - tf_df[["open", "close"]].max(axis=1)
    tf_df["lower_wick"] = tf_df[["open", "close"]].min(axis=1) - tf_df["low"]

    signals: List[RejectionBlock] = []
    n = len(tf_df)

    # Sliding window — cluster nearby upper/lower wicks
    visited_upper: set = set()
    visited_lower: set = set()

    for i in range(min_w - 1, n):
        atr_i = tf_df["atr"].iloc[i]
        tol   = zone_tolerance_mult * atr_i

        # ── Bearish rejection (upper wicks) ──────────────────────────────
        if i not in visited_upper:
            # Wicks at approximately the same high
            ref_level = tf_df["high"].iloc[i]
            wick_idxs = []
            for j in range(max(0, i - 10), i + 1):
                row_j = tf_df.iloc[j]
                is_upper_wick = (row_j["upper_wick"] >
                                 wick_ratio * max(row_j["total_rng"], 1e-9))
                near_level    = abs(row_j["high"] - ref_level) <= tol
                if is_upper_wick and near_level:
                    wick_idxs.append(j)

            if len(wick_idxs) >= min_w:
                visited_upper.update(wick_idxs)
                first_i = wick_idxs[0]
                last_i  = wick_idxs[-1]
                signals.append(RejectionBlock(
                    timestamp   = tf_df.index[i],
                    timeframe   = timeframe,
                    rb_type     = "bearish",
                    zone_price  = ref_level,
                    wick_count  = len(wick_idxs),
                    first_wick  = tf_df.index[first_i],
                    last_wick   = tf_df.index[last_i],
                ))

        # ── Bullish rejection (lower wicks) ──────────────────────────────
        if i not in visited_lower:
            ref_level = tf_df["low"].iloc[i]
            wick_idxs = []
            for j in range(max(0, i - 10), i + 1):
                row_j = tf_df.iloc[j]
                is_lower_wick = (row_j["lower_wick"] >
                                 wick_ratio * max(row_j["total_rng"], 1e-9))
                near_level    = abs(row_j["low"] - ref_level) <= tol
                if is_lower_wick and near_level:
                    wick_idxs.append(j)

            if len(wick_idxs) >= min_w:
                visited_lower.update(wick_idxs)
                first_i = wick_idxs[0]
                last_i  = wick_idxs[-1]
                signals.append(RejectionBlock(
                    timestamp   = tf_df.index[i],
                    timeframe   = timeframe,
                    rb_type     = "bullish",
                    zone_price  = ref_level,
                    wick_count  = len(wick_idxs),
                    first_wick  = tf_df.index[first_i],
                    last_wick   = tf_df.index[last_i],
                ))

    return signals


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random, sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    random.seed(99)
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

    bb = detect_breaker_blocks(df, "15m", "Both", "Wick")
    pb = detect_propulsion_blocks(df, "15m", "Both")
    rb = detect_rejection_blocks(df, "15m", "3")

    print(f"Breaker Blocks   : {len(bb)}")
    print(f"Propulsion Blocks: {len(pb)}")
    print(f"Rejection Blocks : {len(rb)}")

"""
fvg_detector.py
───────────────
ICT Fair Value Gap (FVG) and Inverse FVG (IFVG) detection.

━━ FAIR VALUE GAP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3-candle pattern with a price imbalance (gap):

  Bullish FVG:
    candle[1].low > candle[-1].high    (candle[-1] is 2 bars back)
    i.e.  low of middle candle > high of first candle
    zone: [candle[-1].high  →  candle[1].low]

  Bearish FVG:
    candle[1].high < candle[-1].low
    zone: [candle[1].high  →  candle[-1].low]

Status tracking (after formation):
  Unfilled  — price has not re-entered the FVG zone
  Partial   — price entered but did not close through the full zone
  Filled    — price closed fully through the zone

CE (Consequent Encroachment / 50% level):
  Respect  — price touched the 50% level but bounced
  Violate  — price closed beyond the 50% level
  Any      — no filter

━━ INVERSE FVG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A FVG that was FULLY filled (price closed through it).
The zone now acts as a magnet in the OPPOSITE direction.

  Bullish IFVG — was a bearish FVG; now acts as support.
  Bearish IFVG — was a bullish FVG; now acts as resistance.

Timeframes: 5m | 15m | 1H | 4H
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, List, Optional
import pandas as pd
import numpy as np


FVGType   = Literal["Bullish", "Bearish", "Both"]
FVGStatus = Literal["Unfilled", "Partial", "Any"]
CEFilter  = Literal["Respect", "Violate", "Any"]
FVGTF     = Literal["5m", "15m", "1H", "4H"]


# ── data models ───────────────────────────────────────────────────────────────

@dataclass
class FVG:
    timestamp: pd.Timestamp         # timestamp of the MIDDLE candle
    timeframe: FVGTF
    fvg_type: Literal["bullish", "bearish"]
    zone_high: float
    zone_low: float
    zone_ce: float                   # 50% level
    status: Literal["unfilled", "partial", "filled"] = "unfilled"
    ce_status: Literal["respected", "violated", "untouched"] = "untouched"
    fill_timestamp: Optional[pd.Timestamp] = None
    ce_timestamp: Optional[pd.Timestamp] = None


@dataclass
class InverseFVG:
    timestamp: pd.Timestamp          # when the original FVG was fully filled
    timeframe: FVGTF
    ifvg_type: Literal["bullish", "bearish"]   # FLIPPED from source FVG
    zone_high: float
    zone_low: float
    zone_ce: float
    status: Literal["unfilled", "partial", "any"] = "unfilled"
    ce_status: Literal["respected", "violated", "untouched"] = "untouched"
    origin_fvg_timestamp: Optional[pd.Timestamp] = None


# ── helpers ───────────────────────────────────────────────────────────────────

def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    rule = {"5m": "5min", "15m": "15min", "1H": "1h", "4H": "4h"}[tf]
    return df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna()


def _update_fvg_status(fvg: FVG, row: pd.Series, ts: pd.Timestamp) -> None:
    """Mutate fvg status and CE status based on a new candle."""
    if fvg.status == "filled":
        return

    # Penetration check
    if fvg.fvg_type == "bullish":
        # Bullish FVG — price should retrace DOWN into zone
        low = min(row["open"], row["close"], row["low"])
        if low <= fvg.zone_low:
            fvg.status = "filled"
            fvg.fill_timestamp = ts
        elif low <= fvg.zone_high:
            fvg.status = "partial"
    else:
        # Bearish FVG — price should retrace UP into zone
        high = max(row["open"], row["close"], row["high"])
        if high >= fvg.zone_high:
            fvg.status = "filled"
            fvg.fill_timestamp = ts
        elif high >= fvg.zone_low:
            fvg.status = "partial"

    # CE check
    if fvg.ce_status == "untouched":
        if fvg.fvg_type == "bullish":
            low_body = min(row["open"], row["close"])
            if row["low"] <= fvg.zone_ce:
                if low_body <= fvg.zone_ce:
                    fvg.ce_status = "violated"
                else:
                    fvg.ce_status = "respected"
                fvg.ce_timestamp = ts
        else:
            high_body = max(row["open"], row["close"])
            if row["high"] >= fvg.zone_ce:
                if high_body >= fvg.zone_ce:
                    fvg.ce_status = "violated"
                else:
                    fvg.ce_status = "respected"
                fvg.ce_timestamp = ts


# ════════════════════════════════════════════════════════════════════════════
# FVG DETECTOR
# ════════════════════════════════════════════════════════════════════════════

def detect_fvg(
    df: pd.DataFrame,
    timeframe: FVGTF = "15m",
    fvg_type: FVGType = "Both",
    status_filter: FVGStatus = "Any",
    ce_filter: CEFilter = "Any",
    min_gap_mult: float = 0.1,   # gap must be > mult × ATR to avoid micro-gaps
    atr_period: int = 14,
) -> List[FVG]:
    """
    Detect all FVGs in the resampled OHLCV data and track their status.
    """
    tf_df = _resample(df, timeframe).copy()
    n     = len(tf_df)
    if n < atr_period + 3:
        return []

    tr = pd.concat([
        tf_df["high"] - tf_df["low"],
        (tf_df["high"] - tf_df["close"].shift()).abs(),
        (tf_df["low"]  - tf_df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    tf_df["atr"] = tr.rolling(atr_period, min_periods=1).mean()

    all_fvgs: List[FVG] = []
    idx = tf_df.index

    # ── Formation pass ────────────────────────────────────────────────────
    for i in range(2, n):
        c0 = tf_df.iloc[i - 2]   # candle n-2
        c1 = tf_df.iloc[i - 1]   # middle candle  (FVG body)
        c2 = tf_df.iloc[i]       # candle n
        atr_i = tf_df["atr"].iloc[i]
        min_gap = min_gap_mult * atr_i

        # Bullish FVG: gap between c0.high and c2.low
        if fvg_type in ("Bullish", "Both"):
            gap = c2["low"] - c0["high"]
            if gap > min_gap:
                all_fvgs.append(FVG(
                    timestamp  = idx[i - 1],
                    timeframe  = timeframe,
                    fvg_type   = "bullish",
                    zone_high  = c2["low"],
                    zone_low   = c0["high"],
                    zone_ce    = (c2["low"] + c0["high"]) / 2,
                ))

        # Bearish FVG: gap between c2.high and c0.low
        if fvg_type in ("Bearish", "Both"):
            gap = c0["low"] - c2["high"]
            if gap > min_gap:
                all_fvgs.append(FVG(
                    timestamp  = idx[i - 1],
                    timeframe  = timeframe,
                    fvg_type   = "bearish",
                    zone_high  = c0["low"],
                    zone_low   = c2["high"],
                    zone_ce    = (c0["low"] + c2["high"]) / 2,
                ))

    # ── Status tracking pass ──────────────────────────────────────────────
    # For each FVG, iterate candles AFTER its formation
    fvg_by_ts = {fvg.timestamp: fvg for fvg in all_fvgs}
    active: List[FVG] = []

    for ts, row in tf_df.iterrows():
        # Activate any FVG formed at this timestamp
        if ts in fvg_by_ts:
            active.append(fvg_by_ts[ts])
        # Update all active FVGs
        still_active = []
        for fvg in active:
            if fvg.timestamp >= ts:
                still_active.append(fvg)
                continue
            _update_fvg_status(fvg, row, ts)
            if fvg.status != "filled":
                still_active.append(fvg)
            # filled FVGs are kept in all_fvgs but removed from active queue
        active = still_active

    # ── Apply filters ──────────────────────────────────────────────────────
    result = []
    for fvg in all_fvgs:
        if status_filter != "Any":
            if fvg.status != status_filter.lower():
                continue
        if ce_filter == "Respect" and fvg.ce_status != "respected":
            continue
        if ce_filter == "Violate" and fvg.ce_status != "violated":
            continue
        result.append(fvg)

    return result


# ════════════════════════════════════════════════════════════════════════════
# INVERSE FVG DETECTOR
# ════════════════════════════════════════════════════════════════════════════

def detect_ifvg(
    df: pd.DataFrame,
    timeframe: FVGTF = "15m",
    ifvg_type: FVGType = "Both",
    status_filter: FVGStatus = "Any",
    ce_filter: CEFilter = "Any",
    min_gap_mult: float = 0.1,
    atr_period: int = 14,
) -> List[InverseFVG]:
    """
    Detect Inverse FVGs — fully filled FVGs that now act in reverse.
    """
    # Get ALL FVGs regardless of fill status
    all_fvgs = detect_fvg(df, timeframe, "Both", "Any", "Any",
                           min_gap_mult, atr_period)

    tf_df = _resample(df, timeframe)
    ifvgs: List[InverseFVG] = []

    for fvg in all_fvgs:
        if fvg.status != "filled":
            continue

        # Flip type for IFVG
        new_type: Literal["bullish", "bearish"] = (
            "bullish" if fvg.fvg_type == "bearish" else "bearish"
        )

        wanted = {"bullish", "bearish"} if ifvg_type == "Both" else {ifvg_type.lower()}
        if new_type not in wanted:
            continue

        ifvg_obj = InverseFVG(
            timestamp            = fvg.fill_timestamp or fvg.timestamp,
            timeframe            = timeframe,
            ifvg_type            = new_type,
            zone_high            = fvg.zone_high,
            zone_low             = fvg.zone_low,
            zone_ce              = fvg.zone_ce,
            origin_fvg_timestamp = fvg.timestamp,
        )

        # Track IFVG status and CE post-fill
        if fvg.fill_timestamp is not None:
            try:
                post_fill = tf_df.loc[tf_df.index > fvg.fill_timestamp]
            except Exception:
                ifvgs.append(ifvg_obj)
                continue

            for ts, row in post_fill.iterrows():
                if ifvg_obj.status in ("partial", "any"):
                    break
                # IFVG acts as opposite zone — check for price returning
                if new_type == "bullish":
                    # Expecting price to come back down to this zone as support
                    if row["low"] <= ifvg_obj.zone_high:
                        ifvg_obj.status = "partial"
                        if row["low"] <= ifvg_obj.zone_ce:
                            ifvg_obj.ce_status = "violated"
                            ifvg_obj.ce_timestamp = ts
                        else:
                            ifvg_obj.ce_status = "respected"
                            ifvg_obj.ce_timestamp = ts
                else:
                    if row["high"] >= ifvg_obj.zone_low:
                        ifvg_obj.status = "partial"
                        if row["high"] >= ifvg_obj.zone_ce:
                            ifvg_obj.ce_status = "violated"
                            ifvg_obj.ce_timestamp = ts
                        else:
                            ifvg_obj.ce_status = "respected"
                            ifvg_obj.ce_timestamp = ts

        # Apply filters
        if status_filter != "Any" and ifvg_obj.status != status_filter.lower():
            continue
        if ce_filter == "Respect" and ifvg_obj.ce_status != "respected":
            continue
        if ce_filter == "Violate" and ifvg_obj.ce_status != "violated":
            continue

        ifvgs.append(ifvg_obj)

    return ifvgs


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(55)
    n   = 3000
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 1.0800
    rows  = []
    for i in range(n):
        close += random.gauss(0.0001, 0.0008)
        o = close - abs(random.gauss(0, 0.0003))
        h = max(o, close) + abs(random.gauss(0, 0.0004))
        l = min(o, close) - abs(random.gauss(0, 0.0004))
        rows.append({"open": o, "high": h, "low": l,
                     "close": close, "volume": random.randint(100, 2000)})
    df = pd.DataFrame(rows, index=idx)

    for tf in ("5m", "15m", "1H"):
        fvgs  = detect_fvg(df, tf, "Both", "Any", "Any")
        ifvgs = detect_ifvg(df, tf, "Both", "Any", "Any")
        filled = [f for f in fvgs if f.status == "filled"]
        print(f"[{tf}] FVGs={len(fvgs)}  filled={len(filled)}  IFVGs={len(ifvgs)}")

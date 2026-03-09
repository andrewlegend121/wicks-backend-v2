"""
fvg_detector.py
===============
Wicks SMC Backtesting Engine — Layer 3

Detects Fair Value Gaps (FVGs) from OHLCV data.

Pine Script ground truth: ChartPrime FVG Volume Profile indicator

Definition
----------
A Fair Value Gap is a 3-candle imbalance where there is a price gap
between candles 0 and 2 that candle 1 (the impulse) does not cover.

  Bullish FVG:  low[0] > high[2]   — gap above candle 2's high
                The gap zone: bottom = high[2], top = low[0]

  Bearish FVG:  high[0] < low[2]   — gap below candle 2's low
                The gap zone: bottom = high[0], top = low[2]

Index convention (Pine Script):
  [0] = current bar (the bar AFTER the impulse)
  [1] = impulse bar  (middle candle — typically the largest)
  [2] = bar before the impulse

In Python arrays with oldest-first indexing:
  Detected at bar i, looking back:
    candle_0 = bar i      (current)
    candle_1 = bar i - 1  (impulse)
    candle_2 = bar i - 2  (pre-impulse)

Size filter (ChartPrime uses z-score; Python simplification):
  gap_size > atr_multiplier * ATR(14)
  Default: atr_multiplier = 0.0  (accept all gaps, no filter)
  Recommended production value: 0.1 (10% of ATR)

Invalidation (mitigation):
  Bullish FVG: price wicks INTO the zone (low <= fvg.top)   → partial fill
               price closes BELOW fvg.bottom                → fully invalidated
  Bearish FVG: price wicks INTO the zone (high >= fvg.bottom) → partial fill
               price closes ABOVE fvg.top                   → fully invalidated

States:
  "ACTIVE"    — untouched, full zone available
  "PARTIAL"   — price has entered the zone but not exited the other side
  "MITIGATED" — price has closed beyond the zone (zone consumed)

Output
------
  FVG dataclass:
    bar_index      int    — bar i where FVG was detected/confirmed
    timestamp      int
    direction      str    — "BULLISH" or "BEARISH"
    top            float  — upper boundary of gap zone
    bottom         float  — lower boundary of gap zone
    midline        float  — avg(top, bottom)
    impulse_bar    int    — bar index of candle_1 (the impulse)
    gap_size       float  — top - bottom
    status         str    — "ACTIVE", "PARTIAL", "MITIGATED"
    mitigated_bar  int | None

Usage
-----
  from fvg_detector import detect_fvgs, get_active_fvgs_at_bar

  fvgs = detect_fvgs(highs, lows, closes, timestamps)
  active = get_active_fvgs_at_bar(fvgs, bar_index=100)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


# ---------------------------------------------------------------------------
# ATR helper (simple, no pandas dependency)
# ---------------------------------------------------------------------------

def _atr(highs: List[float], lows: List[float], closes: List[float],
         period: int, idx: int) -> float:
    """Compute ATR at bar idx using Wilder's smoothing (simplified to SMA here)."""
    if idx < 1:
        return highs[idx] - lows[idx]
    start = max(1, idx - period + 1)
    trs = []
    for j in range(start, idx + 1):
        tr = max(
            highs[j] - lows[j],
            abs(highs[j] - closes[j - 1]),
            abs(lows[j]  - closes[j - 1]),
        )
        trs.append(tr)
    return sum(trs) / len(trs) if trs else (highs[idx] - lows[idx])


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FVG:
    bar_index:     int         # bar where FVG was confirmed (candle_0 bar)
    timestamp:     int
    direction:     str         # "BULLISH" or "BEARISH"
    top:           float       # upper boundary
    bottom:        float       # lower boundary
    midline:       float       # avg(top, bottom)
    impulse_bar:   int         # bar index of the impulse candle (candle_1)
    gap_size:      float       # top - bottom

    status:        str         = "ACTIVE"   # "ACTIVE", "PARTIAL", "MITIGATED"
    mitigated_bar: Optional[int] = None
    partial_bar:   Optional[int] = None

    # Set after full scan — bar when first touched (for BPR / IFVG derivation)
    first_touch_bar: Optional[int] = None

    def __repr__(self):
        return (f"FVG({self.direction} top={self.top:.5f} bot={self.bottom:.5f} "
                f"bar={self.bar_index} status={self.status})")

    @property
    def is_active(self) -> bool:
        return self.status in ("ACTIVE", "PARTIAL")

    @property
    def is_fully_filled(self) -> bool:
        """True when price has closed fully through the zone (for IFVG derivation)."""
        return self.status == "MITIGATED"


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_fvgs(
    highs:          Sequence[float],
    lows:           Sequence[float],
    closes:         Sequence[float],
    timestamps:     Sequence[int],
    atr_multiplier: float = 0.0,
    atr_period:     int   = 14,
) -> List[FVG]:
    """
    Detect and track FVGs across the full bar array.

    Detection fires at bar i, using candles i, i-1, i-2.
    Mitigation is updated in the same pass for all active FVGs.

    Parameters
    ----------
    highs, lows, closes : OHLCV arrays, oldest first
    timestamps          : unix ms per bar
    atr_multiplier      : minimum gap size as multiple of ATR(14).
                          0.0 = accept all. 0.1 recommended for production.
    atr_period          : ATR lookback (default 14)

    Returns
    -------
    List[FVG] with status updated through the full history, sorted bar_index asc.
    """
    highs  = list(highs)
    lows   = list(lows)
    closes = list(closes)
    timestamps = list(timestamps)
    bars   = len(closes)

    if not (bars == len(highs) == len(lows) == len(timestamps)):
        raise ValueError("All arrays must have the same length")

    fvgs: List[FVG] = []

    for i in range(2, bars):
        # Candle indices (oldest-first array, Pine [0]=current [2]=older)
        c0_h, c0_l = highs[i],     lows[i]      # current bar
        c2_h, c2_l = highs[i-2],   lows[i-2]    # pre-impulse bar

        # Size filter
        atr_val = _atr(highs, lows, closes, atr_period, i) if atr_multiplier > 0 else 0.0

        # ── Bullish FVG: gap above candle_2's high ──────────────────────
        if c0_l > c2_h:
            gap_size = c0_l - c2_h
            if gap_size > atr_multiplier * atr_val:
                fvg = FVG(
                    bar_index   = i,
                    timestamp   = timestamps[i],
                    direction   = "BULLISH",
                    top         = c0_l,
                    bottom      = c2_h,
                    midline     = (c0_l + c2_h) / 2,
                    impulse_bar = i - 1,
                    gap_size    = gap_size,
                )
                fvgs.append(fvg)

        # ── Bearish FVG: gap below candle_2's low ───────────────────────
        if c0_h < c2_l:
            gap_size = c2_l - c0_h
            if gap_size > atr_multiplier * atr_val:
                fvg = FVG(
                    bar_index   = i,
                    timestamp   = timestamps[i],
                    direction   = "BEARISH",
                    top         = c2_l,
                    bottom      = c0_h,
                    midline     = (c2_l + c0_h) / 2,
                    impulse_bar = i - 1,
                    gap_size    = gap_size,
                )
                fvgs.append(fvg)

    # ── Second pass: update mitigation status for all FVGs ──────────────
    for fvg in fvgs:
        start = fvg.bar_index + 1
        for j in range(start, bars):
            h, l, c = highs[j], lows[j], closes[j]

            if fvg.direction == "BULLISH":
                if l <= fvg.top and fvg.partial_bar is None:
                    fvg.status      = "PARTIAL"
                    fvg.partial_bar = j
                    fvg.first_touch_bar = j
                if c < fvg.bottom:
                    fvg.status        = "MITIGATED"
                    fvg.mitigated_bar = j
                    break

            else:  # BEARISH
                if h >= fvg.bottom and fvg.partial_bar is None:
                    fvg.status      = "PARTIAL"
                    fvg.partial_bar = j
                    fvg.first_touch_bar = j
                if c > fvg.top:
                    fvg.status        = "MITIGATED"
                    fvg.mitigated_bar = j
                    break

    fvgs.sort(key=lambda f: f.bar_index)
    return fvgs


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_active_fvgs_at_bar(
    fvgs:      List[FVG],
    bar_index: int,
    direction: Optional[str] = None,
) -> List[FVG]:
    """
    Return FVGs that:
      1. Were confirmed at or before bar_index
      2. Are still ACTIVE or PARTIAL at bar_index (not yet mitigated)

    Parameters
    ----------
    direction : "BULLISH", "BEARISH", or None (both)
    """
    result = []
    for fvg in fvgs:
        if fvg.bar_index > bar_index:
            continue
        if direction and fvg.direction != direction:
            continue
        # Still active at this bar?
        if fvg.status == "ACTIVE":
            result.append(fvg)
        elif fvg.status == "PARTIAL" and fvg.partial_bar <= bar_index:
            result.append(fvg)
        elif fvg.status == "MITIGATED" and fvg.mitigated_bar > bar_index:
            # Was still active at bar_index even though later mitigated
            result.append(fvg)
    return result


def get_nearest_fvg(
    fvgs:      List[FVG],
    bar_index: int,
    price:     float,
    direction: Optional[str] = None,
) -> Optional[FVG]:
    """
    Return the active FVG whose midline is closest to `price` at bar_index.
    Used by confluence scorer to find the most relevant FVG near current price.
    """
    active = get_active_fvgs_at_bar(fvgs, bar_index, direction)
    if not active:
        return None
    return min(active, key=lambda f: abs(f.midline - price))


def get_unfilled_fvgs(
    fvgs: List[FVG],
    direction: Optional[str] = None,
) -> List[FVG]:
    """Return all FVGs that were never mitigated (status ACTIVE or PARTIAL)."""
    return [f for f in fvgs
            if f.status != "MITIGATED"
            and (direction is None or f.direction == direction)]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math, sys
    sys.path.insert(0, ".")

    print("=" * 60)
    print("fvg_detector.py — self test")
    print("=" * 60)

    N = 100
    ts_base = 1_700_000_000_000

    # Build a price series with intentional gaps
    # At bar 10: bullish impulse with gap above
    # At bar 40: bearish impulse with gap below
    highs  = [100.5] * N
    lows   = [99.5]  * N
    closes = [100.0] * N

    # Inject bullish FVG at bar 12 (candle_2=10, impulse=11, current=12)
    highs[10]  = 101.0; lows[10]  = 99.5;  closes[10] = 100.8
    highs[11]  = 104.0; lows[11]  = 101.2; closes[11] = 103.8  # impulse
    highs[12]  = 106.0; lows[12]  = 101.5; closes[12] = 105.5  # current; low > high[10]=101.0 → gap!

    # Inject bearish FVG at bar 42 (candle_2=40, impulse=41, current=42)
    highs[40]  = 100.5; lows[40]  = 99.0;  closes[40] = 99.2
    highs[41]  = 98.8;  lows[41]  = 95.0;  closes[41] = 95.5   # bearish impulse
    highs[42]  = 98.5;  lows[42]  = 93.0;  closes[42] = 93.8   # current; high[42]=98.5 < low[40]=99.0 → gap!

    ts_arr = [ts_base + i * 60_000 for i in range(N)]

    fvgs = detect_fvgs(highs, lows, closes, ts_arr)

    print(f"\nDetected {len(fvgs)} FVGs:")
    for f in fvgs:
        print(f"  {f}")

    bull_fvgs = [f for f in fvgs if f.direction == "BULLISH"]
    bear_fvgs = [f for f in fvgs if f.direction == "BEARISH"]
    assert len(bull_fvgs) >= 1, "Expected at least 1 bullish FVG"
    assert len(bear_fvgs) >= 1, "Expected at least 1 bearish FVG"
    print(f"\n  Bullish FVGs: {len(bull_fvgs)}")
    print(f"  Bearish FVGs: {len(bear_fvgs)}")

    # Check zones on injected FVGs
    bfvg = next(f for f in bull_fvgs if f.bar_index == 12)
    assert bfvg.bottom == 101.0, f"Bullish FVG bottom wrong: {bfvg.bottom}"
    assert bfvg.top    == 101.5, f"Bullish FVG top wrong: {bfvg.top}"
    print(f"\n  Bullish FVG zone: [{bfvg.bottom}, {bfvg.top}] ✓")

    berfvg = next(f for f in bear_fvgs if f.bar_index == 42)
    assert berfvg.top    == 99.0, f"Bearish FVG top wrong: {berfvg.top}"
    assert berfvg.bottom == 98.5, f"Bearish FVG bottom wrong: {berfvg.bottom}"
    print(f"  Bearish FVG zone: [{berfvg.bottom}, {berfvg.top}] ✓")

    # Test active at bar query
    active_at_50 = get_active_fvgs_at_bar(fvgs, bar_index=50)
    print(f"\n  Active FVGs at bar 50: {len(active_at_50)}")

    # Test FVG not visible before its bar
    active_at_11 = get_active_fvgs_at_bar(fvgs, bar_index=11)
    bull_at_11 = [f for f in active_at_11 if f.bar_index == 12]
    assert len(bull_at_11) == 0, "LOOK-AHEAD: FVG at bar 12 should not be visible at bar 11"
    print("  Look-ahead check: FVG at bar 12 not visible at bar 11 ✓")

    # Test mitigation with a fully isolated price series
    N2  = 40
    h2  = [100.2] * N2; l2 = [99.8] * N2; c2 = [100.0] * N2
    ts2 = [ts_base + i * 60_000 for i in range(N2)]
    # Clean bullish FVG: pre=4, impulse=5, current=6
    h2[4]=100.2; l2[4]=99.8;  c2[4]=100.0
    h2[5]=103.0; l2[5]=100.3; c2[5]=102.8   # impulse
    h2[6]=105.0; l2[6]=102.5; c2[6]=104.8   # low(102.5) > high[4](100.2) → gap zone [100.2, 102.5]
    for i in range(7, 15):
        h2[i]=105.5; l2[i]=104.5; c2[i]=105.0  # flat above zone
    h2[15]=104.0; l2[15]=101.5; c2[15]=103.5   # wick into zone (partial)
    for i in range(16, 20):
        h2[i]=103.5; l2[i]=102.8; c2[i]=103.2  # stay above
    h2[20]=101.5; l2[20]=99.5;  c2[20]=100.0   # close below 100.2 → mitigated

    fvgs2 = detect_fvgs(h2, l2, c2, ts2)
    bfvg2 = next((f for f in fvgs2 if f.direction=="BULLISH"
                  and f.bar_index==6 and abs(f.bottom-100.2)<0.01), None)
    assert bfvg2 is not None, f"Expected bullish FVG at bar 6. Got: {fvgs2}"
    assert bfvg2.status == "MITIGATED", f"Expected MITIGATED, got {bfvg2.status}"
    assert bfvg2.partial_bar == 15, f"Expected partial_bar=15, got {bfvg2.partial_bar}"
    assert bfvg2.mitigated_bar == 20, f"Expected mitigated_bar=20, got {bfvg2.mitigated_bar}"
    print(f"  Mitigation: partial at bar 15, mitigated at bar 20 \u2713")

    print("\n✅ All tests passed")

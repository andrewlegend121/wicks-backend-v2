"""
trend_detector.py
=================
Wicks SMC Backtesting Engine — Layer 2

Determines market trend by reading the sequence of confirmed swing highs
and swing lows produced by swing_detector.py.

Pine Script ground truth: TFlab Market Structures indicator
  - ExternalTrend: sequence of Major swing points (pivot length = 5)
  - InternalTrend: sequence of Minor swing points (same pivot, filtered)

Logic
-----
Trend is determined by the classic HH/HL / LH/LL classification:

  BULLISH  — price is making Higher Highs AND Higher Lows
  BEARISH  — price is making Lower Highs AND Lower Lows
  RANGING  — mixed signals (LH + HL, or insufficient data)

The trend is re-evaluated every time a NEW swing point is confirmed.
Only swings confirmed at or before the current bar are used (no look-ahead).

Swing classification:
  HH — swing HIGH above the previous swing HIGH
  LH — swing HIGH below the previous swing HIGH
  HL — swing LOW  above the previous swing LOW
  LL — swing LOW  below the previous swing LOW

Trend state machine (mirrors TFlab ExternalTrend):
  Start     → RANGING (no data)
  Any BOS/CHoCH bullish  → BULLISH
  Any BOS/CHoCH bearish  → BEARISH

For the trend detector itself (pre-BOS), we use the simpler swing sequence:
  2 consecutive HH+HL     → BULLISH
  2 consecutive LH+LL     → BEARISH
  Mixed                   → RANGING

Output
------
  TrendState dataclass per bar evaluated, containing:
    bar_index   int
    timestamp   int
    trend       str  — "BULLISH", "BEARISH", "RANGING"
    last_hh     float | None  — price of most recent confirmed HH
    last_hl     float | None
    last_lh     float | None
    last_ll     float | None

Usage
-----
  from swing_detector import detect_swings
  from trend_detector import detect_trend, get_trend_at_bar

  swings = detect_swings(highs, lows, timestamps, method="pivot", length=5)
  trend_states = detect_trend(swings, timestamps)
  current = get_trend_at_bar(trend_states, bar_index=150)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence
from swing_detector import SwingPoint


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TrendState:
    bar_index:  int
    timestamp:  int
    trend:      str          # "BULLISH", "BEARISH", "RANGING"

    # Most recent confirmed swing prices at this bar
    last_hh:    Optional[float] = None   # last Higher High
    last_hl:    Optional[float] = None   # last Higher Low
    last_lh:    Optional[float] = None   # last Lower High
    last_ll:    Optional[float] = None   # last Lower Low

    # Raw last swing prices (regardless of HH/LH classification)
    last_swing_high: Optional[float] = None
    last_swing_low:  Optional[float] = None

    def __repr__(self):
        return (f"TrendState(bar={self.bar_index} trend={self.trend} "
                f"HH={self.last_hh} HL={self.last_hl} "
                f"LH={self.last_lh} LL={self.last_ll})")


@dataclass
class SwingLabel:
    """A swing point with its HH/LH/HL/LL classification."""
    swing:      SwingPoint
    label:      str   # "HH", "LH", "HL", "LL"


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_trend(
    swings:     List[SwingPoint],
    timestamps: Sequence[int],
    method:     str = "pivot",
    length:     int = 5,
) -> List[TrendState]:
    """
    Compute trend state for every bar in `timestamps`.

    Parameters
    ----------
    swings     : output of swing_detector.detect_swings()
    timestamps : full bar timestamp array (unix ms), oldest first
    method     : filter swings by this method ("pivot" matches TFlab)
    length     : filter swings by this length (5 matches TFlab default)

    Returns
    -------
    List[TrendState], one per bar, sorted bar_index ascending.
    """
    # Filter to the correct method/length
    filtered = [s for s in swings
                if s.method == method and s.length == length]

    highs_seq = [s for s in filtered if s.direction == "HIGH"]
    lows_seq  = [s for s in filtered if s.direction == "LOW"]

    # Sort by confirmed_bar so we process in time order
    highs_seq.sort(key=lambda s: s.confirmed_bar)
    lows_seq.sort(key=lambda s:  s.confirmed_bar)

    # State tracking
    prev_high:  Optional[float] = None
    prev_low:   Optional[float] = None
    last_hh:    Optional[float] = None
    last_hl:    Optional[float] = None
    last_lh:    Optional[float] = None
    last_ll:    Optional[float] = None
    trend:      str             = "RANGING"

    # Pointers into sorted swing lists
    hi_idx = 0
    lo_idx = 0

    results: List[TrendState] = []

    for bar_i, ts in enumerate(timestamps):

        # Consume all swings confirmed at or before this bar
        while hi_idx < len(highs_seq) and highs_seq[hi_idx].confirmed_bar <= bar_i:
            sh = highs_seq[hi_idx]
            if prev_high is None:
                prev_high = sh.price
            else:
                if sh.price > prev_high:
                    last_hh   = sh.price
                    last_lh   = None          # reset opposite
                else:
                    last_lh   = sh.price
                    last_hh   = None
                prev_high = sh.price
            hi_idx += 1

        while lo_idx < len(lows_seq) and lows_seq[lo_idx].confirmed_bar <= bar_i:
            sl = lows_seq[lo_idx]
            if prev_low is None:
                prev_low = sl.price
            else:
                if sl.price > prev_low:
                    last_hl   = sl.price
                    last_ll   = None
                else:
                    last_ll   = sl.price
                    last_hl   = None
                prev_low = sl.price
            lo_idx += 1

        # Determine trend from current HH/HL/LH/LL state
        if last_hh is not None and last_hl is not None:
            trend = "BULLISH"
        elif last_lh is not None and last_ll is not None:
            trend = "BEARISH"
        # else: trend stays as previous (persistence)

        results.append(TrendState(
            bar_index        = bar_i,
            timestamp        = ts,
            trend            = trend,
            last_hh          = last_hh,
            last_hl          = last_hl,
            last_lh          = last_lh,
            last_ll          = last_ll,
            last_swing_high  = prev_high,
            last_swing_low   = prev_low,
        ))

    return results


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_trend_at_bar(
    trend_states: List[TrendState],
    bar_index:    int,
) -> Optional[TrendState]:
    """Return the TrendState for exactly bar_index, or None."""
    if bar_index < 0 or bar_index >= len(trend_states):
        return None
    return trend_states[bar_index]


def get_trend_str_at_bar(
    trend_states: List[TrendState],
    bar_index:    int,
) -> str:
    """Return trend string at bar_index, defaulting to 'RANGING'."""
    state = get_trend_at_bar(trend_states, bar_index)
    return state.trend if state else "RANGING"


def find_trend_changes(
    trend_states: List[TrendState],
) -> List[TrendState]:
    """Return only the bars where trend changed from the previous bar."""
    changes = []
    prev = "RANGING"
    for ts in trend_states:
        if ts.trend != prev:
            changes.append(ts)
            prev = ts.trend
    return changes


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math, sys
    sys.path.insert(0, ".")
    from swing_detector import detect_swings

    print("=" * 60)
    print("trend_detector.py — self test")
    print("=" * 60)

    N  = 120
    ts_base = 1_700_000_000_000

    # Build an uptrend then downtrend series
    # Bars 0-59:  rising sine → should resolve BULLISH
    # Bars 60-119: falling   → should resolve BEARISH
    def make_price(i):
        if i < 60:
            return 100 + i * 0.3 + 5 * math.sin(2 * math.pi * i / 16)
        else:
            return 118 - (i - 60) * 0.3 + 5 * math.sin(2 * math.pi * i / 16)

    mid  = [make_price(i) for i in range(N)]
    highs  = [m + 0.6 for m in mid]
    lows   = [m - 0.6 for m in mid]
    ts_arr = [ts_base + i * 300_000 for i in range(N)]  # 5-min bars

    swings = detect_swings(highs, lows, ts_arr, method="pivot", length=5)
    states = detect_trend(swings, ts_arr)

    changes = find_trend_changes(states)
    print(f"\nDetected {len(swings)} swing points")
    print(f"Trend changes:")
    for c in changes:
        print(f"  bar={c.bar_index:3d}  → {c.trend}")

    # By bar 50 we should be BULLISH
    t50 = get_trend_at_bar(states, 50)
    print(f"\nTrend at bar 50: {t50.trend}")
    assert t50.trend == "BULLISH", f"Expected BULLISH at bar 50, got {t50.trend}"

    # By bar 110 we should be BEARISH
    t110 = get_trend_at_bar(states, 110)
    print(f"Trend at bar 110: {t110.trend}")
    assert t110.trend == "BEARISH", f"Expected BEARISH at bar 110, got {t110.trend}"

    # Trend at bar 0 should be RANGING (not enough data)
    t0 = get_trend_at_bar(states, 0)
    assert t0.trend == "RANGING"
    print(f"Trend at bar 0:   {t0.trend} (expected RANGING)")

    print("\n✅ All tests passed")

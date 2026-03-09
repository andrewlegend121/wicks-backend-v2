"""
bos_choch_detector.py
=====================
Wicks SMC Backtesting Engine — Layer 2

Detects Break of Structure (BOS) and Change of Character (CHoCH) events
from confirmed swing points and trend state.

Pine Script ground truth: TFlab Market Structures indicator
  - BOS:   close crosses a Major swing level IN THE SAME direction as trend
  - CHoCH: close crosses a Major swing level AGAINST the current trend
  - Uses pivot method, default PP = 5

Definitions
-----------
BOS (Break of Structure):
  Bullish BOS  — close > last confirmed Major swing HIGH, trend is BULLISH
                 (continuation: bull trend breaks above a prior swing high)
  Bearish BOS  — close < last confirmed Major swing LOW,  trend is BEARISH
                 (continuation: bear trend breaks below a prior swing low)

CHoCH (Change of Character):
  Bullish CHoCH — close > last confirmed Major swing HIGH, trend is BEARISH
                  (reversal signal: bear trend broken upward)
  Bearish CHoCH — close < last confirmed Major swing LOW,  trend is BULLISH
                  (reversal signal: bull trend broken downward)

Key implementation detail from Pine Script:
  LockBreak_M flag — once a swing level has been broken, it cannot fire
  again. The detector moves on to the NEXT unbroken swing level.
  In Python: each SwingPoint is consumed (marked broken) at most once.

Output
------
  BosChochEvent dataclass:
    bar_index   int    — bar where the close crossed the level
    timestamp   int
    event_type  str    — "BOS" or "CHoCH"
    direction   str    — "BULLISH" or "BEARISH"
    level       float  — the swing price that was crossed
    swing_bar   int    — bar_index of the originating swing point
    trend_at_break str — trend state at the moment of the break

Usage
-----
  from swing_detector import detect_swings
  from trend_detector import detect_trend
  from bos_choch_detector import detect_bos_choch

  swings = detect_swings(highs, lows, timestamps, method="pivot", length=5)
  trend_states = detect_trend(swings, timestamps)
  events = detect_bos_choch(highs, lows, closes, timestamps,
                             swings, trend_states)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from swing_detector import SwingPoint
from trend_detector import TrendState


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BosChochEvent:
    bar_index:      int
    timestamp:      int
    event_type:     str    # "BOS" or "CHoCH"
    direction:      str    # "BULLISH" or "BEARISH"
    level:          float  # swing price that was crossed
    swing_bar:      int    # bar_index of the originating swing
    trend_at_break: str    # trend state when break occurred

    def __repr__(self):
        return (f"BosChochEvent({self.event_type} {self.direction} "
                f"@ {self.level:.5f} bar={self.bar_index} "
                f"[swing bar={self.swing_bar}])")


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_bos_choch(
    highs:        Sequence[float],
    lows:         Sequence[float],
    closes:       Sequence[float],
    timestamps:   Sequence[int],
    swings:       List[SwingPoint],
    trend_states: List[TrendState],
    method:       str = "pivot",
    length:       int = 5,
) -> List[BosChochEvent]:
    """
    Detect BOS and CHoCH events bar-by-bar.

    Parameters
    ----------
    highs, lows, closes : OHLCV arrays, oldest first
    timestamps          : unix ms per bar
    swings              : from swing_detector.detect_swings()
    trend_states        : from trend_detector.detect_trend()
    method, length      : filter swings to match the Pine Script source

    Returns
    -------
    List[BosChochEvent] sorted by bar_index ascending.
    """
    bars = len(closes)
    if not (bars == len(highs) == len(lows) == len(timestamps)):
        raise ValueError("All input arrays must have the same length")

    # Filter swings to the correct method/length
    filtered_highs = sorted(
        [s for s in swings if s.direction == "HIGH"
         and s.method == method and s.length == length],
        key=lambda s: s.confirmed_bar
    )
    filtered_lows = sorted(
        [s for s in swings if s.direction == "LOW"
         and s.method == method and s.length == length],
        key=lambda s: s.confirmed_bar
    )

    events: List[BosChochEvent] = []

    # Track which swing bar_indices have already been broken (LockBreak_M)
    broken_swings: Set[int] = set()

    # Pointer into confirmed swing lists — we process in bar order
    hi_confirmed_idx = 0
    lo_confirmed_idx = 0

    # Active unbroken swing levels visible so far
    active_highs: List[SwingPoint] = []
    active_lows:  List[SwingPoint] = []

    for bar_i in range(bars):
        ts    = timestamps[bar_i]
        close = closes[bar_i]

        # Get trend at this bar (no look-ahead — trend computed from swings
        # confirmed <= bar_i, same as this detector)
        trend = trend_states[bar_i].trend if bar_i < len(trend_states) else "RANGING"

        # Add newly confirmed swing HIGHs to active list
        while (hi_confirmed_idx < len(filtered_highs) and
               filtered_highs[hi_confirmed_idx].confirmed_bar <= bar_i):
            sh = filtered_highs[hi_confirmed_idx]
            if sh.bar_index not in broken_swings:
                active_highs.append(sh)
            hi_confirmed_idx += 1

        # Add newly confirmed swing LOWs to active list
        while (lo_confirmed_idx < len(filtered_lows) and
               filtered_lows[lo_confirmed_idx].confirmed_bar <= bar_i):
            sl = filtered_lows[lo_confirmed_idx]
            if sl.bar_index not in broken_swings:
                active_lows.append(sl)
            lo_confirmed_idx += 1

        # Check for breaks of swing HIGHs (bullish close above)
        # Iterate over unbroken highs — oldest first (TFlab checks Major level)
        # Use the most recent unbroken high (last in list)
        newly_broken_hi: List[int] = []
        for sh in active_highs:
            if sh.bar_index in broken_swings:
                continue
            if close > sh.price:
                # Determine BOS vs CHoCH
                if trend == "BULLISH":
                    etype = "BOS"
                elif trend == "BEARISH":
                    etype = "CHoCH"
                else:
                    # RANGING: treat upward break as CHoCH (potential reversal)
                    etype = "CHoCH"

                events.append(BosChochEvent(
                    bar_index      = bar_i,
                    timestamp      = ts,
                    event_type     = etype,
                    direction      = "BULLISH",
                    level          = sh.price,
                    swing_bar      = sh.bar_index,
                    trend_at_break = trend,
                ))
                broken_swings.add(sh.bar_index)
                newly_broken_hi.append(sh.bar_index)
                # Only break ONE level per bar (TFlab fires once per swing)
                break

        # Check for breaks of swing LOWs (bearish close below)
        for sl in active_lows:
            if sl.bar_index in broken_swings:
                continue
            if close < sl.price:
                if trend == "BEARISH":
                    etype = "BOS"
                elif trend == "BULLISH":
                    etype = "CHoCH"
                else:
                    etype = "CHoCH"

                events.append(BosChochEvent(
                    bar_index      = bar_i,
                    timestamp      = ts,
                    event_type     = etype,
                    direction      = "BEARISH",
                    level          = sl.price,
                    swing_bar      = sl.bar_index,
                    trend_at_break = trend,
                ))
                broken_swings.add(sl.bar_index)
                break

        # Clean up broken swings from active lists
        active_highs = [s for s in active_highs if s.bar_index not in broken_swings]
        active_lows  = [s for s in active_lows  if s.bar_index not in broken_swings]

    events.sort(key=lambda e: e.bar_index)
    return events


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_last_bos_choch(
    events:    List[BosChochEvent],
    bar_index: int,
    direction: Optional[str] = None,
    event_type: Optional[str] = None,
) -> Optional[BosChochEvent]:
    """
    Return the most recent BOS/CHoCH event at or before bar_index.

    Parameters
    ----------
    direction  : filter to "BULLISH" or "BEARISH" (None = any)
    event_type : filter to "BOS" or "CHoCH" (None = any)
    """
    candidates = [
        e for e in events
        if e.bar_index <= bar_index
        and (direction  is None or e.direction  == direction)
        and (event_type is None or e.event_type == event_type)
    ]
    return candidates[-1] if candidates else None


def get_last_choch(
    events:    List[BosChochEvent],
    bar_index: int,
    direction: Optional[str] = None,
) -> Optional[BosChochEvent]:
    """Convenience: most recent CHoCH at or before bar_index."""
    return get_last_bos_choch(events, bar_index, direction, "CHoCH")


def get_last_bos(
    events:    List[BosChochEvent],
    bar_index: int,
    direction: Optional[str] = None,
) -> Optional[BosChochEvent]:
    """Convenience: most recent BOS at or before bar_index."""
    return get_last_bos_choch(events, bar_index, direction, "BOS")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math, sys
    sys.path.insert(0, ".")
    from swing_detector import detect_swings
    from trend_detector import detect_trend

    print("=" * 60)
    print("bos_choch_detector.py — self test")
    print("=" * 60)

    N = 150
    ts_base = 1_700_000_000_000

    # Phase 1 (0-59):   Downtrend  → bear BOS events expected
    # Phase 2 (60-89):  Reversal   → bullish CHoCH expected
    # Phase 3 (90-149): Uptrend    → bull BOS events expected
    def make_price(i):
        if i < 60:
            return 110 - i * 0.25 + 4 * math.sin(2 * math.pi * i / 14)
        elif i < 90:
            return 95 + (i - 60) * 0.5 + 4 * math.sin(2 * math.pi * i / 14)
        else:
            return 110 + (i - 90) * 0.3 + 4 * math.sin(2 * math.pi * i / 14)

    mid    = [make_price(i) for i in range(N)]
    highs  = [m + 0.8 for m in mid]
    lows   = [m - 0.8 for m in mid]
    closes = [m + 0.1 * math.sin(i) for m in mid for i in [mid.index(m)]]
    closes = mid  # close = mid for simplicity
    ts_arr = [ts_base + i * 300_000 for i in range(N)]

    swings = detect_swings(highs, lows, ts_arr, method="pivot", length=5)
    states = detect_trend(swings, ts_arr)
    events = detect_bos_choch(highs, lows, closes, ts_arr, swings, states)

    print(f"\nDetected {len(events)} BOS/CHoCH events:")
    for e in events:
        print(f"  {e}")

    bos_events   = [e for e in events if e.event_type == "BOS"]
    choch_events = [e for e in events if e.event_type == "CHoCH"]
    print(f"\nBOS total:   {len(bos_events)}")
    print(f"CHoCH total: {len(choch_events)}")

    assert len(events) > 0, "Expected at least some BOS/CHoCH events"
    assert len(choch_events) > 0, "Expected at least one CHoCH event"

    # Check no look-ahead: no event fires before its swing is confirmed
    for e in events:
        # The swing must have been confirmed before or at the break bar
        matching = [s for s in swings if s.bar_index == e.swing_bar]
        if matching:
            assert matching[0].confirmed_bar <= e.bar_index, \
                f"LOOK-AHEAD: event at bar {e.bar_index} but swing confirmed at {matching[0].confirmed_bar}"
    print("\nLook-ahead bias check: PASS")

    # Query helpers
    last_bull_choch = get_last_choch(events, 149, direction="BULLISH")
    print(f"Last bullish CHoCH by bar 149: {last_bull_choch}")

    print("\n✅ All tests passed")

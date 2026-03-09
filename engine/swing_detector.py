"""
swing_detector.py
=================
Wicks SMC Backtesting Engine — Layer 1 (Foundation)

Detects swing highs and swing lows from OHLCV data using two methods that
mirror the Pine Script indicators used as ground truth:

  METHOD A — "pivot"
    Mirrors: ta.pivothigh(n, n) / ta.pivotlow(n, n)
    Used by: TFlab Market Structures (BOS/CHoCH, Trend)
    Logic:   A bar is a confirmed swing high if its high is strictly the
             highest in a window of (n) bars on each side. Confirmation
             arrives n bars after the pivot bar, adding exactly n bars of lag.
    Default: n = 5

  METHOD B — "rolling"
    Mirrors: ta.highest(length) / ta.lowest(length) with os state machine
    Used by: LuxAlgo OB & Breaker Blocks (length=10), Propulsion Block (length=3)
    Logic:   A swing high fires when the bar at position [length] ago is
             higher than the current rolling max of the full window. Uses an
             os (oscillator state) variable that flips between 0 (up swing)
             and 1 (down swing), and only emits a new swing on a state
             transition — preventing repeated signals on the same move.
    Defaults: length = 10 (Order Blocks), length = 3 (Propulsion Blocks)

Both methods are implemented in a single module so that all downstream
detectors share a common, tested foundation.

Output
------
Each detected swing is a SwingPoint dataclass:
  bar_index   int     — position in the input arrays (0-based)
  timestamp   int     — unix ms timestamp of the candle
  price       float   — the high (swing high) or low (swing low) price
  direction   str     — "HIGH" or "LOW"
  method      str     — "pivot" or "rolling"
  length      int     — the window parameter used (n for pivot, length for rolling)
  confirmed_bar int   — bar_index where this swing became confirmed
                        (= bar_index for rolling; = bar_index + n for pivot)

Usage
-----
  from swing_detector import detect_swings, SwingPoint

  # Pivot method — matches TFlab BOS/CHoCH indicator
  swings = detect_swings(highs, lows, timestamps, method="pivot", length=5)

  # Rolling method — matches LuxAlgo Order Blocks
  swings = detect_swings(highs, lows, timestamps, method="rolling", length=10)

  # Both at once (returns combined sorted list)
  swings = detect_swings(highs, lows, timestamps, method="both",
                         pivot_length=5, rolling_lengths=[10, 3])

  # Filter helpers
  swing_highs = [s for s in swings if s.direction == "HIGH"]
  swing_lows  = [s for s in swings if s.direction == "LOW"]
  ob_swings   = [s for s in swings if s.method == "rolling" and s.length == 10]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SwingPoint:
    """A single confirmed swing high or low."""

    bar_index:     int    # index in the input arrays where the candle lives
    timestamp:     int    # unix milliseconds
    price:         float  # high price (HIGH) or low price (LOW)
    direction:     str    # "HIGH" or "LOW"
    method:        str    # "pivot" or "rolling"
    length:        int    # window parameter used for detection

    # Bar index at which this swing became known / confirmed.
    # For "pivot":   confirmed_bar = bar_index + length  (lag)
    # For "rolling": confirmed_bar = bar_index           (same bar as swing)
    confirmed_bar: int = field(default=0)

    def __post_init__(self):
        if self.direction not in ("HIGH", "LOW"):
            raise ValueError(f"direction must be 'HIGH' or 'LOW', got {self.direction!r}")
        if self.method not in ("pivot", "rolling"):
            raise ValueError(f"method must be 'pivot' or 'rolling', got {self.method!r}")

    def __repr__(self) -> str:
        return (
            f"SwingPoint({self.direction} @ {self.price:.5f} "
            f"bar={self.bar_index} confirmed={self.confirmed_bar} "
            f"method={self.method}[{self.length}])"
        )


# ---------------------------------------------------------------------------
# Method A — Pivot (mirrors ta.pivothigh / ta.pivotlow)
# ---------------------------------------------------------------------------

def _detect_pivot_swings(
    highs:      Sequence[float],
    lows:       Sequence[float],
    timestamps: Sequence[int],
    n:          int = 5,
) -> List[SwingPoint]:
    """
    Pine Script equivalent:
        ph = ta.pivothigh(n, n)
        pl = ta.pivotlow(n, n)

    A pivot high at bar i is confirmed when we are at bar i+n, and
    highs[i] is STRICTLY greater than every bar in [i-n .. i-1] and [i+1 .. i+n].

    A pivot low at bar i is confirmed when lows[i] is STRICTLY less than
    every bar in the same window.

    Strict inequality matches Pine Script behaviour: ties are not pivots.

    Parameters
    ----------
    highs, lows : sequences of float — OHLCV high/low arrays (oldest first)
    timestamps  : sequence of int    — unix ms per bar, same length
    n           : int                — bars required on each side (default 5)

    Returns
    -------
    List[SwingPoint] sorted by bar_index ascending.
    """
    if n < 1:
        raise ValueError(f"pivot length n must be >= 1, got {n}")

    bars = len(highs)
    if bars != len(lows) or bars != len(timestamps):
        raise ValueError("highs, lows, timestamps must have the same length")

    results: List[SwingPoint] = []

    # We can only confirm pivots for bars in [n .. bars-n-1]
    for i in range(n, bars - n):
        pivot_high = highs[i]
        pivot_low  = lows[i]

        is_swing_high = True
        is_swing_low  = True

        for j in range(i - n, i + n + 1):
            if j == i:
                continue
            if highs[j] >= pivot_high:
                is_swing_high = False
            if lows[j] <= pivot_low:
                is_swing_low = False
            # Early exit once both are disqualified
            if not is_swing_high and not is_swing_low:
                break

        confirmed_bar = i + n  # pivot becomes known at bar i+n

        if is_swing_high:
            results.append(SwingPoint(
                bar_index     = i,
                timestamp     = timestamps[i],
                price         = pivot_high,
                direction     = "HIGH",
                method        = "pivot",
                length        = n,
                confirmed_bar = confirmed_bar,
            ))

        if is_swing_low:
            results.append(SwingPoint(
                bar_index     = i,
                timestamp     = timestamps[i],
                price         = pivot_low,
                direction     = "LOW",
                method        = "pivot",
                length        = n,
                confirmed_bar = confirmed_bar,
            ))

    results.sort(key=lambda s: s.bar_index)
    return results


# ---------------------------------------------------------------------------
# Method B — Rolling / state-machine (mirrors LuxAlgo os state machine)
# ---------------------------------------------------------------------------

def _detect_rolling_swings(
    highs:      Sequence[float],
    lows:       Sequence[float],
    timestamps: Sequence[int],
    length:     int = 10,
) -> List[SwingPoint]:
    """
    Pine Script equivalent (LuxAlgo OB & Breaker Blocks, Propulsion Block):

        upper = ta.highest(length)
        lower = ta.lowest (length)
        os   := high[length] > upper ? 0 : low[length] < lower ? 1 : os

        if os == 0 and os[1] != 0:   // transition to upswing
            swingHigh = (high[length], bar_index[length])
        if os == 1 and os[1] != 1:   // transition to downswing
            swingLow  = (low[length],  bar_index[length])

    The state variable `os` (oscillator state):
      os = 0  →  price is in an upswing  (last break was upward)
      os = 1  →  price is in a downswing (last break was downward)

    The rolling max/min window looks at the CURRENT bar's window of `length`
    bars. The bar being tested as a potential pivot is `length` bars in the
    past relative to the current bar (index i - length in 0-based arrays).

    A swing HIGH is emitted on the 0→1→0 (upswing) state transition:
      os flips to 0, meaning high[i-length] is above the rolling max of the
      full [i-length+1 .. i] window — so that bar is the highest in the window.

    A swing LOW is emitted on the 1→0→1 (downswing) transition:
      os flips to 1, meaning low[i-length] is below the rolling min.

    Confirmed_bar = current bar i (swing is known immediately on state flip).
    Swing bar     = i - length    (the candle that is actually the pivot).

    Parameters
    ----------
    highs, lows : sequences of float
    timestamps  : sequence of int
    length      : int — rolling window size (default 10; use 3 for PropBlock)

    Returns
    -------
    List[SwingPoint] sorted by bar_index ascending.
    """
    if length < 1:
        raise ValueError(f"rolling length must be >= 1, got {length}")

    bars = len(highs)
    if bars != len(lows) or bars != len(timestamps):
        raise ValueError("highs, lows, timestamps must have the same length")

    results: List[SwingPoint] = []

    # Need at least length+1 bars to start
    if bars <= length:
        return results

    # Initialise os to match Pine Script var os = 0
    os     = 0
    os_prev = 0

    for i in range(length, bars):
        # Rolling max and min of the CURRENT window [i-length+1 .. i]
        # This is ta.highest(length) and ta.lowest(length) at bar i.
        window_highs = highs[i - length + 1 : i + 1]
        window_lows  = lows [i - length + 1 : i + 1]
        upper = max(window_highs)
        lower = min(window_lows)

        # The bar being examined as a potential pivot
        pivot_bar = i - length

        # State update — exact Pine Script logic:
        # os := high[length] > upper ? 0 : low[length] < lower ? 1 : os
        # Note: high[length] in Pine at bar i = highs[i - length] in Python
        if highs[pivot_bar] > upper:
            os = 0
        elif lows[pivot_bar] < lower:
            os = 1
        # else: os remains unchanged

        # Swing HIGH: upswing transition (os was not 0, now is 0)
        if os == 0 and os_prev != 0:
            results.append(SwingPoint(
                bar_index     = pivot_bar,
                timestamp     = timestamps[pivot_bar],
                price         = highs[pivot_bar],
                direction     = "HIGH",
                method        = "rolling",
                length        = length,
                confirmed_bar = i,
            ))

        # Swing LOW: downswing transition (os was not 1, now is 1)
        elif os == 1 and os_prev != 1:
            results.append(SwingPoint(
                bar_index     = pivot_bar,
                timestamp     = timestamps[pivot_bar],
                price         = lows[pivot_bar],
                direction     = "LOW",
                method        = "rolling",
                length        = length,
                confirmed_bar = i,
            ))

        os_prev = os

    results.sort(key=lambda s: s.bar_index)
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_swings(
    highs:           Sequence[float],
    lows:            Sequence[float],
    timestamps:      Sequence[int],
    method:          str = "pivot",
    length:          int = 5,
    pivot_length:    Optional[int] = None,
    rolling_lengths: Optional[List[int]] = None,
) -> List[SwingPoint]:
    """
    Detect swing highs and lows from OHLCV arrays.

    Parameters
    ----------
    highs, lows : array-like of float
        High and low prices, oldest bar first.
    timestamps : array-like of int
        Unix millisecond timestamps, same length as highs/lows.
    method : str
        One of:
          "pivot"   — pivot method only (default)
          "rolling" — rolling/state-machine method only
          "both"    — run both methods; use pivot_length and rolling_lengths
                      to configure each independently
    length : int
        Window size for the chosen method when method is "pivot" or "rolling".
        Ignored when method is "both".
    pivot_length : int, optional
        Window size for the pivot method when method="both". Defaults to 5.
    rolling_lengths : list of int, optional
        One or more window sizes for the rolling method when method="both".
        Defaults to [10, 3] (OB and PropBlock windows).

    Returns
    -------
    List[SwingPoint] sorted by bar_index ascending, then confirmed_bar ascending.

    Examples
    --------
    # Used by BOS/CHoCH detector (TFlab, n=5)
    swings = detect_swings(h, l, ts, method="pivot", length=5)

    # Used by OB detector (LuxAlgo, length=10)
    swings = detect_swings(h, l, ts, method="rolling", length=10)

    # Used by Propulsion Block detector (LuxAlgo, length=3)
    swings = detect_swings(h, l, ts, method="rolling", length=3)

    # All at once for a multi-detector pipeline
    swings = detect_swings(h, l, ts, method="both",
                           pivot_length=5, rolling_lengths=[10, 3])
    """
    highs      = list(highs)
    lows       = list(lows)
    timestamps = list(timestamps)

    if method == "pivot":
        return _detect_pivot_swings(highs, lows, timestamps, n=length)

    elif method == "rolling":
        return _detect_rolling_swings(highs, lows, timestamps, length=length)

    elif method == "both":
        plen  = pivot_length    if pivot_length    is not None else 5
        rlens = rolling_lengths if rolling_lengths is not None else [10, 3]

        all_swings: List[SwingPoint] = []
        all_swings.extend(_detect_pivot_swings(highs, lows, timestamps, n=plen))
        for rlen in rlens:
            all_swings.extend(_detect_rolling_swings(highs, lows, timestamps, length=rlen))

        all_swings.sort(key=lambda s: (s.bar_index, s.confirmed_bar))
        return all_swings

    else:
        raise ValueError(f"method must be 'pivot', 'rolling', or 'both', got {method!r}")


def get_swings_confirmed_by(
    swings:    List[SwingPoint],
    bar_index: int,
    method:    Optional[str] = None,
    length:    Optional[int] = None,
    direction: Optional[str] = None,
) -> List[SwingPoint]:
    """
    Return all swings that were confirmed AT OR BEFORE `bar_index`.

    This is the correct way for downstream detectors to query swings —
    they must only see swings that were knowable at the time of the bar
    being evaluated, preventing look-ahead bias.

    Parameters
    ----------
    swings    : full list of SwingPoints from detect_swings()
    bar_index : current bar being evaluated (inclusive upper bound)
    method    : optional filter — "pivot" or "rolling"
    length    : optional filter — window size
    direction : optional filter — "HIGH" or "LOW"

    Returns
    -------
    List[SwingPoint] matching all filters, confirmed_bar <= bar_index,
    sorted by confirmed_bar ascending (oldest first).
    """
    result = [
        s for s in swings
        if s.confirmed_bar <= bar_index
        and (method    is None or s.method    == method)
        and (length    is None or s.length    == length)
        and (direction is None or s.direction == direction)
    ]
    result.sort(key=lambda s: s.confirmed_bar)
    return result


def get_most_recent_swing(
    swings:    List[SwingPoint],
    bar_index: int,
    direction: str,
    method:    Optional[str] = None,
    length:    Optional[int] = None,
) -> Optional[SwingPoint]:
    """
    Return the most recently CONFIRMED swing in `direction` as of `bar_index`.

    This is what detectors like the OB detector call when they need
    "the last swing high before this bar". Uses confirmed_bar not bar_index
    so there is no look-ahead.

    Parameters
    ----------
    swings    : full list from detect_swings()
    bar_index : current evaluation bar (inclusive)
    direction : "HIGH" or "LOW"
    method    : optional filter
    length    : optional filter

    Returns
    -------
    The SwingPoint with the largest confirmed_bar <= bar_index, or None.
    """
    candidates = get_swings_confirmed_by(
        swings, bar_index,
        method=method, length=length, direction=direction
    )
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    print("=" * 60)
    print("swing_detector.py — self test")
    print("=" * 60)

    # Build a simple synthetic price series: a clean sine wave
    # so we know exactly where highs and lows should be.
    N = 80
    ts_base = 1_700_000_000_000  # arbitrary unix ms start

    # Sine wave: period = 20 bars, amplitude = 10, centre = 100
    prices = [100 + 10 * math.sin(2 * math.pi * i / 20) for i in range(N)]
    # Add a small trend so highs and lows are strict (no exact ties)
    trend = [0.02 * i for i in range(N)]
    mid   = [p + t for p, t in zip(prices, trend)]

    highs = [m + 0.5 for m in mid]
    lows  = [m - 0.5 for m in mid]
    ts    = [ts_base + i * 60_000 for i in range(N)]  # 1-min bars

    # ── Pivot method ────────────────────────────────────────────────
    print("\n── Pivot method (n=5) ─────────────────────────────────────")
    piv_swings = detect_swings(highs, lows, ts, method="pivot", length=5)
    highs_piv  = [s for s in piv_swings if s.direction == "HIGH"]
    lows_piv   = [s for s in piv_swings if s.direction == "LOW"]
    print(f"  Detected {len(highs_piv)} swing highs, {len(lows_piv)} swing lows")
    for s in piv_swings[:6]:
        print(f"  {s}")

    # ── Rolling method, length=10 ────────────────────────────────────
    print("\n── Rolling method (length=10) ──────────────────────────────")
    rol_swings = detect_swings(highs, lows, ts, method="rolling", length=10)
    highs_rol  = [s for s in rol_swings if s.direction == "HIGH"]
    lows_rol   = [s for s in rol_swings if s.direction == "LOW"]
    print(f"  Detected {len(highs_rol)} swing highs, {len(lows_rol)} swing lows")
    for s in rol_swings[:6]:
        print(f"  {s}")

    # ── Rolling method, length=3 ─────────────────────────────────────
    print("\n── Rolling method (length=3) ────────────────────────────────")
    rol3_swings = detect_swings(highs, lows, ts, method="rolling", length=3)
    print(f"  Detected {len([s for s in rol3_swings if s.direction=='HIGH'])} highs, "
          f"{len([s for s in rol3_swings if s.direction=='LOW'])} lows")

    # ── Both at once ──────────────────────────────────────────────────
    print("\n── Both methods combined ────────────────────────────────────")
    both = detect_swings(highs, lows, ts, method="both",
                         pivot_length=5, rolling_lengths=[10, 3])
    print(f"  Total swings: {len(both)}")

    # ── No look-ahead bias check ─────────────────────────────────────
    print("\n── Look-ahead bias check ────────────────────────────────────")
    # At bar 20, only swings confirmed by bar 20 should be visible
    visible_at_20 = get_swings_confirmed_by(piv_swings, bar_index=20)
    print(f"  Pivot swings visible at bar 20: {len(visible_at_20)}")
    for s in visible_at_20:
        assert s.confirmed_bar <= 20, f"LOOK-AHEAD BUG: {s}"
    print("  All confirmed_bar <= 20 ✓")

    # ── Most recent swing query ───────────────────────────────────────
    print("\n── get_most_recent_swing ────────────────────────────────────")
    last_high = get_most_recent_swing(piv_swings, bar_index=50, direction="HIGH")
    last_low  = get_most_recent_swing(piv_swings, bar_index=50, direction="LOW")
    print(f"  Most recent pivot HIGH at bar 50: {last_high}")
    print(f"  Most recent pivot LOW  at bar 50: {last_low}")

    # ── Edge cases ────────────────────────────────────────────────────
    print("\n── Edge cases ───────────────────────────────────────────────")
    short = detect_swings([1.0, 2.0, 1.5], [0.5, 1.5, 1.0],
                          [1000, 2000, 3000], method="pivot", length=5)
    assert short == [], f"Expected [] for too-short input, got {short}"
    print("  Too-short input → [] ✓")

    none_result = get_most_recent_swing([], bar_index=10, direction="HIGH")
    assert none_result is None
    print("  Empty swing list → None ✓")

    print("\n✅ All tests passed")

"""
liquidity_detector.py
=====================
Wicks SMC Backtesting Engine — Layer 3

Detects Liquidity Sweeps — price movements that briefly exceed a prior swing
high/low (grabbing stops), then sharply reverse direction.

Definition (ICT)
----------------
An ICT Liquidity Sweep occurs when price:
  1. Targets a prior swing high or low (where retail stops are clustered).
  2. Exceeds the swing level (either via wick or close).
  3. REVERSES and closes back in the opposite direction.

This is DISTINCT from a Liquidity Run, where price exceeds the level and
CONTINUES in that direction. A sweep reverses.

  Buy-side Liquidity Sweep (Bearish sweep of highs):
    - Price wicks ABOVE a prior swing HIGH.
    - The bar CLOSES BELOW the swept swing high.
    - Signals smart money selling after grabbing buy stops above the high.
    - Bearish reversal expected.

  Sell-side Liquidity Sweep (Bullish sweep of lows):
    - Price wicks BELOW a prior swing LOW.
    - The bar CLOSES ABOVE the swept swing low.
    - Signals smart money buying after grabbing sell stops below the low.
    - Bullish reversal expected.

Confirmation (optional):
  Strong sweeps have a next-bar close in the reversal direction.
  sweep.confirmed = True if close[i+1] continues the reversal.

Sweep quality filter:
  sweep_ratio: how far price exceeded the swing level as % of ATR.
  Default: 0.0 (accept any sweep regardless of size).
  Recommended: 0.1 (price must exceed swing by at least 10% of ATR).

Output
------
  LiquiditySweep dataclass:
    bar_index      int    — bar where the sweep occurred
    timestamp      int
    direction      str    — "BULLISH" (sell-side swept → bullish) or
                            "BEARISH" (buy-side swept → bearish)
    swept_price    float  — the prior swing price that was swept
    sweep_high     float  — the actual high of the sweep bar
    sweep_low      float  — the actual low of the sweep bar
    sweep_excess   float  — how far price went beyond the swing level
    confirmed      bool   — True if next bar confirms reversal direction
    swing_bar      int    — bar index of the swept swing point
    used           bool   — True if already consumed by a trade signal

Usage
-----
  from swing_detector import detect_swings
  from liquidity_detector import detect_sweeps, get_sweeps_at_bar

  swings = detect_swings(highs, lows, timestamps, method="rolling", length=10)
  sweeps = detect_sweeps(opens, highs, lows, closes, timestamps, swings)
  recent = get_sweeps_at_bar(sweeps, bar_index=50, lookback=5)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from swing_detector import SwingPoint


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LiquiditySweep:
    bar_index:    int
    timestamp:    int
    direction:    str     # "BULLISH" (sweep of lows → expect up) or
                          # "BEARISH" (sweep of highs → expect down)
    swept_price:  float   # prior swing price exceeded
    sweep_high:   float
    sweep_low:    float
    sweep_excess: float   # abs(extremity - swept_price)
    confirmed:    bool    # next bar closes in reversal direction
    swing_bar:    int     # bar of the swept swing point
    used:         bool    = False   # consumed by backtest entry

    def __repr__(self):
        conf = "✓" if self.confirmed else "~"
        return (f"LiquiditySweep({self.direction}{conf} swept={self.swept_price:.5f} "
                f"excess={self.sweep_excess:.5f} bar={self.bar_index})")


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_sweeps(
    opens:        Sequence[float],
    highs:        Sequence[float],
    lows:         Sequence[float],
    closes:       Sequence[float],
    timestamps:   Sequence[int],
    swings:       List[SwingPoint],
    sweep_ratio:  float = 0.0,   # min excess as fraction of price (0 = any)
    swing_method: str   = "rolling",
    swing_length: int   = 10,
    require_close_back: bool = True,  # close must be back inside swing level
) -> List[LiquiditySweep]:
    """
    Detect Liquidity Sweeps across the full bar history.

    Parameters
    ----------
    opens, highs, lows, closes : OHLCV arrays, oldest first
    timestamps                 : unix ms per bar
    swings                     : from detect_swings()
    sweep_ratio                : minimum excess beyond swing as fraction of swept_price
    swing_method, swing_length : filter swings
    require_close_back         : if True, bar must close back inside swing level
                                  (strict sweep definition). If False, wick-only sweeps
                                  are also counted.

    Returns
    -------
    List[LiquiditySweep] sorted by bar_index.
    """
    highs  = list(highs)
    lows   = list(lows)
    closes = list(closes)
    opens  = list(opens)
    timestamps = list(timestamps)
    bars   = len(closes)

    # Separate and sort swing highs/lows
    swing_highs = sorted(
        [s for s in swings if s.direction == "HIGH"
         and s.method == swing_method and s.length == swing_length],
        key=lambda s: s.bar_index
    )
    swing_lows = sorted(
        [s for s in swings if s.direction == "LOW"
         and s.method == swing_method and s.length == swing_length],
        key=lambda s: s.bar_index
    )

    sweeps: List[LiquiditySweep] = []
    used_bars: set = set()
    used_swing_bars: set = set()   # prevent multiple sweeps of same swing level

    # Track confirmed swings visible at each bar
    active_highs: List[SwingPoint] = []
    active_lows:  List[SwingPoint] = []
    sh_ptr = 0
    sl_ptr = 0

    for i in range(1, bars):
        # Ingest newly confirmed swings
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr].confirmed_bar <= i:
            active_highs.append(swing_highs[sh_ptr])
            sh_ptr += 1
        while sl_ptr < len(swing_lows) and swing_lows[sl_ptr].confirmed_bar <= i:
            active_lows.append(swing_lows[sl_ptr])
            sl_ptr += 1

        if i in used_bars:
            continue

        h = highs[i]
        l = lows[i]
        c = closes[i]

        # ── Buy-side Liquidity Sweep (bearish outcome) ─────────────────────
        # Price wicks above prior swing HIGH then closes back below
        # Only check the most recent confirmed swing high (not already used this bar)
        recent_highs = [s for s in active_highs if s.bar_index < i and s.bar_index not in used_swing_bars]
        if recent_highs:
            sh = recent_highs[-1]
            if h > sh.price:
                excess = h - sh.price
                if sweep_ratio == 0 or excess >= sh.price * sweep_ratio:
                    close_back = c < sh.price

                    if not require_close_back or close_back:
                        confirmed = False
                        if i + 1 < bars:
                            confirmed = closes[i + 1] < c

                        sweep = LiquiditySweep(
                            bar_index    = i,
                            timestamp    = timestamps[i],
                            direction    = "BEARISH",
                            swept_price  = sh.price,
                            sweep_high   = h,
                            sweep_low    = l,
                            sweep_excess = excess,
                            confirmed    = confirmed,
                            swing_bar    = sh.bar_index,
                        )
                        sweeps.append(sweep)
                        used_bars.add(i)
                        used_swing_bars.add(sh.bar_index)

        if i in used_bars:
            continue

        # ── Sell-side Liquidity Sweep (bullish outcome) ────────────────────
        # Price wicks below prior swing LOW then closes back above
        recent_lows = [s for s in active_lows if s.bar_index < i and s.bar_index not in used_swing_bars]
        if recent_lows and i not in used_bars:
            sl = recent_lows[-1]
            if l < sl.price:
                excess = sl.price - l
                if sweep_ratio == 0 or excess >= sl.price * sweep_ratio:
                    close_back = c > sl.price

                    if not require_close_back or close_back:
                        confirmed = False
                        if i + 1 < bars:
                            confirmed = closes[i + 1] > c

                        sweep = LiquiditySweep(
                            bar_index    = i,
                            timestamp    = timestamps[i],
                            direction    = "BULLISH",
                            swept_price  = sl.price,
                            sweep_high   = h,
                            sweep_low    = l,
                            sweep_excess = excess,
                            confirmed    = confirmed,
                            swing_bar    = sl.bar_index,
                        )
                        sweeps.append(sweep)
                        used_bars.add(i)
                        used_swing_bars.add(sl.bar_index)

    sweeps.sort(key=lambda s: s.bar_index)
    return sweeps


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_sweeps_at_bar(
    sweeps:    List[LiquiditySweep],
    bar_index: int,
    lookback:  int = 10,
    direction: Optional[str] = None,
    confirmed_only: bool = False,
) -> List[LiquiditySweep]:
    """
    Return sweeps within `lookback` bars of bar_index.

    Parameters
    ----------
    lookback       : how many bars back to look (default 10)
    direction      : "BULLISH", "BEARISH", or None for both
    confirmed_only : if True, only return sweeps with confirmed=True
    """
    result = []
    for s in sweeps:
        if s.bar_index > bar_index:
            continue
        if s.bar_index < bar_index - lookback:
            continue
        if direction and s.direction != direction:
            continue
        if confirmed_only and not s.confirmed:
            continue
        result.append(s)
    return result


def get_most_recent_sweep(
    sweeps:    List[LiquiditySweep],
    bar_index: int,
    lookback:  int = 10,
    direction: Optional[str] = None,
) -> Optional[LiquiditySweep]:
    """Return the most recent sweep within lookback bars."""
    candidates = get_sweeps_at_bar(sweeps, bar_index, lookback, direction)
    if not candidates:
        return None
    return max(candidates, key=lambda s: s.bar_index)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from swing_detector import detect_swings

    print("=" * 60)
    print("liquidity_detector.py — self test")
    print("=" * 60)

    N = 100
    ts_base = 1_700_000_000_000

    opens  = [1.0] * N
    highs  = [1.01] * N
    lows   = [0.99] * N
    closes = [1.005] * N

    # Phase 1: build swing high at bar ~10 (price ~1.05)
    for i in range(0, 20):
        v = 1.0 + (10 - abs(i - 10)) * 0.005
        opens[i]  = v - 0.001
        highs[i]  = v + 0.002
        lows[i]   = v - 0.002
        closes[i] = v

    # Phase 2: pullback
    for i in range(20, 35):
        v = 1.05 - (i - 20) * 0.003
        opens[i]  = v + 0.001
        highs[i]  = v + 0.002
        lows[i]   = v - 0.001
        closes[i] = v

    # Phase 3: sweep ABOVE the prior swing high (bar 10 high ~1.05)
    # Bar 40: wicks above 1.05, but CLOSES back below → buy-side sweep
    opens[40]  = 1.04
    highs[40]  = 1.062   # sweeps above 1.05
    lows[40]   = 1.038
    closes[40] = 1.039   # closes back below swept high → bearish sweep signal

    # Phase 4: price drops (confirming the bearish sweep)
    for i in range(41, 60):
        v = 1.039 - (i - 41) * 0.003
        opens[i]  = v + 0.001
        highs[i]  = v + 0.002
        lows[i]   = v - 0.002
        closes[i] = v - 0.001

    # Phase 5: build swing low at bar ~65 (~0.92)
    for i in range(60, 75):
        v = 0.96 - abs(i - 67) * 0.005
        opens[i]  = v + 0.001
        highs[i]  = v + 0.003
        lows[i]   = v - 0.002
        closes[i] = v

    # Phase 6: sweep BELOW the prior swing low
    opens[80]  = 0.955
    lows[80]   = 0.903    # sweeps below prior low (~0.925)
    highs[80]  = 0.960
    closes[80] = 0.958    # closes back above → bullish sweep signal

    for i in range(81, N):
        v = 0.958 + (i - 81) * 0.003
        opens[i]  = v
        highs[i]  = v + 0.002
        lows[i]   = v - 0.001
        closes[i] = v + 0.001

    ts_arr = [ts_base + i * 3_600_000 for i in range(N)]

    swings = detect_swings(highs, lows, ts_arr, method="rolling", length=10)
    sweeps = detect_sweeps(opens, highs, lows, closes, ts_arr, swings)

    print(f"\nSwings detected: {len(swings)}")
    print(f"Sweeps detected: {len(sweeps)}")
    for s in sweeps:
        print(f"  {s}")

    bullish = [s for s in sweeps if s.direction == "BULLISH"]
    bearish = [s for s in sweeps if s.direction == "BEARISH"]
    print(f"\n  Bullish sweeps (sell-side grabbed): {len(bullish)}")
    print(f"  Bearish sweeps (buy-side grabbed):  {len(bearish)}")

    recent = get_sweeps_at_bar(sweeps, 85, lookback=10)
    print(f"\nSweeps within 10 bars of bar 85: {len(recent)}")

    print("\n✅ liquidity_detector self-test complete")

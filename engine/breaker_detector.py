"""
breaker_detector.py
===================
Wicks SMC Backtesting Engine — Layer 4

Detects Breaker Blocks by promoting mitigated Order Blocks that then
get reclaimed by price — the classic "failed OB → flipped zone" concept.

Definition (ICT / LuxAlgo)
--------------------------
A Breaker Block is an Order Block that was mitigated (price closed through
the zone) and then price returns INTO that same zone from the other side,
treating it as a new Point of Interest in the opposite direction.

  Bullish Breaker:
    1. A BEARISH OB exists (top/bottom defined by bearish OB body).
    2. Price closes ABOVE ob.top  → OB is mitigated (bearish OB broken bullish).
    3. Price later pulls back INTO the zone (low <= ob.top AND high >= ob.bottom).
    4. That re-entry bar is the Breaker activation bar.
    5. The Breaker now acts as a BULLISH support zone.

  Bearish Breaker:
    1. A BULLISH OB exists.
    2. Price closes BELOW ob.bottom → OB is mitigated (bullish OB broken bearish).
    3. Price later rallies back INTO the zone (high >= ob.bottom AND low <= ob.top).
    4. That re-entry bar is the Breaker activation bar.
    5. The Breaker now acts as a BEARISH resistance zone.

Invalidation:
  Bullish Breaker: close < breaker.bottom  (price exits through the bottom)
  Bearish Breaker: close > breaker.top     (price exits through the top)

Zone definition:
  Identical to the parent OB zone (top / bottom of OB candle body).
  Midline = (top + bottom) / 2.

Output
------
  BreakerBlock dataclass:
    bar_index       int    — bar where OB was mitigated (promotion bar)
    activation_bar  int    — bar where price re-entered the zone (trade signal bar)
    timestamp       int    — timestamp of activation_bar
    direction       str    — "BULLISH" or "BEARISH"
    top             float
    bottom          float
    midline         float
    parent_ob_bar   int    — ob_bar of the parent OrderBlock
    status          str    — "PENDING" | "ACTIVE" | "MITIGATED"
    mitigated_bar   int | None

Usage
-----
  from ob_detector import detect_obs
  from breaker_detector import detect_breakers, get_active_breakers_at_bar

  obs      = detect_obs(opens, highs, lows, closes, timestamps, swings)
  breakers = detect_breakers(highs, lows, closes, timestamps, obs)
  active   = get_active_breakers_at_bar(breakers, bar_index=100)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ob_detector import OrderBlock


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BreakerBlock:
    bar_index:      int          # bar OB was mitigated (promoted to breaker)
    activation_bar: int          # bar price re-entered zone (signal bar)
    timestamp:      int          # timestamp of activation_bar
    direction:      str          # "BULLISH" or "BEARISH"
    top:            float
    bottom:         float
    midline:        float
    parent_ob_bar:  int

    status:         str          = "ACTIVE"
    mitigated_bar:  Optional[int] = None

    def __repr__(self):
        return (f"BreakerBlock({self.direction} top={self.top:.5f} "
                f"bot={self.bottom:.5f} act={self.activation_bar} "
                f"status={self.status})")

    @property
    def is_active(self) -> bool:
        return self.status == "ACTIVE"


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_breakers(
    highs:      Sequence[float],
    lows:       Sequence[float],
    closes:     Sequence[float],
    timestamps: Sequence[int],
    obs:        List[OrderBlock],
    mitigation_mode: str = "close",   # "close" or "wick"
) -> List[BreakerBlock]:
    """
    Scan all mitigated OBs and detect when price re-enters the zone
    from the opposite side, creating a Breaker Block.

    Parameters
    ----------
    highs, lows, closes : OHLCV arrays, oldest first
    timestamps          : unix ms per bar
    obs                 : output of detect_obs() — full history
    mitigation_mode     : how breaker itself is invalidated

    Returns
    -------
    List[BreakerBlock] sorted by activation_bar.
    """
    highs  = list(highs)
    lows   = list(lows)
    closes = list(closes)
    timestamps = list(timestamps)
    bars   = len(closes)

    breakers: List[BreakerBlock] = []

    for ob in obs:
        # Only promote mitigated OBs
        if ob.status != "MITIGATED" or ob.mitigated_bar is None:
            continue

        mit_bar = ob.mitigated_bar
        top     = ob.top
        bottom  = ob.bottom

        # Scan bars AFTER mitigation for re-entry into the zone
        activation_bar = None
        for j in range(mit_bar + 1, bars):
            h = highs[j]
            l = lows[j]

            if ob.direction == "BEARISH":
                # Bearish OB was broken bullish → now bullish breaker
                # Re-entry: price pulls back into the zone from above
                if l <= top and h >= bottom:
                    activation_bar = j
                    break
            else:
                # Bullish OB was broken bearish → now bearish breaker
                # Re-entry: price rallies back into the zone from below
                if h >= bottom and l <= top:
                    activation_bar = j
                    break

        if activation_bar is None:
            continue

        # Determine breaker direction (opposite of parent OB)
        bb_direction = "BULLISH" if ob.direction == "BEARISH" else "BEARISH"

        bb = BreakerBlock(
            bar_index      = mit_bar,
            activation_bar = activation_bar,
            timestamp      = timestamps[activation_bar],
            direction      = bb_direction,
            top            = top,
            bottom         = bottom,
            midline        = (top + bottom) / 2,
            parent_ob_bar  = ob.ob_bar,
        )
        breakers.append(bb)

        # Mark parent OB
        ob.is_breaker = True

    # Second pass: invalidation
    for bb in breakers:
        for j in range(bb.activation_bar + 1, bars):
            h = highs[j]
            l = lows[j]
            c = closes[j]

            if bb.direction == "BULLISH":
                trigger = c if mitigation_mode == "close" else l
                if trigger < bb.bottom:
                    bb.status        = "MITIGATED"
                    bb.mitigated_bar = j
                    break
            else:
                trigger = c if mitigation_mode == "close" else h
                if trigger > bb.top:
                    bb.status        = "MITIGATED"
                    bb.mitigated_bar = j
                    break

    breakers.sort(key=lambda b: b.activation_bar)
    return breakers


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_active_breakers_at_bar(
    breakers:  List[BreakerBlock],
    bar_index: int,
    direction: Optional[str] = None,
) -> List[BreakerBlock]:
    """
    Return Breakers activated at or before bar_index that are still ACTIVE.
    """
    result = []
    for bb in breakers:
        if bb.activation_bar > bar_index:
            continue
        if direction and bb.direction != direction:
            continue
        if bb.status == "ACTIVE":
            result.append(bb)
        elif bb.status == "MITIGATED" and bb.mitigated_bar > bar_index:
            result.append(bb)
    return result


def get_nearest_breaker(
    breakers:  List[BreakerBlock],
    bar_index: int,
    price:     float,
    direction: Optional[str] = None,
) -> Optional[BreakerBlock]:
    """Return the active Breaker whose midline is closest to price."""
    active = get_active_breakers_at_bar(breakers, bar_index, direction)
    if not active:
        return None
    return min(active, key=lambda b: abs(b.midline - price))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from swing_detector import detect_swings
    from ob_detector import detect_obs

    print("=" * 60)
    print("breaker_detector.py — self test")
    print("=" * 60)

    N = 150
    ts_base = 1_700_000_000_000

    # Build price series:
    # Phase 1 (0-29): rise to swing high
    # Phase 2 (30-49): pullback with bearish candles → bearish OB forms
    # Phase 3 (50-69): big bullish breakout → OB mitigated
    # Phase 4 (70-90): retracement back into OB zone → breaker activation

    opens  = [1.0] * N
    highs  = [1.01] * N
    lows   = [0.99] * N
    closes = [1.005] * N

    # Phase 1: bullish run up to ~1.10
    for i in range(0, 30):
        v = 1.0 + i * 0.004
        opens[i]  = v
        highs[i]  = v + 0.003
        lows[i]   = v - 0.001
        closes[i] = v + 0.002

    # Phase 2: pullback — bearish candles (these become bullish OB candidates)
    for i in range(30, 50):
        v = 1.12 - (i - 30) * 0.002
        opens[i]  = v + 0.002  # open > close = bearish
        highs[i]  = v + 0.004
        lows[i]   = v - 0.001
        closes[i] = v

    # Phase 3: break above swing high (bullish → bearish OB gets a bullish OB confirmed)
    # Then price crashes down — bullish OB gets mitigated
    for i in range(50, 70):
        v = 1.09 + (i - 50) * 0.003  # rises to ~1.15
        opens[i]  = v
        highs[i]  = v + 0.002
        lows[i]   = v - 0.001
        closes[i] = v + 0.001

    # Phase 4: sharp drop — mitigates any bullish OBs formed, creates breaker scenario
    for i in range(70, 100):
        v = 1.21 - (i - 70) * 0.005
        opens[i]  = v + 0.001
        highs[i]  = v + 0.002
        lows[i]   = v - 0.003
        closes[i] = v - 0.002

    # Phase 5: bounce back up into the mitigated OB zone
    for i in range(100, 130):
        v = 1.06 + (i - 100) * 0.003
        opens[i]  = v
        highs[i]  = v + 0.002
        lows[i]   = v - 0.001
        closes[i] = v + 0.001

    # Phase 6: continuation
    for i in range(130, N):
        opens[i]  = 1.15
        highs[i]  = 1.16
        lows[i]   = 1.14
        closes[i] = 1.155

    ts_arr = [ts_base + i * 3_600_000 for i in range(N)]

    swings   = detect_swings(highs, lows, ts_arr, method="rolling", length=10)
    obs      = detect_obs(opens, highs, lows, closes, ts_arr, swings)
    breakers = detect_breakers(highs, lows, closes, ts_arr, obs)

    print(f"\nSwings detected:   {len(swings)}")
    print(f"OBs detected:      {len(obs)}")
    print(f"Breakers detected: {len(breakers)}")

    for bb in breakers:
        print(f"  {bb}")

    active_at_120 = get_active_breakers_at_bar(breakers, 120)
    print(f"\nActive breakers at bar 120: {len(active_at_120)}")

    print("\n✅ breaker_detector self-test complete")

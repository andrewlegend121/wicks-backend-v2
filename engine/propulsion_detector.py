"""
propulsion_detector.py
======================
Wicks SMC Backtesting Engine — Layer 4

Detects Propulsion Blocks — the last opposing candle that traded INTO an
Order Block zone before price displaced away from it.

Definition (ICT)
----------------
An ICT Propulsion Block is the single candlestick that traded into an Order
Block and then price moved away strongly from that area.

  Bullish Propulsion Block:
    - Identified within a bullish OB context.
    - The LAST BEARISH candle before the impulse move UP from the OB zone.
    - This candle's body defines the propulsion zone.
    - Entry zone: above the midline (50% retracement) of propulsion candle body.
    - A good bullish PB holds above its own midline on retests.

  Bearish Propulsion Block:
    - Identified within a bearish OB context.
    - The LAST BULLISH candle before the impulse move DOWN from the OB zone.
    - Entry zone: below the midline (50% retracement) of propulsion candle body.
    - A good bearish PB holds below its own midline on retests.

Detection algorithm
-------------------
For each confirmed Order Block:
  1. Walk backward from ob.bar_index to ob.ob_bar (the actual OB candle).
  2. Find the last candle BEFORE the OB candle that is opposing direction.
     - Bullish OB → find last BEARISH candle before impulse break.
     - Bearish OB → find last BULLISH candle before impulse break.
  3. That candle is the Propulsion Block.
  4. Zone: top = max(open, close), bottom = min(open, close) of that candle.
  5. Midline = (top + bottom) / 2  (the 50% threshold for entries).

Activation:
  PB becomes tradeable when price returns to the zone after the initial move.
  A bullish PB activates when price LOW enters [bottom, top] range.
  A bearish PB activates when price HIGH enters [bottom, top] range.

Invalidation:
  Bullish PB: close < propulsion.bottom
  Bearish PB: close > propulsion.top

Output
------
  PropulsionBlock dataclass:
    bar_index      int    — bar of OB confirmation (same as parent OB)
    pb_bar         int    — bar index of the propulsion candle
    activation_bar int | None  — bar price first re-entered the zone
    timestamp      int    — timestamp of pb_bar
    direction      str    — "BULLISH" or "BEARISH"
    top            float
    bottom         float
    midline        float  — 50% level (key entry threshold)
    parent_ob_bar  int    — ob.ob_bar
    status         str    — "PENDING" | "ACTIVE" | "MITIGATED"
    mitigated_bar  int | None

Usage
-----
  from ob_detector import detect_obs
  from propulsion_detector import detect_propulsions, get_active_propulsions_at_bar

  obs = detect_obs(opens, highs, lows, closes, timestamps, swings)
  pbs = detect_propulsions(opens, highs, lows, closes, timestamps, obs)
  active = get_active_propulsions_at_bar(pbs, bar_index=100)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ob_detector import OrderBlock


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class PropulsionBlock:
    bar_index:      int           # OB confirmation bar
    pb_bar:         int           # propulsion candle bar
    activation_bar: Optional[int] # bar price entered the zone (None = never yet)
    timestamp:      int
    direction:      str           # "BULLISH" or "BEARISH"
    top:            float
    bottom:         float
    midline:        float
    parent_ob_bar:  int

    status:         str           = "PENDING"   # "PENDING"|"ACTIVE"|"MITIGATED"
    mitigated_bar:  Optional[int] = None

    def __repr__(self):
        return (f"PropulsionBlock({self.direction} top={self.top:.5f} "
                f"bot={self.bottom:.5f} pb_bar={self.pb_bar} "
                f"status={self.status})")

    @property
    def is_active(self) -> bool:
        return self.status == "ACTIVE"


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_propulsions(
    opens:      Sequence[float],
    highs:      Sequence[float],
    lows:       Sequence[float],
    closes:     Sequence[float],
    timestamps: Sequence[int],
    obs:        List[OrderBlock],
    mitigation_mode: str = "close",
) -> List[PropulsionBlock]:
    """
    Detect Propulsion Blocks across all confirmed Order Blocks.

    Parameters
    ----------
    opens, highs, lows, closes : OHLCV arrays, oldest first
    timestamps                 : unix ms per bar
    obs                        : output of detect_obs()
    mitigation_mode            : "close" or "wick"

    Returns
    -------
    List[PropulsionBlock] sorted by bar_index.
    """
    opens  = list(opens)
    highs  = list(highs)
    lows   = list(lows)
    closes = list(closes)
    timestamps = list(timestamps)
    bars   = len(closes)

    pbs: List[PropulsionBlock] = []

    for ob in obs:
        ob_candle_bar = ob.ob_bar          # the actual OB candle
        confirm_bar   = ob.bar_index       # bar where swing was broken

        # Walk backward from just before the OB candle to find
        # the last opposing candle
        # Bullish OB: find last bearish candle in [ob_candle_bar-lookback, ob_candle_bar-1]
        # Bearish OB: find last bullish candle

        lookback_start = max(0, ob_candle_bar - 20)  # reasonable lookback window
        pb_bar = None

        if ob.direction == "BULLISH":
            # Last bearish candle before the OB candle
            for j in range(ob_candle_bar - 1, lookback_start - 1, -1):
                if opens[j] > closes[j]:  # bearish
                    pb_bar = j
                    break
        else:
            # Last bullish candle before the OB candle
            for j in range(ob_candle_bar - 1, lookback_start - 1, -1):
                if opens[j] < closes[j]:  # bullish
                    pb_bar = j
                    break

        if pb_bar is None:
            continue

        # Zone = body of the propulsion candle
        top    = max(opens[pb_bar], closes[pb_bar])
        bottom = min(opens[pb_bar], closes[pb_bar])

        # Skip if body is degenerate (doji)
        if top - bottom < 1e-8:
            continue

        pb = PropulsionBlock(
            bar_index      = confirm_bar,
            pb_bar         = pb_bar,
            activation_bar = None,
            timestamp      = timestamps[pb_bar],
            direction      = ob.direction,
            top            = top,
            bottom         = bottom,
            midline        = (top + bottom) / 2,
            parent_ob_bar  = ob.ob_bar,
        )
        pbs.append(pb)

    # Second pass: activation and invalidation scan
    for pb in pbs:
        activated = False
        for j in range(pb.bar_index + 1, bars):
            h = highs[j]
            l = lows[j]
            c = closes[j]

            # Check activation first
            if not activated:
                if pb.direction == "BULLISH":
                    if l <= pb.top and h >= pb.bottom:
                        pb.activation_bar = j
                        pb.status = "ACTIVE"
                        activated = True
                else:
                    if h >= pb.bottom and l <= pb.top:
                        pb.activation_bar = j
                        pb.status = "ACTIVE"
                        activated = True

            # Check invalidation once active
            if activated:
                if pb.direction == "BULLISH":
                    trigger = c if mitigation_mode == "close" else l
                    if trigger < pb.bottom:
                        pb.status        = "MITIGATED"
                        pb.mitigated_bar = j
                        break
                else:
                    trigger = c if mitigation_mode == "close" else h
                    if trigger > pb.top:
                        pb.status        = "MITIGATED"
                        pb.mitigated_bar = j
                        break

    pbs.sort(key=lambda p: p.bar_index)
    return pbs


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_active_propulsions_at_bar(
    pbs:       List[PropulsionBlock],
    bar_index: int,
    direction: Optional[str] = None,
) -> List[PropulsionBlock]:
    """
    Return PropulsionBlocks that are ACTIVE at bar_index.
    """
    result = []
    for pb in pbs:
        if pb.bar_index > bar_index:
            continue
        if direction and pb.direction != direction:
            continue
        if pb.status == "ACTIVE":
            result.append(pb)
        elif pb.status == "MITIGATED" and pb.mitigated_bar > bar_index:
            result.append(pb)
    return result


def get_nearest_propulsion(
    pbs:       List[PropulsionBlock],
    bar_index: int,
    price:     float,
    direction: Optional[str] = None,
) -> Optional[PropulsionBlock]:
    """Return the active PropulsionBlock whose midline is closest to price."""
    active = get_active_propulsions_at_bar(pbs, bar_index, direction)
    if not active:
        return None
    return min(active, key=lambda p: abs(p.midline - price))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from swing_detector import detect_swings
    from ob_detector import detect_obs

    print("=" * 60)
    print("propulsion_detector.py — self test")
    print("=" * 60)

    N = 120
    ts_base = 1_700_000_000_000

    # Build price: bullish trend with OB zone and propulsion candle inside it
    opens  = [1.0] * N
    highs  = [1.01] * N
    lows   = [0.99] * N
    closes = [1.005] * N

    # Phase 1: rise to swing high
    for i in range(0, 20):
        v = 1.0 + i * 0.005
        opens[i]  = v
        highs[i]  = v + 0.003
        lows[i]   = v - 0.001
        closes[i] = v + 0.002

    # Phase 2: bearish consolidation — includes the propulsion candle
    for i in range(20, 35):
        v = 1.10 - (i - 20) * 0.002
        opens[i]  = v + 0.002
        highs[i]  = v + 0.003
        lows[i]   = v - 0.001
        closes[i] = v  # bearish

    # Phase 3: break above swing high (OB confirmed, propulsion bar = bar 34)
    for i in range(35, 70):
        v = 1.07 + (i - 35) * 0.004
        opens[i]  = v
        highs[i]  = v + 0.002
        lows[i]   = v - 0.001
        closes[i] = v + 0.001

    # Phase 4: retracement back into propulsion zone
    for i in range(70, 90):
        v = 1.21 - (i - 70) * 0.003
        opens[i]  = v + 0.001
        highs[i]  = v + 0.002
        lows[i]   = v - 0.002
        closes[i] = v

    # Phase 5: continuation up
    for i in range(90, N):
        v = 1.15 + (i - 90) * 0.003
        opens[i]  = v
        highs[i]  = v + 0.002
        lows[i]   = v - 0.001
        closes[i] = v + 0.001

    ts_arr = [ts_base + i * 3_600_000 for i in range(N)]

    swings = detect_swings(highs, lows, ts_arr, method="rolling", length=10)
    obs    = detect_obs(opens, highs, lows, closes, ts_arr, swings)
    pbs    = detect_propulsions(opens, highs, lows, closes, ts_arr, obs)

    print(f"\nSwings: {len(swings)}  OBs: {len(obs)}  Propulsions: {len(pbs)}")
    for pb in pbs:
        print(f"  {pb}")

    active = get_active_propulsions_at_bar(pbs, bar_index=85)
    print(f"\nActive propulsions at bar 85: {len(active)}")

    print("\n✅ propulsion_detector self-test complete")

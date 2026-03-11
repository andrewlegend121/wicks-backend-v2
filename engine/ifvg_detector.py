"""
ifvg_detector.py
================
Wicks SMC Backtesting Engine — Layer 4

Detects Inversion Fair Value Gaps (IFVGs) — FVGs that were violated by price,
which then flip polarity and act as magnets in the opposite direction.

NAMING NOTE
-----------
"Inversion FVG" (IFVG) = a MITIGATED / VIOLATED FVG acting in opposite direction.
"Implied FVG"          = a different concept (hidden gap from overlapping wicks).
These are separate detectors. This file handles INVERSION FVG only.

Definition (ICT)
----------------
A Fair Value Gap is a 3-candle imbalance (gap between candle[0].low and
candle[2].high for bullish, or candle[0].high and candle[2].low for bearish).

An INVERSION FVG forms when:
  Bullish IFVG:
    1. A BEARISH FVG existed (top > bottom, gap zone defined).
    2. Price CLOSES ABOVE fvg.top  (violates the bearish FVG from below).
    3. The violated zone now acts as SUPPORT — a bullish IFVG.
    4. It will push price higher when retested.

  Bearish IFVG:
    1. A BULLISH FVG existed.
    2. Price CLOSES BELOW fvg.bottom  (violates the bullish FVG from above).
    3. The violated zone now acts as RESISTANCE — a bearish IFVG.
    4. It will push price lower when retested.

The IFVG zone is IDENTICAL to the parent FVG zone (same top/bottom).

Activation:
  IFVG becomes tradeable when price returns INTO the zone after violation.
  Bullish IFVG activation: price LOW enters [bottom, top] (retest from above).
  Bearish IFVG activation: price HIGH enters [bottom, top] (retest from below).

Invalidation:
  Bullish IFVG: close < ifvg.bottom  (exits zone on the wrong side)
  Bearish IFVG: close > ifvg.top

Output
------
  InversionFVG dataclass:
    bar_index       int    — bar where parent FVG was violated (promotion bar)
    activation_bar  int | None — bar price re-entered the zone
    timestamp       int    — timestamp of bar_index
    direction       str    — "BULLISH" or "BEARISH"
    top             float  — same as parent FVG top
    bottom          float  — same as parent FVG bottom
    midline         float
    parent_fvg_bar  int    — bar_index of parent FVG
    gap_size        float
    status          str    — "PENDING" | "ACTIVE" | "MITIGATED"
    mitigated_bar   int | None

Usage
-----
  from fvg_detector import detect_fvgs
  from ifvg_detector import detect_ifvgs, get_active_ifvgs_at_bar

  fvgs  = detect_fvgs(highs, lows, timestamps)
  ifvgs = detect_ifvgs(highs, lows, closes, timestamps, fvgs)
  active = get_active_ifvgs_at_bar(ifvgs, bar_index=100)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence

from fvg_detector import FVG


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class InversionFVG:
    bar_index:      int            # bar where the parent FVG was violated
    activation_bar: Optional[int]  # bar price re-entered zone for the trade
    timestamp:      int
    direction:      str            # "BULLISH" or "BEARISH"
    top:            float
    bottom:         float
    midline:        float
    parent_fvg_bar: int
    gap_size:       float

    status:         str            = "PENDING"
    mitigated_bar:  Optional[int]  = None

    def __repr__(self):
        return (f"InversionFVG({self.direction} top={self.top:.5f} "
                f"bot={self.bottom:.5f} act={self.activation_bar} "
                f"status={self.status})")

    @property
    def is_active(self) -> bool:
        return self.status == "ACTIVE"


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_ifvgs(
    highs:      Sequence[float],
    lows:       Sequence[float],
    closes:     Sequence[float],
    timestamps: Sequence[int],
    fvgs:       List[FVG],
    mitigation_mode: str = "close",
) -> List[InversionFVG]:
    """
    Detect Inversion FVGs from a list of previously detected FVGs.

    Parameters
    ----------
    highs, lows, closes : OHLCV arrays, oldest first
    timestamps          : unix ms per bar
    fvgs                : output of fvg_detector.detect_fvgs()
    mitigation_mode     : how the IFVG itself is invalidated ("close" or "wick")

    Returns
    -------
    List[InversionFVG] sorted by bar_index.
    """
    highs  = list(highs)
    lows   = list(lows)
    closes = list(closes)
    timestamps = list(timestamps)
    bars   = len(closes)

    ifvgs: List[InversionFVG] = []

    for fvg in fvgs:
        violation_bar = None

        # Scan bars after FVG was detected for a close beyond the zone
        for j in range(fvg.bar_index + 1, bars):
            c = closes[j]

            if fvg.direction == "BULLISH":
                # Bullish FVG violated when close < bottom
                if c < fvg.bottom:
                    violation_bar = j
                    ifvg_direction = "BEARISH"
                    break
            else:
                # Bearish FVG violated when close > top
                if c > fvg.top:
                    violation_bar = j
                    ifvg_direction = "BULLISH"
                    break

        if violation_bar is None:
            continue

        ifvg = InversionFVG(
            bar_index      = violation_bar,
            activation_bar = None,
            timestamp      = timestamps[violation_bar],
            direction      = ifvg_direction,
            top            = fvg.top,
            bottom         = fvg.bottom,
            midline        = fvg.midline,
            parent_fvg_bar = fvg.bar_index,
            gap_size       = fvg.gap_size,
        )
        ifvgs.append(ifvg)

    # Second pass: activation and invalidation
    for ifvg in ifvgs:
        activated = False
        for j in range(ifvg.bar_index + 1, bars):
            h = highs[j]
            l = lows[j]
            c = closes[j]

            if not activated:
                if ifvg.direction == "BULLISH":
                    # Retest from above: price enters zone from above (pull back)
                    if l <= ifvg.top and h >= ifvg.bottom:
                        ifvg.activation_bar = j
                        ifvg.status = "ACTIVE"
                        activated = True
                else:
                    # Retest from below: price enters zone from below (rally)
                    if h >= ifvg.bottom and l <= ifvg.top:
                        ifvg.activation_bar = j
                        ifvg.status = "ACTIVE"
                        activated = True

            if activated:
                if ifvg.direction == "BULLISH":
                    trigger = c if mitigation_mode == "close" else l
                    if trigger < ifvg.bottom:
                        ifvg.status        = "MITIGATED"
                        ifvg.mitigated_bar = j
                        break
                else:
                    trigger = c if mitigation_mode == "close" else h
                    if trigger > ifvg.top:
                        ifvg.status        = "MITIGATED"
                        ifvg.mitigated_bar = j
                        break

    ifvgs.sort(key=lambda f: f.bar_index)
    return ifvgs


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_active_ifvgs_at_bar(
    ifvgs:     List[InversionFVG],
    bar_index: int,
    direction: Optional[str] = None,
) -> List[InversionFVG]:
    """Return IFVGs that are ACTIVE at bar_index."""
    result = []
    for ifvg in ifvgs:
        if ifvg.bar_index > bar_index:
            continue
        if direction and ifvg.direction != direction:
            continue
        if ifvg.status == "ACTIVE":
            result.append(ifvg)
        elif ifvg.status == "MITIGATED" and ifvg.mitigated_bar > bar_index:
            result.append(ifvg)
    return result


def get_nearest_ifvg(
    ifvgs:     List[InversionFVG],
    bar_index: int,
    price:     float,
    direction: Optional[str] = None,
) -> Optional[InversionFVG]:
    """Return the active IFVG whose midline is closest to price."""
    active = get_active_ifvgs_at_bar(ifvgs, bar_index, direction)
    if not active:
        return None
    return min(active, key=lambda f: abs(f.midline - price))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from fvg_detector import detect_fvgs

    print("=" * 60)
    print("ifvg_detector.py — self test")
    print("=" * 60)

    N = 100
    ts_base = 1_700_000_000_000

    opens  = [1.0] * N
    highs  = [1.01] * N
    lows   = [0.99] * N
    closes = [1.005] * N

    # Phase 1: create a bullish FVG at bars 10-12
    # candle_2 (bar 10): high = 1.02
    opens[10]  = 1.00; highs[10]  = 1.02; lows[10]  = 0.99; closes[10] = 1.01
    # candle_1 (bar 11): big impulse up
    opens[11]  = 1.01; highs[11]  = 1.08; lows[11]  = 1.00; closes[11] = 1.07
    # candle_0 (bar 12): low = 1.05 → bullish FVG: [high[10]=1.02, low[12]=1.05]
    opens[12]  = 1.07; highs[12]  = 1.09; lows[12]  = 1.05; closes[12] = 1.08

    # Phase 2: price trades sideways around 1.07
    for i in range(13, 30):
        opens[i]  = 1.07; highs[i]  = 1.09; lows[i]  = 1.06; closes[i] = 1.08

    # Phase 3: price DROPS below FVG.bottom (1.02) → FVG violated → becomes bearish IFVG
    for i in range(30, 45):
        v = 1.07 - (i - 30) * 0.006
        opens[i]  = v + 0.002
        highs[i]  = v + 0.003
        lows[i]   = v - 0.003
        closes[i] = v - 0.001   # closes below 1.02 around bar 38

    # Phase 4: dead-cat bounce back into the IFVG zone [1.02, 1.05]
    for i in range(45, 60):
        v = 0.99 + (i - 45) * 0.003
        opens[i]  = v
        highs[i]  = v + 0.002
        lows[i]   = v - 0.001
        closes[i] = v + 0.001

    # Phase 5: resume drop
    for i in range(60, N):
        v = 1.04 - (i - 60) * 0.003
        opens[i]  = v + 0.001
        highs[i]  = v + 0.002
        lows[i]   = v - 0.002
        closes[i] = v - 0.001

    ts_arr = [ts_base + i * 3_600_000 for i in range(N)]

    fvgs  = detect_fvgs(highs, lows, ts_arr, closes)
    ifvgs = detect_ifvgs(highs, lows, closes, ts_arr, fvgs)

    print(f"\nFVGs detected:  {len(fvgs)}")
    print(f"IFVGs detected: {len(ifvgs)}")
    for ifvg in ifvgs:
        print(f"  {ifvg}")

    active_at_55 = get_active_ifvgs_at_bar(ifvgs, 55)
    print(f"\nActive IFVGs at bar 55: {len(active_at_55)}")

    print("\n✅ ifvg_detector self-test complete")

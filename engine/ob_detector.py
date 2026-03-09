"""
ob_detector.py
==============
Wicks SMC Backtesting Engine — Layer 3

Detects Order Blocks (OBs) from OHLCV data and confirmed swing points.

Pine Script ground truth: LuxAlgo Order Blocks & Breaker Blocks indicator

Definition
----------
An Order Block is the last opposing candle before a significant swing break.
"Opposing" means the candle's direction is opposite to the direction of the
resulting move.

  Bullish OB: the last BEARISH candle (open > close) before price breaks
              above a swing HIGH. Walk backward from the swing high to find
              the bearish candle with the LOWEST low — that candle is the OB.
              Zone: top = max(open, close), bottom = min(open, close) of OB candle
                    OR top = high, bottom = low if useBody = False (wick mode)

  Bearish OB: the last BULLISH candle (open < close) before price breaks
              below a swing LOW. Walk backward to find the bullish candle
              with the HIGHEST high.
              Zone: top = max(open, close), bottom = min(open, close)

Swing break trigger (mirrors LuxAlgo rolling method, length=10):
  Bullish OB fires when: close > swingHigh.value  (confirmed swing high price)
  Bearish OB fires when: close < swingLow.value   (confirmed swing low price)

Walk-back window:
  From bar N-1 back to the swing high/low bar (inclusive).
  Among all bearish candles in that window, pick the one with lowest low.
  Among all bullish candles, pick the one with highest high.

Invalidation (mitigation):
  Bullish OB: close < ob.bottom  (close mode, default)
              OR low < ob.bottom  (wick mode)
  Bearish OB: close > ob.top    (close mode)
              OR high > ob.top   (wick mode)

Breaker Block promotion:
  When an OB is mitigated, it becomes a Breaker Block (handled in breaker_detector.py).
  The OB carries a `is_breaker` flag set by that module.

Output
------
  OrderBlock dataclass:
    bar_index      int    — bar where the swing break confirmed this OB
    timestamp      int
    ob_bar         int    — bar index of the actual OB candle
    direction      str    — "BULLISH" or "BEARISH"
    top            float
    bottom         float
    midline        float
    ob_open        float  — open of the OB candle
    ob_close       float  — close of the OB candle
    ob_high        float  — high of the OB candle
    ob_low         float  — low of the OB candle
    swing_value    float  — the swing price that was broken
    swing_bar      int    — bar index of the swing point
    status         str    — "ACTIVE", "MITIGATED"
    mitigated_bar  int | None
    is_breaker     bool   — set True by breaker_detector when mitigated

Usage
-----
  from swing_detector import detect_swings
  from ob_detector import detect_obs, get_active_obs_at_bar

  swings = detect_swings(highs, lows, timestamps, method="rolling", length=10)
  obs = detect_obs(opens, highs, lows, closes, timestamps, swings)
  active = get_active_obs_at_bar(obs, bar_index=100)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from swing_detector import SwingPoint


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class OrderBlock:
    bar_index:     int          # bar where swing break confirmed the OB
    timestamp:     int
    ob_bar:        int          # bar index of the OB candle itself
    direction:     str          # "BULLISH" or "BEARISH"
    top:           float
    bottom:        float
    midline:       float
    ob_open:       float
    ob_close:      float
    ob_high:       float
    ob_low:        float
    swing_value:   float        # swing price that was broken
    swing_bar:     int

    status:        str          = "ACTIVE"   # "ACTIVE", "MITIGATED"
    mitigated_bar: Optional[int] = None
    is_breaker:    bool          = False      # promoted by breaker_detector
    use_body:      bool          = True       # True = body zone, False = wick zone

    def __repr__(self):
        return (f"OrderBlock({self.direction} top={self.top:.5f} "
                f"bot={self.bottom:.5f} ob_bar={self.ob_bar} "
                f"confirmed={self.bar_index} status={self.status})")

    @property
    def is_active(self) -> bool:
        return self.status == "ACTIVE"


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_obs(
    opens:          Sequence[float],
    highs:          Sequence[float],
    lows:           Sequence[float],
    closes:         Sequence[float],
    timestamps:     Sequence[int],
    swings:         List[SwingPoint],
    use_body:       bool = True,
    mitigation_mode: str = "close",   # "close" or "wick"
    method:         str  = "rolling",
    length:         int  = 10,
) -> List[OrderBlock]:
    """
    Detect Order Blocks across the full bar history.

    Parameters
    ----------
    opens, highs, lows, closes : OHLCV arrays, oldest first
    timestamps                 : unix ms per bar
    swings                     : from swing_detector — use rolling, length=10
    use_body                   : True  = zone is candle body (default)
                                 False = zone is full wick (high/low)
    mitigation_mode            : "close" = close beyond zone triggers mitigation
                                 "wick"  = wick beyond zone triggers mitigation
    method, length             : filter swings (rolling + 10 matches LuxAlgo OB)

    Returns
    -------
    List[OrderBlock] with status updated through full history.
    """
    opens  = list(opens)
    highs  = list(highs)
    lows   = list(lows)
    closes = list(closes)
    timestamps = list(timestamps)
    bars   = len(closes)

    if not (bars == len(opens) == len(highs) == len(lows) == len(timestamps)):
        raise ValueError("All arrays must have the same length")

    # Filter to correct swing source
    swing_highs = sorted(
        [s for s in swings if s.direction == "HIGH"
         and s.method == method and s.length == length],
        key=lambda s: s.confirmed_bar
    )
    swing_lows = sorted(
        [s for s in swings if s.direction == "LOW"
         and s.method == method and s.length == length],
        key=lambda s: s.confirmed_bar
    )

    obs: List[OrderBlock] = []
    used_swing_highs = set()   # bar indices of swings already broken → OB created
    used_swing_lows  = set()

    hi_ptr = 0
    lo_ptr = 0
    active_swing_highs: List[SwingPoint] = []
    active_swing_lows:  List[SwingPoint] = []

    for bar_i in range(bars):
        o = opens[bar_i]
        h = highs[bar_i]
        l = lows[bar_i]
        c = closes[bar_i]

        # Ingest newly confirmed swings
        while hi_ptr < len(swing_highs) and swing_highs[hi_ptr].confirmed_bar <= bar_i:
            sh = swing_highs[hi_ptr]
            if sh.bar_index not in used_swing_highs:
                active_swing_highs.append(sh)
            hi_ptr += 1

        while lo_ptr < len(swing_lows) and swing_lows[lo_ptr].confirmed_bar <= bar_i:
            sl = swing_lows[lo_ptr]
            if sl.bar_index not in used_swing_lows:
                active_swing_lows.append(sl)
            lo_ptr += 1

        # ── Bullish OB: close > swing HIGH ─────────────────────────────
        for sh in list(active_swing_highs):
            if c > sh.price and sh.bar_index not in used_swing_highs:
                # Walk back from bar_i-1 to sh.bar_index
                # Find bearish candle (open > close) with lowest low
                ob_bar  = None
                ob_low  = float("inf")

                search_start = max(0, sh.bar_index)
                search_end   = bar_i  # bar_i-1 inclusive

                for j in range(search_end - 1, search_start - 1, -1):
                    if opens[j] > closes[j]:   # bearish candle
                        if lows[j] < ob_low:
                            ob_low = lows[j]
                            ob_bar = j

                if ob_bar is not None:
                    if use_body:
                        top    = max(opens[ob_bar], closes[ob_bar])
                        bottom = min(opens[ob_bar], closes[ob_bar])
                    else:
                        top    = highs[ob_bar]
                        bottom = lows[ob_bar]

                    ob = OrderBlock(
                        bar_index   = bar_i,
                        timestamp   = timestamps[bar_i],
                        ob_bar      = ob_bar,
                        direction   = "BULLISH",
                        top         = top,
                        bottom      = bottom,
                        midline     = (top + bottom) / 2,
                        ob_open     = opens[ob_bar],
                        ob_close    = closes[ob_bar],
                        ob_high     = highs[ob_bar],
                        ob_low      = lows[ob_bar],
                        swing_value = sh.price,
                        swing_bar   = sh.bar_index,
                        use_body    = use_body,
                    )
                    obs.append(ob)
                    used_swing_highs.add(sh.bar_index)

        # ── Bearish OB: close < swing LOW ──────────────────────────────
        for sl in list(active_swing_lows):
            if c < sl.price and sl.bar_index not in used_swing_lows:
                # Walk back — find bullish candle (open < close) with highest high
                ob_bar  = None
                ob_high = float("-inf")

                search_start = max(0, sl.bar_index)

                for j in range(bar_i - 1, search_start - 1, -1):
                    if opens[j] < closes[j]:   # bullish candle
                        if highs[j] > ob_high:
                            ob_high = highs[j]
                            ob_bar  = j

                if ob_bar is not None:
                    if use_body:
                        top    = max(opens[ob_bar], closes[ob_bar])
                        bottom = min(opens[ob_bar], closes[ob_bar])
                    else:
                        top    = highs[ob_bar]
                        bottom = lows[ob_bar]

                    ob = OrderBlock(
                        bar_index   = bar_i,
                        timestamp   = timestamps[bar_i],
                        ob_bar      = ob_bar,
                        direction   = "BEARISH",
                        top         = top,
                        bottom      = bottom,
                        midline     = (top + bottom) / 2,
                        ob_open     = opens[ob_bar],
                        ob_close    = closes[ob_bar],
                        ob_high     = highs[ob_bar],
                        ob_low      = lows[ob_bar],
                        swing_value = sl.price,
                        swing_bar   = sl.bar_index,
                        use_body    = use_body,
                    )
                    obs.append(ob)
                    used_swing_lows.add(sl.bar_index)

        # Clean consumed swings from active lists
        active_swing_highs = [s for s in active_swing_highs
                               if s.bar_index not in used_swing_highs]
        active_swing_lows  = [s for s in active_swing_lows
                               if s.bar_index not in used_swing_lows]

    # ── Second pass: update mitigation ──────────────────────────────────
    for ob in obs:
        for j in range(ob.bar_index + 1, bars):
            h, l, c = highs[j], lows[j], closes[j]

            if ob.direction == "BULLISH":
                trigger = c if mitigation_mode == "close" else l
                if trigger < ob.bottom:
                    ob.status        = "MITIGATED"
                    ob.mitigated_bar = j
                    break

            else:  # BEARISH
                trigger = c if mitigation_mode == "close" else h
                if trigger > ob.top:
                    ob.status        = "MITIGATED"
                    ob.mitigated_bar = j
                    break

    obs.sort(key=lambda ob: ob.bar_index)
    return obs


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_active_obs_at_bar(
    obs:       List[OrderBlock],
    bar_index: int,
    direction: Optional[str] = None,
) -> List[OrderBlock]:
    """
    Return OBs confirmed at or before bar_index that are still ACTIVE.
    Correctly handles OBs that were later mitigated: if mitigated_bar > bar_index
    the OB was still active at bar_index.
    """
    result = []
    for ob in obs:
        if ob.bar_index > bar_index:
            continue
        if direction and ob.direction != direction:
            continue
        if ob.status == "ACTIVE":
            result.append(ob)
        elif ob.status == "MITIGATED" and ob.mitigated_bar > bar_index:
            result.append(ob)
    return result


def get_nearest_ob(
    obs:       List[OrderBlock],
    bar_index: int,
    price:     float,
    direction: Optional[str] = None,
) -> Optional[OrderBlock]:
    """Return the active OB whose midline is closest to price at bar_index."""
    active = get_active_obs_at_bar(obs, bar_index, direction)
    if not active:
        return None
    return min(active, key=lambda ob: abs(ob.midline - price))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math, sys
    sys.path.insert(0, ".")
    from swing_detector import detect_swings

    print("=" * 60)
    print("ob_detector.py — self test")
    print("=" * 60)

    N = 100
    ts_base = 1_700_000_000_000

    # Build a simple uptrend with a pullback containing bearish candles
    # then a break above the swing high → should create a bullish OB
    opens  = [100.0] * N
    highs  = [101.0] * N
    lows   = [99.0]  * N
    closes = [100.5] * N

    # Swing high region (bars 5-14): price rises, makes a high around bar 10
    for i in range(5, 15):
        v = 100 + i * 0.5
        opens[i]  = v
        highs[i]  = v + 1.0
        lows[i]   = v - 0.5
        closes[i] = v + 0.8

    # Pullback (bars 15-25): bearish candles pulling back — these are OB candidates
    for i in range(15, 26):
        v = 107 - (i - 15) * 0.4
        opens[i]  = v + 0.5   # open > close = bearish
        highs[i]  = v + 0.8
        lows[i]   = v - 0.3
        closes[i] = v - 0.1

    # Break above swing high (bar 30+): bullish close above ~107
    for i in range(26, N):
        v = 102 + (i - 26) * 0.5
        opens[i]  = v
        highs[i]  = v + 1.2
        lows[i]   = v - 0.3
        closes[i] = v + 0.9

    ts_arr = [ts_base + i * 60_000 for i in range(N)]

    swings = detect_swings(highs, lows, ts_arr, method="rolling", length=10)
    print(f"\nDetected {len(swings)} swing points:")
    for s in swings:
        print(f"  {s}")

    obs = detect_obs(opens, highs, lows, closes, ts_arr, swings)

    print(f"\nDetected {len(obs)} Order Blocks:")
    for ob in obs:
        print(f"  {ob}")

    bull_obs = [ob for ob in obs if ob.direction == "BULLISH"]
    bear_obs = [ob for ob in obs if ob.direction == "BEARISH"]
    print(f"\n  Bullish OBs: {len(bull_obs)}")
    print(f"  Bearish OBs: {len(bear_obs)}")

    # Active query test
    if obs:
        active_at_50 = get_active_obs_at_bar(obs, 50)
        print(f"\n  Active OBs at bar 50: {len(active_at_50)}")
        for ob in active_at_50:
            print(f"    {ob}")

    assert len(obs) > 0, "Expected at least one OB to be detected"
    print("\n✅ All tests passed")

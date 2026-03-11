"""
rejection_detector.py
=====================
Wicks SMC Backtesting Engine — Layer 4

Detects Rejection Blocks — long wicks formed after liquidity sweeps at
major swing highs/lows that shift market structure.

Definition (ICT)
----------------
An ICT Rejection Block is a zone defined by the long rejection wick(s) that
form at a liquidity sweep of a prior swing high or low, followed by a
Market Structure Shift (MSS) confirming the reversal.

  Bullish Rejection Block:
    1. Price sweeps BELOW a prior swing LOW (sell-side liquidity grab).
    2. The candle that swept forms a long LOWER wick.
    3. Price reverses and closes ABOVE the open of that sweep candle (MSS signal).
    4. The zone is the wick area: from the candle's BODY BOTTOM down to the LOW.
    5. Entry: when price pulls back below the body (into the wick zone) to run
       sell stops — trader buys there.

  Bearish Rejection Block:
    1. Price sweeps ABOVE a prior swing HIGH (buy-side liquidity grab).
    2. The candle that swept forms a long UPPER wick.
    3. Price reverses and closes BELOW the open of that sweep candle (MSS signal).
    4. The zone is the wick area: from the candle's BODY TOP up to the HIGH.
    5. Entry: when price pulls back above the body (into the wick zone) to run
       buy stops — trader sells there.

Detection algorithm
-------------------
For each bar i, check:
  Bullish sweep conditions:
    - low[i] < prior_swing_low_price  (sweeps below prior low)
    - lower_wick = min(open, close) - low > wick_threshold * atr
    - close[i] > open[i]  OR  close[i+1] > high[i]  (bullish follow-through)
    Zone: top = min(open[i], close[i]), bottom = low[i]
    The zone bottom IS the swept liquidity level.

  Bearish sweep conditions:
    - high[i] > prior_swing_high_price  (sweeps above prior high)
    - upper_wick = high - max(open, close) > wick_threshold * atr
    - close[i] < open[i]  OR  close[i+1] < low[i]  (bearish follow-through)
    Zone: top = high[i], bottom = max(open[i], close[i])

Wick quality filter:
  wick_ratio: lower/upper wick must be >= wick_ratio * total_candle_range
  Default: 0.4 (wick must be at least 40% of the candle's full range)

Invalidation:
  Bullish RB: close < rb.bottom  (price violates the sweep low)
  Bearish RB: close > rb.top    (price violates the sweep high)

Output
------
  RejectionBlock dataclass:
    bar_index      int    — bar where the sweep+rejection occurred
    timestamp      int
    direction      str    — "BULLISH" or "BEARISH"
    top            float  — upper bound of rejection zone
    bottom         float  — lower bound (= swept low for bullish)
    body_top       float  — top of the sweep candle's body
    body_bottom    float  — bottom of the sweep candle's body
    midline        float
    swept_price    float  — the prior swing price that was swept
    wick_size      float  — size of the rejection wick
    status         str    — "ACTIVE" | "MITIGATED"
    mitigated_bar  int | None

Usage
-----
  from swing_detector import detect_swings
  from rejection_detector import detect_rejections, get_active_rejections_at_bar

  swings = detect_swings(highs, lows, timestamps, method="rolling", length=10)
  rbs = detect_rejections(opens, highs, lows, closes, timestamps, swings)
  active = get_active_rejections_at_bar(rbs, bar_index=100)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence

from swing_detector import SwingPoint


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RejectionBlock:
    bar_index:    int
    timestamp:    int
    direction:    str        # "BULLISH" or "BEARISH"
    top:          float      # upper bound of full rejection zone
    bottom:       float      # lower bound of full rejection zone
    body_top:     float      # candle body top
    body_bottom:  float      # candle body bottom
    midline:      float
    swept_price:  float      # prior swing price that was swept
    wick_size:    float      # size of rejection wick

    status:       str        = "ACTIVE"
    mitigated_bar: Optional[int] = None

    def __repr__(self):
        return (f"RejectionBlock({self.direction} top={self.top:.5f} "
                f"bot={self.bottom:.5f} wick={self.wick_size:.5f} "
                f"bar={self.bar_index} status={self.status})")

    @property
    def is_active(self) -> bool:
        return self.status == "ACTIVE"


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def _atr(highs: list, lows: list, closes: list, bar: int, period: int = 14) -> float:
    """Simple ATR calculation at bar."""
    start = max(1, bar - period + 1)
    trs = []
    for i in range(start, bar + 1):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i]  - closes[i - 1]),
        )
        trs.append(tr)
    return sum(trs) / len(trs) if trs else highs[bar] - lows[bar]


def detect_rejections(
    opens:        Sequence[float],
    highs:        Sequence[float],
    lows:         Sequence[float],
    closes:       Sequence[float],
    timestamps:   Sequence[int],
    swings:       List[SwingPoint],
    wick_ratio:   float = 0.40,   # wick must be >= 40% of candle range
    swing_method: str   = "rolling",
    swing_length: int   = 10,
    mitigation_mode: str = "close",
) -> List[RejectionBlock]:
    """
    Detect Rejection Blocks across the full bar history.

    Parameters
    ----------
    opens, highs, lows, closes : OHLCV arrays, oldest first
    timestamps                 : unix ms per bar
    swings                     : from detect_swings()
    wick_ratio                 : minimum wick-to-range ratio (0.0 = no filter)
    swing_method, swing_length : filter swings
    mitigation_mode            : "close" or "wick"

    Returns
    -------
    List[RejectionBlock] sorted by bar_index.
    """
    opens  = list(opens)
    highs  = list(highs)
    lows   = list(lows)
    closes = list(closes)
    timestamps = list(timestamps)
    bars   = len(closes)

    # Build sorted swing lists
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

    rbs: List[RejectionBlock] = []
    used_bars: set = set()

    # Track the most recent confirmed swing high/low prices
    latest_sh_price: Optional[float] = None
    latest_sl_price: Optional[float] = None
    sh_ptr = 0
    sl_ptr = 0

    for i in range(2, bars - 1):
        # Update latest confirmed swings (confirmed_bar <= i)
        while sh_ptr < len(swing_highs) and swing_highs[sh_ptr].confirmed_bar <= i:
            latest_sh_price = swing_highs[sh_ptr].price
            sh_ptr += 1
        while sl_ptr < len(swing_lows) and swing_lows[sl_ptr].confirmed_bar <= i:
            latest_sl_price = swing_lows[sl_ptr].price
            sl_ptr += 1

        if i in used_bars:
            continue

        o = opens[i]
        h = highs[i]
        l = lows[i]
        c = closes[i]
        candle_range = h - l
        if candle_range < 1e-10:
            continue

        body_top    = max(o, c)
        body_bottom = min(o, c)

        # ── Bullish Rejection Block ────────────────────────────────────────
        if latest_sl_price is not None and l < latest_sl_price:
            lower_wick = body_bottom - l
            wick_ratio_actual = lower_wick / candle_range

            if wick_ratio_actual >= wick_ratio and lower_wick > 0:
                # Check bullish follow-through: current close bullish OR next bar closes above high
                follow_through = (c > o) or (i + 1 < bars and closes[i + 1] > h)

                if follow_through:
                    rb = RejectionBlock(
                        bar_index   = i,
                        timestamp   = timestamps[i],
                        direction   = "BULLISH",
                        top         = body_bottom,   # top of the wick zone = body bottom
                        bottom      = l,             # actual low (swept level)
                        body_top    = body_top,
                        body_bottom = body_bottom,
                        midline     = (body_bottom + l) / 2,
                        swept_price = latest_sl_price,
                        wick_size   = lower_wick,
                    )
                    rbs.append(rb)
                    used_bars.add(i)
                    continue

        # ── Bearish Rejection Block ────────────────────────────────────────
        if latest_sh_price is not None and h > latest_sh_price:
            upper_wick = h - body_top
            wick_ratio_actual = upper_wick / candle_range

            if wick_ratio_actual >= wick_ratio and upper_wick > 0:
                follow_through = (c < o) or (i + 1 < bars and closes[i + 1] < l)

                if follow_through:
                    rb = RejectionBlock(
                        bar_index   = i,
                        timestamp   = timestamps[i],
                        direction   = "BEARISH",
                        top         = h,             # actual high (swept level)
                        bottom      = body_top,      # bottom of wick zone = body top
                        body_top    = body_top,
                        body_bottom = body_bottom,
                        midline     = (h + body_top) / 2,
                        swept_price = latest_sh_price,
                        wick_size   = upper_wick,
                    )
                    rbs.append(rb)
                    used_bars.add(i)

    # Second pass: invalidation
    for rb in rbs:
        for j in range(rb.bar_index + 1, bars):
            h = highs[j]
            l = lows[j]
            c = closes[j]

            if rb.direction == "BULLISH":
                trigger = c if mitigation_mode == "close" else l
                if trigger < rb.bottom:
                    rb.status        = "MITIGATED"
                    rb.mitigated_bar = j
                    break
            else:
                trigger = c if mitigation_mode == "close" else h
                if trigger > rb.top:
                    rb.status        = "MITIGATED"
                    rb.mitigated_bar = j
                    break

    rbs.sort(key=lambda r: r.bar_index)
    return rbs


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_active_rejections_at_bar(
    rbs:       List[RejectionBlock],
    bar_index: int,
    direction: Optional[str] = None,
) -> List[RejectionBlock]:
    """Return RejectionBlocks active at bar_index."""
    result = []
    for rb in rbs:
        if rb.bar_index > bar_index:
            continue
        if direction and rb.direction != direction:
            continue
        if rb.status == "ACTIVE":
            result.append(rb)
        elif rb.status == "MITIGATED" and rb.mitigated_bar > bar_index:
            result.append(rb)
    return result


def get_nearest_rejection(
    rbs:       List[RejectionBlock],
    bar_index: int,
    price:     float,
    direction: Optional[str] = None,
) -> Optional[RejectionBlock]:
    """Return the active RejectionBlock whose midline is closest to price."""
    active = get_active_rejections_at_bar(rbs, bar_index, direction)
    if not active:
        return None
    return min(active, key=lambda r: abs(r.midline - price))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from swing_detector import detect_swings

    print("=" * 60)
    print("rejection_detector.py — self test")
    print("=" * 60)

    N = 100
    ts_base = 1_700_000_000_000

    opens  = [1.0] * N
    highs  = [1.01] * N
    lows   = [0.99] * N
    closes = [1.005] * N

    # Phase 1: establish a swing low at bar 10 (price = ~0.95)
    for i in range(0, 20):
        v = 1.0 - abs(i - 10) * 0.005
        opens[i]  = v + 0.001
        highs[i]  = v + 0.003
        lows[i]   = v - 0.001
        closes[i] = v

    # Phase 2: rise after swing low
    for i in range(20, 40):
        v = 0.95 + (i - 20) * 0.004
        opens[i]  = v
        highs[i]  = v + 0.002
        lows[i]   = v - 0.001
        closes[i] = v + 0.001

    # Phase 3: sweep BELOW the prior swing low with a long wick
    # Bar 50: sweeps below 0.95 with a strong bullish candle (pin bar)
    opens[50]  = 0.96    # opens bullish
    lows[50]   = 0.935   # sweeps below prior swing low (0.95)
    closes[50] = 0.975   # closes bullish above open
    highs[50]  = 0.980

    # Phase 4: continuation up (MSS)
    for i in range(51, N):
        v = 0.975 + (i - 51) * 0.003
        opens[i]  = v
        highs[i]  = v + 0.002
        lows[i]   = v - 0.001
        closes[i] = v + 0.001

    ts_arr = [ts_base + i * 3_600_000 for i in range(N)]

    swings = detect_swings(highs, lows, ts_arr, method="rolling", length=10)
    rbs    = detect_rejections(opens, highs, lows, closes, ts_arr, swings,
                               wick_ratio=0.30)

    print(f"\nSwings detected: {len(swings)}")
    print(f"Rejection Blocks detected: {len(rbs)}")
    for rb in rbs:
        print(f"  {rb}")

    active_at_70 = get_active_rejections_at_bar(rbs, 70)
    print(f"\nActive at bar 70: {len(active_at_70)}")

    print("\n✅ rejection_detector self-test complete")

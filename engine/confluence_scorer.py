"""
confluence_scorer.py
====================
Wicks SMC Backtesting Engine — Layer 6

Evaluates how many of the user's selected confluence conditions align
at a specific bar, and returns a score + breakdown.

A trade is triggered when:
  1. The score meets or exceeds the user's required minimum (default: 2)
  2. The trend filter passes (if enabled)
  3. A valid entry zone exists (OB or FVG that price is touching)

Confluence items supported (Phase 1 — Layers 2-3):
  - trend       : HTF trend matches trade direction
  - bos         : recent BOS in trade direction
  - choch       : recent CHoCH in trade direction (reversal mode)
  - fvg         : active untouched/partial FVG in trade direction at current price
  - ob          : active OB in trade direction at current price
  - fvg_ob      : price inside BOTH an FVG and an OB simultaneously (premium confluence)

Each item scores 1 point. fvg_ob scores 2 (counts as both).

Output
------
  ConfluenceResult dataclass:
    score        int
    max_score    int   — total possible from selected items
    passed       bool  — score >= min_required
    direction    str   — "BULLISH" or "BEARISH"
    items        dict  — {confluence_id: bool} for each selected item
    entry_zone   tuple — (top, bottom) of the best entry zone, or None
    entry_type   str   — "OB", "FVG", "OB+FVG", or None
    notes        list  — human-readable explanation strings

Usage
-----
  from confluence_scorer import score_bar, ConfluenceResult

  result = score_bar(
      bar_index   = 150,
      price       = 1.2345,
      direction   = "BULLISH",
      selected    = ["trend", "bos", "fvg", "ob"],
      min_required= 2,
      trend_states= trend_states,
      bos_events  = bos_events,
      fvgs        = fvgs,
      obs         = obs,
      lookback    = 20,   # bars to look back for recent BOS/CHoCH
  )
  if result.passed:
      entry_top, entry_bot = result.entry_zone
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from trend_detector import TrendState
from bos_choch_detector import BosChochEvent
from fvg_detector import FVG
from ob_detector import OrderBlock


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ConfluenceResult:
    score:       int
    max_score:   int
    passed:      bool
    direction:   str                        # "BULLISH" or "BEARISH"
    items:       Dict[str, bool]            # {confluence_id: passed}
    entry_zone:  Optional[Tuple[float, float]]  # (top, bottom)
    entry_type:  Optional[str]              # "OB", "FVG", "OB+FVG"
    notes:       List[str] = field(default_factory=list)

    @property
    def score_pct(self) -> float:
        return (self.score / self.max_score * 100) if self.max_score > 0 else 0.0

    def __repr__(self):
        return (f"ConfluenceResult(score={self.score}/{self.max_score} "
                f"passed={self.passed} dir={self.direction} "
                f"entry={self.entry_type})")


# ---------------------------------------------------------------------------
# Zone proximity helper
# ---------------------------------------------------------------------------

def _price_in_zone(price: float, top: float, bottom: float,
                   tolerance_pct: float = 0.001) -> bool:
    """
    True if price is within or very near the zone [bottom, top].
    tolerance_pct: allow price to be this % above/below zone edges.
    Default 0.1% — catches price approaching zone without requiring exact touch.
    """
    tol = (top - bottom) * tolerance_pct + (top * 0.0001)
    return (bottom - tol) <= price <= (top + tol)


def _zone_overlap(top1, bot1, top2, bot2) -> Optional[Tuple[float, float]]:
    """Return overlap zone (top, bottom) of two zones, or None if no overlap."""
    overlap_top = min(top1, top2)
    overlap_bot = max(bot1, bot2)
    if overlap_top > overlap_bot:
        return (overlap_top, overlap_bot)
    return None


# ---------------------------------------------------------------------------
# Individual confluence checks
# ---------------------------------------------------------------------------

def _check_trend(
    bar_index:    int,
    direction:    str,
    trend_states: List[TrendState],
) -> Tuple[bool, str]:
    if bar_index >= len(trend_states):
        return False, "Trend: no data"
    state = trend_states[bar_index]
    passed = state.trend == direction
    return passed, f"Trend: {state.trend} ({'✓' if passed else '✗'} need {direction})"


def _check_bos(
    bar_index:  int,
    direction:  str,
    bos_events: List[BosChochEvent],
    lookback:   int = 20,
) -> Tuple[bool, str]:
    recent = [
        e for e in bos_events
        if e.event_type == "BOS"
        and e.direction == direction
        and (bar_index - lookback) <= e.bar_index <= bar_index
    ]
    passed = len(recent) > 0
    note = f"BOS ({direction}): {'found at bar ' + str(recent[-1].bar_index) if passed else 'none in last ' + str(lookback) + ' bars'}"
    return passed, note


def _check_choch(
    bar_index:  int,
    direction:  str,
    bos_events: List[BosChochEvent],
    lookback:   int = 20,
) -> Tuple[bool, str]:
    recent = [
        e for e in bos_events
        if e.event_type == "CHoCH"
        and e.direction == direction
        and (bar_index - lookback) <= e.bar_index <= bar_index
    ]
    passed = len(recent) > 0
    note = f"CHoCH ({direction}): {'found at bar ' + str(recent[-1].bar_index) if passed else 'none in last ' + str(lookback) + ' bars'}"
    return passed, note


def _check_fvg(
    bar_index: int,
    price:     float,
    direction: str,
    fvgs:      List[FVG],
    tolerance: float = 0.001,
) -> Tuple[bool, str, Optional[Tuple[float, float]]]:
    """Returns (passed, note, zone) — zone is (top, bottom) or None."""
    active = [
        f for f in fvgs
        if f.bar_index <= bar_index
        and f.direction == direction
        and f.is_active
        and (f.mitigated_bar is None or f.mitigated_bar > bar_index)
    ]
    # Find FVGs where price is at or inside the zone
    touching = [
        f for f in active
        if _price_in_zone(price, f.top, f.bottom, tolerance)
    ]
    if touching:
        # Use the most recently formed one
        best = max(touching, key=lambda f: f.bar_index)
        return True, f"FVG ({direction}): price in zone [{best.bottom:.5f}-{best.top:.5f}]", (best.top, best.bottom)
    return False, f"FVG ({direction}): no active zone at price {price:.5f}", None


def _check_ob(
    bar_index: int,
    price:     float,
    direction: str,
    obs:       List[OrderBlock],
    tolerance: float = 0.001,
) -> Tuple[bool, str, Optional[Tuple[float, float]]]:
    """Returns (passed, note, zone)."""
    active = [
        ob for ob in obs
        if ob.bar_index <= bar_index
        and ob.direction == direction
        and ob.is_active
        and (ob.mitigated_bar is None or ob.mitigated_bar > bar_index)
    ]
    touching = [
        ob for ob in active
        if _price_in_zone(price, ob.top, ob.bottom, tolerance)
    ]
    if touching:
        best = max(touching, key=lambda ob: ob.bar_index)
        return True, f"OB ({direction}): price in zone [{best.bottom:.5f}-{best.top:.5f}]", (best.top, best.bottom)
    return False, f"OB ({direction}): no active zone at price {price:.5f}", None


def _check_fvg_ob_overlap(
    fvg_zone: Optional[Tuple[float, float]],
    ob_zone:  Optional[Tuple[float, float]],
) -> Tuple[bool, str, Optional[Tuple[float, float]]]:
    """Check if both FVG and OB are active and overlapping (premium confluence)."""
    if fvg_zone is None or ob_zone is None:
        return False, "FVG+OB overlap: one or both zones missing", None
    overlap = _zone_overlap(fvg_zone[0], fvg_zone[1], ob_zone[0], ob_zone[1])
    if overlap:
        return True, f"FVG+OB overlap: [{overlap[1]:.5f}-{overlap[0]:.5f}]", overlap
    return False, "FVG+OB overlap: zones don't overlap", None


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

# All supported confluence IDs and their display labels
CONFLUENCE_REGISTRY = {
    "trend":   "Trend Alignment",
    "bos":     "Break of Structure",
    "choch":   "Change of Character",
    "fvg":     "Fair Value Gap",
    "ob":      "Order Block",
    "fvg_ob":  "FVG + OB Overlap",
}

# Items that contribute to entry zone (not scoring items)
ZONE_ITEMS = {"fvg", "ob", "fvg_ob"}


def score_bar(
    bar_index:    int,
    price:        float,
    direction:    str,
    selected:     List[str],
    min_required: int                    = 2,
    trend_states: Optional[List[TrendState]]      = None,
    bos_events:   Optional[List[BosChochEvent]]   = None,
    fvgs:         Optional[List[FVG]]             = None,
    obs:          Optional[List[OrderBlock]]       = None,
    lookback:     int                    = 20,
    zone_tolerance: float                = 0.001,
) -> ConfluenceResult:
    """
    Score a single bar against the user's selected confluence conditions.

    Parameters
    ----------
    bar_index    : current bar being evaluated
    price        : current close price (used for zone proximity checks)
    direction    : "BULLISH" or "BEARISH" — direction of the proposed trade
    selected     : list of confluence IDs the user has toggled ON
                   e.g. ["trend", "bos", "fvg", "ob"]
    min_required : minimum score to consider the bar a valid setup
    trend_states : from trend_detector.detect_trend()
    bos_events   : from bos_choch_detector.detect_bos_choch()
    fvgs         : from fvg_detector.detect_fvgs()
    obs          : from ob_detector.detect_obs()
    lookback     : bars to look back for recent BOS/CHoCH events
    zone_tolerance: price proximity tolerance for zone checks (0.001 = 0.1%)

    Returns
    -------
    ConfluenceResult
    """
    if direction not in ("BULLISH", "BEARISH"):
        raise ValueError(f"direction must be 'BULLISH' or 'BEARISH', got {direction!r}")

    items:  Dict[str, bool] = {}
    notes:  List[str]       = []
    score   = 0
    fvg_zone: Optional[Tuple[float, float]] = None
    ob_zone:  Optional[Tuple[float, float]] = None

    # ── Evaluate each selected confluence ───────────────────────────────

    if "trend" in selected:
        if trend_states is None:
            items["trend"] = False
            notes.append("Trend: no trend_states provided")
        else:
            passed, note = _check_trend(bar_index, direction, trend_states)
            items["trend"] = passed
            notes.append(note)
            if passed:
                score += 1

    if "bos" in selected:
        if bos_events is None:
            items["bos"] = False
            notes.append("BOS: no bos_events provided")
        else:
            passed, note = _check_bos(bar_index, direction, bos_events, lookback)
            items["bos"] = passed
            notes.append(note)
            if passed:
                score += 1

    if "choch" in selected:
        if bos_events is None:
            items["choch"] = False
            notes.append("CHoCH: no bos_events provided")
        else:
            passed, note = _check_choch(bar_index, direction, bos_events, lookback)
            items["choch"] = passed
            notes.append(note)
            if passed:
                score += 1

    if "fvg" in selected:
        if fvgs is None:
            items["fvg"] = False
            notes.append("FVG: no fvgs provided")
            fvg_zone = None
        else:
            passed, note, fvg_zone = _check_fvg(
                bar_index, price, direction, fvgs, zone_tolerance)
            items["fvg"] = passed
            notes.append(note)
            if passed:
                score += 1

    if "ob" in selected:
        if obs is None:
            items["ob"] = False
            notes.append("OB: no obs provided")
            ob_zone = None
        else:
            passed, note, ob_zone = _check_ob(
                bar_index, price, direction, obs, zone_tolerance)
            items["ob"] = passed
            notes.append(note)
            if passed:
                score += 1

    if "fvg_ob" in selected:
        passed, note, overlap_zone = _check_fvg_ob_overlap(fvg_zone, ob_zone)
        items["fvg_ob"] = passed
        notes.append(note)
        if passed:
            score += 2   # premium confluence — counts double

    # ── Determine best entry zone ────────────────────────────────────────

    entry_zone: Optional[Tuple[float, float]] = None
    entry_type: Optional[str] = None

    if items.get("fvg_ob"):
        _, _, overlap = _check_fvg_ob_overlap(fvg_zone, ob_zone)
        entry_zone = overlap
        entry_type = "OB+FVG"
    elif items.get("ob"):
        entry_zone = ob_zone
        entry_type = "OB"
    elif items.get("fvg"):
        entry_zone = fvg_zone
        entry_type = "FVG"

    # ── Max score calculation ────────────────────────────────────────────
    max_score = 0
    for item_id in selected:
        if item_id in CONFLUENCE_REGISTRY:
            max_score += 2 if item_id == "fvg_ob" else 1

    passed = score >= min_required

    return ConfluenceResult(
        score      = score,
        max_score  = max_score,
        passed     = passed,
        direction  = direction,
        items      = items,
        entry_zone = entry_zone,
        entry_type = entry_type,
        notes      = notes,
    )


def score_both_directions(
    bar_index:    int,
    price:        float,
    selected:     List[str],
    min_required: int = 2,
    **kwargs,
) -> Dict[str, ConfluenceResult]:
    """
    Score both BULLISH and BEARISH at the same bar.
    Returns {"BULLISH": result, "BEARISH": result}.
    Useful for the backtest runner when scanning without a directional bias.
    """
    return {
        "BULLISH": score_bar(bar_index, price, "BULLISH", selected, min_required, **kwargs),
        "BEARISH": score_bar(bar_index, price, "BEARISH", selected, min_required, **kwargs),
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math, sys
    sys.path.insert(0, ".")
    from swing_detector import detect_swings
    from trend_detector import detect_trend
    from bos_choch_detector import detect_bos_choch
    from fvg_detector import detect_fvgs
    from ob_detector import detect_obs

    print("=" * 60)
    print("confluence_scorer.py — self test")
    print("=" * 60)

    N = 150
    ts_base = 1_700_000_000_000

    def mp(i):
        if i < 80:
            return 100 + i * 0.25 + 5 * math.sin(2 * math.pi * i / 18)
        return 120 - (i - 80) * 0.2 + 5 * math.sin(2 * math.pi * i / 18)

    mid    = [mp(i) for i in range(N)]
    opens  = [m - 0.15 for m in mid]
    highs  = [m + 0.9  for m in mid]
    lows   = [m - 0.9  for m in mid]
    closes = mid
    ts_arr = [ts_base + i * 300_000 for i in range(N)]

    swings_piv  = detect_swings(highs, lows, ts_arr, method="pivot",   length=5)
    swings_roll = detect_swings(highs, lows, ts_arr, method="rolling", length=10)
    trend_states = detect_trend(swings_piv, ts_arr)
    bos_events   = detect_bos_choch(highs, lows, closes, ts_arr, swings_piv, trend_states)
    fvgs         = detect_fvgs(highs, lows, closes, ts_arr)
    obs          = detect_obs(opens, highs, lows, closes, ts_arr, swings_roll)

    print(f"\nSetup: {len(swings_piv)} pivot swings, {len(bos_events)} BOS/CHoCH events")
    print(f"       {len(fvgs)} FVGs, {len(obs)} OBs")

    # Test 1: score at bar 70 (should be in uptrend)
    bar = 70
    price = closes[bar]
    result = score_bar(
        bar_index    = bar,
        price        = price,
        direction    = "BULLISH",
        selected     = ["trend", "bos", "fvg", "ob"],
        min_required = 2,
        trend_states = trend_states,
        bos_events   = bos_events,
        fvgs         = fvgs,
        obs          = obs,
        lookback     = 30,
        zone_tolerance = 0.05,   # wide tolerance for synthetic data
    )
    print(f"\nTest 1 — bar={bar} price={price:.3f} BULLISH:")
    print(f"  Score: {result.score}/{result.max_score}  passed={result.passed}")
    for note in result.notes:
        print(f"  {note}")

    # Test 2: score both directions
    results = score_both_directions(
        bar_index    = bar,
        price        = price,
        selected     = ["trend", "bos", "fvg"],
        min_required = 2,
        trend_states = trend_states,
        bos_events   = bos_events,
        fvgs         = fvgs,
        lookback     = 30,
        zone_tolerance = 0.05,
    )
    print(f"\nTest 2 — score_both_directions at bar {bar}:")
    for d, r in results.items():
        print(f"  {d}: {r.score}/{r.max_score} passed={r.passed}")

    # Test 3: missing data gracefully handled
    result3 = score_bar(
        bar_index    = 50,
        price        = 110.0,
        direction    = "BULLISH",
        selected     = ["trend", "bos", "fvg", "ob"],
        min_required = 2,
        # Deliberately omit all data sources
    )
    print(f"\nTest 3 — no data provided: score={result3.score} passed={result3.passed}")
    assert result3.score == 0
    assert result3.passed == False

    # Test 4: fvg_ob premium scoring
    result4 = score_bar(
        bar_index    = bar,
        price        = price,
        direction    = "BULLISH",
        selected     = ["fvg", "ob", "fvg_ob"],
        min_required = 2,
        fvgs         = fvgs,
        obs          = obs,
        zone_tolerance = 0.05,
    )
    print(f"\nTest 4 — fvg_ob premium: score={result4.score}/{result4.max_score}")
    assert result4.max_score == 4, f"fvg_ob should give max_score=4, got {result4.max_score}"

    print("\n✅ All tests passed")

"""
backtest_runner.py
==================
Wicks SMC Backtesting Engine — Layer 6

The main backtest engine. Takes OHLCV data + user config and produces
a complete set of trade results.

Pipeline
--------
  1. Run all detectors (swing → trend → BOS/CHoCH → FVG → OB)
  2. Scan every bar for confluence setups matching user's selected filters
  3. On a valid setup, enter a trade with risk/reward from user config
  4. Manage open trades bar-by-bar (stop loss, take profit, trailing)
  5. Return trades + aggregate stats

Trade Entry Logic
-----------------
  - Entry triggers when confluence score >= min_required at bar close
  - Entry price = close of the trigger bar (market order on next open
    is a future enhancement; close is conservative and accurate)
  - Stop loss:
      BULLISH: bottom of entry zone - (atr * sl_atr_mult)
      BEARISH: top  of entry zone + (atr * sl_atr_mult)
  - Take profit:
      Entry ± (stop_distance * rr_ratio)

Trade Management
----------------
  - One trade open at a time per direction (configurable)
  - Trades close at:
      1. Stop loss breach   (low < SL for longs, high > SL for shorts)
      2. Take profit breach (high > TP for longs, low < TP for shorts)
      3. Max bars in trade  (optional, default 50)
  - Outcome: "WIN", "LOSS", "TIMEOUT"

Session Filter
--------------
  If sessions are specified, setups are only taken during those session windows.
  Session windows are defined as UTC hour ranges (matching market_sessions.py).

Output
------
  BacktestResult dataclass containing:
    trades       List[Trade]
    stats        Stats
    config       dict   — the params used
    detector_summary dict — counts of each detector's output
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from swing_detector import detect_swings
from trend_detector import detect_trend
from bos_choch_detector import detect_bos_choch
from fvg_detector import detect_fvgs
from ob_detector import detect_obs
from confluence_scorer import score_bar, CONFLUENCE_REGISTRY


# ---------------------------------------------------------------------------
# Session windows (UTC hours)  — mirrors market_sessions.py Pine Script
# ---------------------------------------------------------------------------

SESSION_WINDOWS: Dict[str, Tuple[int, int]] = {
    "asian":        (19, 23),   # 19:00-23:59 UTC (previous day)
    "london":       (1,  5),    # 01:00-05:59 UTC
    "new_york":     (7,  11),   # 07:00-11:59 UTC
    "london_close": (15, 16),   # 15:00-16:59 UTC
    "overlap":      (12, 16),   # London/NY overlap
}


def _bar_in_session(ts_ms: int, sessions: List[str]) -> bool:
    """Return True if the bar's UTC hour falls within any selected session."""
    if not sessions:
        return True   # no filter → all bars valid
    import datetime
    dt   = datetime.datetime.utcfromtimestamp(ts_ms / 1000)
    hour = dt.hour
    for sess in sessions:
        window = SESSION_WINDOWS.get(sess)
        if window:
            start, end = window
            # Handle sessions that wrap midnight
            if start <= end:
                if start <= hour <= end:
                    return True
            else:
                if hour >= start or hour <= end:
                    return True
    return False


# ---------------------------------------------------------------------------
# ATR (reuse from fvg_detector pattern)
# ---------------------------------------------------------------------------

def _atr(highs: List[float], lows: List[float], closes: List[float],
         period: int, idx: int) -> float:
    if idx < 1:
        return highs[idx] - lows[idx]
    start = max(1, idx - period + 1)
    trs   = []
    for j in range(start, idx + 1):
        tr = max(
            highs[j] - lows[j],
            abs(highs[j] - closes[j - 1]),
            abs(lows[j]  - closes[j - 1]),
        )
        trs.append(tr)
    return sum(trs) / len(trs) if trs else highs[idx] - lows[idx]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    trade_id:      int
    direction:     str          # "BULLISH" (long) or "BEARISH" (short)
    entry_bar:     int
    entry_ts:      int          # unix ms
    entry_price:   float
    stop_loss:     float
    take_profit:   float
    risk_pips:     float        # distance from entry to SL in price units
    reward_pips:   float        # distance from entry to TP in price units
    rr_ratio:      float        # reward / risk
    exit_bar:      Optional[int]   = None
    exit_ts:       Optional[int]   = None
    exit_price:    Optional[float] = None
    outcome:       Optional[str]   = None  # "WIN", "LOSS", "TIMEOUT"
    bars_held:     Optional[int]   = None
    pnl_r:         Optional[float] = None  # PnL in R-multiples (+1 = 1R win)
    entry_type:    Optional[str]   = None  # "OB", "FVG", "OB+FVG"
    confluence_score: int          = 0
    confluence_max:   int          = 0
    confluence_items: Dict[str, bool] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        return self.exit_bar is None

    @property
    def won(self) -> bool:
        return self.outcome == "WIN"

    def __repr__(self):
        return (f"Trade(#{self.trade_id} {self.direction} "
                f"entry={self.entry_price:.5f} SL={self.stop_loss:.5f} "
                f"TP={self.take_profit:.5f} outcome={self.outcome} "
                f"pnl={self.pnl_r:.2f}R)" if self.pnl_r is not None else
                f"Trade(#{self.trade_id} {self.direction} OPEN)")


@dataclass
class Stats:
    total_trades:    int   = 0
    wins:            int   = 0
    losses:          int   = 0
    timeouts:        int   = 0
    win_rate:        float = 0.0
    avg_rr:          float = 0.0
    total_r:         float = 0.0
    avg_r_per_trade: float = 0.0
    max_drawdown_r:  float = 0.0
    profit_factor:   float = 0.0
    avg_bars_held:   float = 0.0
    longest_win_streak:  int = 0
    longest_loss_streak: int = 0

    # Breakdown by confluence item
    confluence_breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass
class BacktestResult:
    trades:           List[Trade]
    stats:            Stats
    config:           Dict
    detector_summary: Dict


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def _compute_stats(trades: List[Trade], selected: List[str]) -> Stats:
    closed = [t for t in trades if not t.is_open]
    if not closed:
        return Stats()

    wins     = [t for t in closed if t.outcome == "WIN"]
    losses   = [t for t in closed if t.outcome == "LOSS"]
    timeouts = [t for t in closed if t.outcome == "TIMEOUT"]

    total_r = sum(t.pnl_r for t in closed if t.pnl_r is not None)
    gross_profit = sum(t.pnl_r for t in wins    if t.pnl_r and t.pnl_r > 0)
    gross_loss   = abs(sum(t.pnl_r for t in losses if t.pnl_r and t.pnl_r < 0))

    # Max drawdown (peak-to-trough in R)
    equity = 0.0
    peak   = 0.0
    max_dd = 0.0
    for t in closed:
        if t.pnl_r:
            equity += t.pnl_r
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

    # Win/loss streaks
    longest_win  = 0
    longest_loss = 0
    cur_w = cur_l = 0
    for t in closed:
        if t.outcome == "WIN":
            cur_w += 1; cur_l = 0
        elif t.outcome == "LOSS":
            cur_l += 1; cur_w = 0
        longest_win  = max(longest_win,  cur_w)
        longest_loss = max(longest_loss, cur_l)

    # Confluence breakdown — how often each item was present in winning trades
    breakdown: Dict[str, int] = {item: 0 for item in selected}
    for t in wins:
        for item, passed in t.confluence_items.items():
            if passed and item in breakdown:
                breakdown[item] += 1

    bars_held = [t.bars_held for t in closed if t.bars_held is not None]

    return Stats(
        total_trades    = len(closed),
        wins            = len(wins),
        losses          = len(losses),
        timeouts        = len(timeouts),
        win_rate        = len(wins) / len(closed) if closed else 0.0,
        avg_rr          = sum(t.rr_ratio for t in closed) / len(closed),
        total_r         = round(total_r, 4),
        avg_r_per_trade = round(total_r / len(closed), 4) if closed else 0.0,
        max_drawdown_r  = round(max_dd, 4),
        profit_factor   = round(gross_profit / gross_loss, 4) if gross_loss > 0 else float("inf"),
        avg_bars_held   = round(sum(bars_held) / len(bars_held), 1) if bars_held else 0.0,
        longest_win_streak  = longest_win,
        longest_loss_streak = longest_loss,
        confluence_breakdown = breakdown,
    )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_backtest(
    opens:          Sequence[float],
    highs:          Sequence[float],
    lows:           Sequence[float],
    closes:         Sequence[float],
    timestamps:     Sequence[int],

    # User configuration
    selected_confluences: List[str]   = None,
    min_required:         int         = 2,
    directions:           List[str]   = None,  # ["BULLISH","BEARISH"] or one
    sessions:             List[str]   = None,  # session filter
    rr_ratio:             float       = 2.0,
    sl_atr_mult:          float       = 0.5,   # SL buffer beyond zone edge
    atr_period:           int         = 14,
    max_bars_in_trade:    int         = 50,
    zone_tolerance:       float       = 0.001,
    lookback:             int         = 20,

    # Swing detector params
    pivot_length:         int         = 5,
    rolling_length:       int         = 10,
) -> BacktestResult:
    """
    Run a full backtest over the provided OHLCV data.

    Parameters
    ----------
    opens/highs/lows/closes : OHLCV arrays, oldest first
    timestamps              : unix ms per bar
    selected_confluences    : list of confluence IDs to use
                              e.g. ["trend","bos","fvg","ob"]
    min_required            : minimum confluence score to trigger a trade
    directions              : which trade directions to look for
    sessions                : session names to filter entries
    rr_ratio                : reward-to-risk ratio for TP placement
    sl_atr_mult             : ATR multiplier for SL buffer beyond zone
    atr_period              : ATR period
    max_bars_in_trade       : force-close trades after this many bars
    zone_tolerance          : proximity tolerance for zone checks
    lookback                : bars to look back for BOS/CHoCH events

    Returns
    -------
    BacktestResult
    """
    # Defaults
    if selected_confluences is None:
        selected_confluences = ["trend", "bos", "fvg", "ob"]
    if directions is None:
        directions = ["BULLISH", "BEARISH"]
    if sessions is None:
        sessions = []

    opens      = list(opens)
    highs      = list(highs)
    lows       = list(lows)
    closes     = list(closes)
    timestamps = list(timestamps)
    bars       = len(closes)

    # ── Step 1: Run all detectors ────────────────────────────────────────
    swings_piv  = detect_swings(highs, lows, timestamps,
                                method="pivot",   length=pivot_length)
    swings_roll = detect_swings(highs, lows, timestamps,
                                method="rolling", length=rolling_length)

    trend_states = detect_trend(swings_piv, timestamps, method="pivot", length=pivot_length)
    bos_events   = detect_bos_choch(highs, lows, closes, timestamps,
                                     swings_piv, trend_states,
                                     method="pivot", length=pivot_length)
    fvgs         = detect_fvgs(highs, lows, closes, timestamps)
    obs          = detect_obs(opens, highs, lows, closes, timestamps,
                               swings_roll, method="rolling", length=rolling_length)

    # ── Step 2: Scan bars for entries ────────────────────────────────────
    trades:       List[Trade] = []
    open_trades:  List[Trade] = []
    trade_id      = 0
    warmup        = max(pivot_length * 2, rolling_length * 2, atr_period, 30)

    for bar_i in range(warmup, bars):
        ts    = timestamps[bar_i]
        h     = highs[bar_i]
        l     = lows[bar_i]
        c     = closes[bar_i]
        atr   = _atr(highs, lows, closes, atr_period, bar_i)

        # ── Manage open trades ───────────────────────────────────────────
        still_open: List[Trade] = []
        for trade in open_trades:
            # Check exit conditions using this bar's OHLC
            exited   = False
            exit_pnl = None

            if trade.direction == "BULLISH":
                if l <= trade.stop_loss:
                    trade.outcome     = "LOSS"
                    trade.exit_price  = trade.stop_loss
                    exit_pnl          = -1.0
                    exited            = True
                elif h >= trade.take_profit:
                    trade.outcome     = "WIN"
                    trade.exit_price  = trade.take_profit
                    exit_pnl          = trade.rr_ratio
                    exited            = True
            else:  # BEARISH (short)
                if h >= trade.stop_loss:
                    trade.outcome     = "LOSS"
                    trade.exit_price  = trade.stop_loss
                    exit_pnl          = -1.0
                    exited            = True
                elif l <= trade.take_profit:
                    trade.outcome     = "WIN"
                    trade.exit_price  = trade.take_profit
                    exit_pnl          = trade.rr_ratio
                    exited            = True

            # Max bars timeout
            if not exited and (bar_i - trade.entry_bar) >= max_bars_in_trade:
                trade.outcome    = "TIMEOUT"
                trade.exit_price = c
                exit_pnl         = (c - trade.entry_price) / (trade.entry_price - trade.stop_loss) \
                                   if trade.direction == "BULLISH" else \
                                   (trade.entry_price - c) / (trade.stop_loss - trade.entry_price)
                exited           = True

            if exited:
                trade.exit_bar  = bar_i
                trade.exit_ts   = ts
                trade.bars_held = bar_i - trade.entry_bar
                trade.pnl_r     = round(exit_pnl, 4) if exit_pnl is not None else 0.0
            else:
                still_open.append(trade)

        open_trades = still_open

        # ── Session filter ───────────────────────────────────────────────
        if not _bar_in_session(ts, sessions):
            continue

        # ── Score each direction ─────────────────────────────────────────
        for direction in directions:
            # Skip if already in a trade in this direction
            already_open = any(t.direction == direction for t in open_trades)
            if already_open:
                continue

            result = score_bar(
                bar_index      = bar_i,
                price          = c,
                direction      = direction,
                selected       = selected_confluences,
                min_required   = min_required,
                trend_states   = trend_states,
                bos_events     = bos_events,
                fvgs           = fvgs,
                obs            = obs,
                lookback       = lookback,
                zone_tolerance = zone_tolerance,
            )

            if not result.passed or result.entry_zone is None:
                continue

            # ── Build the trade ──────────────────────────────────────────
            zone_top, zone_bot = result.entry_zone
            entry_price        = c

            if direction == "BULLISH":
                sl        = zone_bot - (atr * sl_atr_mult)
                risk      = entry_price - sl
                if risk <= 0:
                    continue
                tp        = entry_price + (risk * rr_ratio)
            else:  # BEARISH
                sl        = zone_top + (atr * sl_atr_mult)
                risk      = sl - entry_price
                if risk <= 0:
                    continue
                tp        = entry_price - (risk * rr_ratio)

            trade_id += 1
            trade = Trade(
                trade_id          = trade_id,
                direction         = direction,
                entry_bar         = bar_i,
                entry_ts          = ts,
                entry_price       = entry_price,
                stop_loss         = sl,
                take_profit       = tp,
                risk_pips         = round(risk, 6),
                reward_pips       = round(risk * rr_ratio, 6),
                rr_ratio          = rr_ratio,
                entry_type        = result.entry_type,
                confluence_score  = result.score,
                confluence_max    = result.max_score,
                confluence_items  = result.items.copy(),
            )
            open_trades.append(trade)
            trades.append(trade)

    # Close any still-open trades at last bar
    for trade in open_trades:
        trade.outcome    = "TIMEOUT"
        trade.exit_bar   = bars - 1
        trade.exit_ts    = timestamps[-1]
        trade.exit_price = closes[-1]
        trade.bars_held  = trade.exit_bar - trade.entry_bar
        if trade.direction == "BULLISH":
            pnl = (closes[-1] - trade.entry_price) / max(trade.risk_pips, 1e-10)
        else:
            pnl = (trade.entry_price - closes[-1]) / max(trade.risk_pips, 1e-10)
        trade.pnl_r = round(pnl, 4)

    # ── Step 3: Compute stats ────────────────────────────────────────────
    stats = _compute_stats(trades, selected_confluences)

    detector_summary = {
        "swing_points_pivot":   len(swings_piv),
        "swing_points_rolling": len(swings_roll),
        "bos_events":           len([e for e in bos_events if e.event_type == "BOS"]),
        "choch_events":         len([e for e in bos_events if e.event_type == "CHoCH"]),
        "fvgs_total":           len(fvgs),
        "fvgs_mitigated":       len([f for f in fvgs if f.status == "MITIGATED"]),
        "obs_total":            len(obs),
        "obs_mitigated":        len([ob for ob in obs if ob.status == "MITIGATED"]),
    }

    config = {
        "selected_confluences": selected_confluences,
        "min_required":         min_required,
        "directions":           directions,
        "sessions":             sessions,
        "rr_ratio":             rr_ratio,
        "sl_atr_mult":          sl_atr_mult,
        "atr_period":           atr_period,
        "max_bars_in_trade":    max_bars_in_trade,
        "zone_tolerance":       zone_tolerance,
        "lookback":             lookback,
        "pivot_length":         pivot_length,
        "rolling_length":       rolling_length,
        "total_bars":           bars,
    }

    return BacktestResult(trades=trades, stats=stats,
                          config=config, detector_summary=detector_summary)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math, sys
    sys.path.insert(0, ".")

    print("=" * 60)
    print("backtest_runner.py — self test")
    print("=" * 60)

    # Build a 500-bar synthetic dataset: uptrend then downtrend
    N = 500
    ts_base = 1_700_000_000_000

    def mp(i):
        if i < 250:
            return 1.2000 + i * 0.0003 + 0.005 * math.sin(2 * math.pi * i / 20)
        return 1.2750 - (i - 250) * 0.0002 + 0.005 * math.sin(2 * math.pi * i / 20)

    mid    = [mp(i)       for i in range(N)]
    opens  = [m - 0.0002  for m in mid]
    highs  = [m + 0.0010  for m in mid]
    lows   = [m - 0.0010  for m in mid]
    closes = mid
    ts_arr = [ts_base + i * 3_600_000 for i in range(N)]  # 1-hour bars

    print("\nRunning backtest with: trend + bos + fvg + ob, min=2, RR=2.0")
    result = run_backtest(
        opens      = opens,
        highs      = highs,
        lows       = lows,
        closes     = closes,
        timestamps = ts_arr,
        selected_confluences = ["trend", "bos", "fvg", "ob"],
        min_required         = 2,
        directions           = ["BULLISH", "BEARISH"],
        rr_ratio             = 2.0,
        sl_atr_mult          = 0.5,
        zone_tolerance       = 0.05,   # wide for synthetic data
        lookback             = 30,
    )

    s = result.stats
    print(f"\n{'='*40}")
    print(f"  RESULTS")
    print(f"{'='*40}")
    print(f"  Total trades:     {s.total_trades}")
    print(f"  Wins:             {s.wins}")
    print(f"  Losses:           {s.losses}")
    print(f"  Timeouts:         {s.timeouts}")
    print(f"  Win rate:         {s.win_rate*100:.1f}%")
    print(f"  Total R:          {s.total_r:+.2f}R")
    print(f"  Avg R/trade:      {s.avg_r_per_trade:+.3f}R")
    print(f"  Max drawdown:     {s.max_drawdown_r:.2f}R")
    print(f"  Profit factor:    {s.profit_factor:.2f}")
    print(f"  Avg bars held:    {s.avg_bars_held}")
    print(f"  Longest win str:  {s.longest_win_streak}")
    print(f"  Longest loss str: {s.longest_loss_streak}")
    print(f"\nDetector summary:")
    for k, v in result.detector_summary.items():
        print(f"  {k:<25} {v}")

    if result.trades:
        print(f"\nFirst 5 trades:")
        for t in result.trades[:5]:
            print(f"  {t}")

    assert isinstance(result.trades, list)
    assert isinstance(result.stats, Stats)
    assert result.stats.total_trades == len([t for t in result.trades if not t.is_open or t.outcome])
    print("\n✅ All tests passed")

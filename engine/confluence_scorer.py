"""
confluence_scorer.py
────────────────────
Aggregates all SMC detector signals into a per-candle confluence score.

Each selected confluence is evaluated at the timestamp of a potential entry.
The scorer returns a list of ConfluentSetup objects — each representing a
moment in time where the requested confluence conditions aligned.

Confluence IDs supported:
  Structure:  trend | bos | choch | ftd
  Blocks:     ob | bb | pb | rb
  Liquidity:  sweep | engineered
  FVGs:       fvg | ifvg
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional
import pandas as pd
import numpy as np

# Detector imports
from trend_detector      import detect_trend,           TrendTF
from structure_detector  import detect_bos, detect_choch, detect_ftd
from order_block_detector import detect_order_blocks
from blocks_detector     import (detect_breaker_blocks,
                                  detect_propulsion_blocks,
                                  detect_rejection_blocks)
from liquidity_detector  import (detect_liquidity_sweeps,
                                  detect_engineered_liquidity)
from fvg_detector        import detect_fvg, detect_ifvg


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class ConfluentSetup:
    timestamp: pd.Timestamp
    direction: Literal["bullish", "bearish"]
    confluences_hit: List[str]          # list of confluence IDs that fired
    confluence_count: int
    confluence_score: float             # 0–1 weighted score
    entry_price: float
    details: Dict[str, Any] = field(default_factory=dict)


# ── weighting (each confluence contributes to score) ─────────────────────────

WEIGHTS: Dict[str, float] = {
    "trend":      0.15,
    "bos":        0.12,
    "choch":      0.12,
    "ftd":        0.08,
    "ob":         0.14,
    "bb":         0.10,
    "pb":         0.08,
    "rb":         0.05,
    "sweep":      0.06,
    "engineered": 0.05,
    "fvg":        0.10,
    "ifvg":       0.08,
}
MAX_POSSIBLE = sum(WEIGHTS.values())


# ── helpers ───────────────────────────────────────────────────────────────────

def _signals_in_window(
    signal_list,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    direction_attr: str = "direction",
    direction_val: Optional[str] = None,
) -> list:
    out = []
    for s in signal_list:
        ts = getattr(s, "timestamp", None)
        if ts is None:
            continue
        if window_start <= ts <= window_end:
            if direction_val is None:
                out.append(s)
            else:
                dv = getattr(s, direction_attr, None)
                if dv is None or dv == direction_val:
                    out.append(s)
    return out


def _build_all_signals(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, list]:
    """
    Run every enabled detector and return a dict of signal lists.
    params keys match confluence IDs.
    """
    signals: Dict[str, list] = {}

    if "trend" in params:
        p = params["trend"]
        signals["trend"] = detect_trend(
            df,
            timeframe     = p.get("Timeframe", "4H"),
            bias_filter   = p.get("Bias", "Both"),
        )

    if "bos" in params:
        p = params["bos"]
        signals["bos"] = detect_bos(
            df,
            timeframe  = p.get("TF", "1H"),
            direction  = p.get("Direction", "Both"),
        )

    if "choch" in params:
        p = params["choch"]
        signals["choch"] = detect_choch(
            df,
            timeframe  = p.get("TF", "15m"),
            direction  = p.get("Direction", "Both"),
        )

    if "ftd" in params:
        p = params["ftd"]
        signals["ftd"] = detect_ftd(
            df,
            timeframe   = p.get("TF", "15m"),
            level_type  = p.get("Level", "High"),
            direction   = p.get("Direction", "Both"),
        )

    if "ob" in params:
        p = params["ob"]
        signals["ob"] = detect_order_blocks(
            df,
            timeframe      = p.get("TF", "1H"),
            ob_type        = p.get("Type", "Both"),
            mitigation     = p.get("Mitigation", "Wick"),
            volume_filter  = p.get("Volume", "Any"),
        )

    if "bb" in params:
        p = params["bb"]
        signals["bb"] = detect_breaker_blocks(
            df,
            timeframe   = p.get("TF", "1H"),
            bb_type     = p.get("Type", "Both"),
            mitigation  = p.get("Mitigation", "Wick"),
        )

    if "pb" in params:
        p = params["pb"]
        signals["pb"] = detect_propulsion_blocks(
            df,
            timeframe = p.get("TF", "1H"),
            pb_type   = p.get("Direction", "Both"),
        )

    if "rb" in params:
        p = params["rb"]
        signals["rb"] = detect_rejection_blocks(
            df,
            timeframe  = p.get("TF", "15m"),
            min_wicks  = p.get("Min wicks", "3"),
        )

    if "sweep" in params:
        p = params["sweep"]
        signals["sweep"] = detect_liquidity_sweeps(
            df,
            timeframe   = p.get("TF", "1H"),
            sweep_type  = p.get("Target", "Both"),
        )

    if "engineered" in params:
        p = params["engineered"]
        signals["engineered"] = detect_engineered_liquidity(
            df,
            timeframe = p.get("TF", "1H"),
            eng_type  = p.get("Type", "Inducement"),
        )

    if "fvg" in params:
        p = params["fvg"]
        signals["fvg"] = detect_fvg(
            df,
            timeframe      = p.get("TF", "15m"),
            fvg_type       = p.get("Type", "Both"),
            status_filter  = p.get("Status", "Any"),
            ce_filter      = p.get("CE (50%)", "Any"),
        )

    if "ifvg" in params:
        p = params["ifvg"]
        signals["ifvg"] = detect_ifvg(
            df,
            timeframe      = p.get("TF", "15m"),
            ifvg_type      = p.get("Type", "Both"),
            status_filter  = p.get("Status", "Any"),
            ce_filter      = p.get("CE (50%)", "Any"),
        )

    return signals


# ── DIRECTION RESOLVER ────────────────────────────────────────────────────────

def _get_direction(signal_id: str, sig) -> Optional[str]:
    """Extract normalised direction string from any signal type."""
    for attr in ("direction", "bias", "ob_type", "bb_type", "pb_type",
                 "rb_type", "sweep_type", "eng_type", "fvg_type", "ifvg_type"):
        val = getattr(sig, attr, None)
        if val is None:
            continue
        if "bull" in str(val).lower() or "buy" in str(val).lower():
            return "bullish"
        if "bear" in str(val).lower() or "sell" in str(val).lower():
            return "bearish"
    return None


# ── MAIN SCORER ───────────────────────────────────────────────────────────────

def score_confluences(
    df: pd.DataFrame,
    params: Dict[str, Dict[str, Any]],
    min_confluences: int = 2,
    lookback_bars: int = 5,      # candles to look back for signal alignment
    entry_timeframe: str = "15m",
) -> List[ConfluentSetup]:
    """
    Build and score all confluence setups.

    params  — dict of {confluence_id: {param_key: value}}
               Only IDs present in dict are evaluated.

    Returns list of ConfluentSetup sorted by timestamp.
    """
    if not params:
        return []

    # Run all detectors
    all_signals = _build_all_signals(df, params)

    # Determine entry candles from entry_timeframe
    rule = {"5m": "5min", "15m": "15min", "1H": "1h", "4H": "4h"}.get(
        entry_timeframe, "15min"
    )
    entry_df = df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna()

    setups: List[ConfluentSetup] = []

    for i, (ts, candle) in enumerate(entry_df.iterrows()):
        if i < lookback_bars:
            continue

        window_start = entry_df.index[i - lookback_bars]
        window_end   = ts

        # Try both directions
        for direction in ("bullish", "bearish"):
            hits:    List[str] = []
            details: Dict[str, Any] = {}

            for cid, sig_list in all_signals.items():
                matched = _signals_in_window(
                    sig_list, window_start, window_end
                )
                dir_matched = [s for s in matched
                               if _get_direction(cid, s) in (direction, None)]
                if dir_matched:
                    hits.append(cid)
                    details[cid] = len(dir_matched)

            if len(hits) < min_confluences:
                continue

            raw_score = sum(WEIGHTS.get(cid, 0.05) for cid in hits)
            norm_score = round(min(raw_score / MAX_POSSIBLE, 1.0), 3)

            setups.append(ConfluentSetup(
                timestamp        = ts,
                direction        = direction,
                confluences_hit  = hits,
                confluence_count = len(hits),
                confluence_score = norm_score,
                entry_price      = candle["close"],
                details          = details,
            ))

    # Deduplicate: keep highest-score setup per timestamp
    dedup: Dict[pd.Timestamp, ConfluentSetup] = {}
    for s in setups:
        key = (s.timestamp, s.direction)
        if key not in dedup or s.confluence_score > dedup[key].confluence_score:
            dedup[key] = s

    return sorted(dedup.values(), key=lambda s: s.timestamp)


# ── convenience: summary stats ────────────────────────────────────────────────

def summarise(setups: List[ConfluentSetup]) -> Dict[str, Any]:
    if not setups:
        return {"total": 0}
    scores = [s.confluence_score for s in setups]
    dirs   = [s.direction for s in setups]
    return {
        "total"         : len(setups),
        "bullish"       : dirs.count("bullish"),
        "bearish"       : dirs.count("bearish"),
        "avg_score"     : round(float(np.mean(scores)), 3),
        "max_score"     : round(float(np.max(scores)), 3),
        "min_score"     : round(float(np.min(scores)), 3),
        "high_conf"     : sum(1 for s in scores if s >= 0.5),
        "confluence_freq": {
            cid: sum(1 for s in setups if cid in s.confluences_hit)
            for cid in WEIGHTS
        },
    }


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(77)
    n   = 5000
    idx = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    close = 1.0800
    rows  = []
    for i in range(n):
        close += random.gauss(0.0001, 0.0009)
        o = close - abs(random.gauss(0, 0.0003))
        h = max(o, close) + abs(random.gauss(0, 0.0005))
        l = min(o, close) - abs(random.gauss(0, 0.0005))
        rows.append({"open": o, "high": h, "low": l,
                     "close": close, "volume": random.randint(100, 2000)})
    df = pd.DataFrame(rows, index=idx)

    params = {
        "trend":  {"Bias": "Both",     "Timeframe": "4H"},
        "bos":    {"Direction": "Both","TF": "1H"},
        "fvg":    {"Type": "Both",     "TF": "15m",  "Status": "Any", "CE (50%)": "Any"},
        "ob":     {"Type": "Both",     "TF": "15m",  "Mitigation": "Wick", "Volume": "Any"},
    }

    setups = score_confluences(df, params, min_confluences=2)
    stats  = summarise(setups)
    print(f"Setups found: {stats['total']}")
    print(f"  Bullish   : {stats['bullish']}")
    print(f"  Bearish   : {stats['bearish']}")
    print(f"  Avg score : {stats['avg_score']}")
    print(f"  High conf : {stats['high_conf']}")

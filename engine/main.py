"""
main.py
=======
Wicks SMC Backtesting Engine — FastAPI Application

Exposes the backtest engine over HTTP for the React Native frontend.

Endpoints
---------
  GET  /health               — liveness check
  GET  /symbols              — list of supported instruments
  GET  /confluences          — list of supported confluence items
  POST /backtest             — run a full backtest, returns trades + stats
  GET  /backtest/{job_id}    — fetch a cached result (future: async jobs)

CORS
----
  Configured to allow all origins in development.
  Set ALLOWED_ORIGINS env var to restrict in production.

Environment Variables
---------------------
  POLYGON_API_KEY  — required for live data fetching
  ALLOWED_ORIGINS  — comma-separated list of allowed CORS origins
                     defaults to "*" (all origins, for Expo dev)
"""

import os
import time
import hashlib
import json
import math
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Local engine imports
import sys
sys.path.insert(0, os.path.dirname(__file__))

from data_fetcher import fetch_ohlcv
from backtest_runner import run_backtest, BacktestResult
from confluence_scorer import CONFLUENCE_REGISTRY


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Wicks SMC Backtesting Engine",
    description = "ICT/SMC backtesting API for the Wicks mobile app",
    version     = "1.0.0",
)

# CORS — allow Expo Go and production app
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = (
    ["*"] if allowed_origins_env == "*"
    else [o.strip() for o in allowed_origins_env.split(",")]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = allowed_origins,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# In-memory result cache (keyed by job_id = hash of request)
_result_cache: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Supported instruments
# ---------------------------------------------------------------------------

SYMBOLS = {
    "forex": [
        "C:EURUSD", "C:GBPUSD", "C:USDJPY", "C:AUDUSD",
        "C:USDCAD", "C:NZDUSD", "C:USDCHF", "C:GBPJPY",
        "C:EURJPY", "C:EURGBP",
    ],
    "futures": [
        "I:SPX",    # S&P 500
        "I:NDX",    # Nasdaq 100
        "I:DJI",    # Dow Jones
        "C:XAUUSD", # Gold
        "C:XAGUSD", # Silver
        "C:USOIL",  # Crude Oil
    ],
    "crypto": [
        "X:BTCUSD", "X:ETHUSD", "X:SOLUSD",
        "X:XRPUSD", "X:ADAUSD",
    ],
}

TIMEFRAMES = {
    "1m":  1,
    "5m":  5,
    "15m": 15,
    "1h":  60,
    "4h":  240,
    "1d":  1440,
}

WINDOWS = {
    "1m":  1,
    "3m":  3,
    "6m":  6,
    "1y":  12,
    "2y":  24,
    "5y":  60,
}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class BacktestRequest(BaseModel):
    symbol:     str = Field(..., example="C:EURUSD")
    timeframe:  str = Field("1h",  example="1h")
    window:     str = Field("6m",  example="6m",
                            description="Data window: 1m, 3m, 6m, 1y, 2y, 5y")

    # Confluence settings
    confluences:  List[str] = Field(
        default=["trend", "bos", "fvg", "ob"],
        description="Active confluence IDs"
    )
    min_required: int   = Field(2, ge=1, le=10)
    directions:   List[str] = Field(default=["BULLISH", "BEARISH"])
    sessions:     List[str] = Field(default=[])

    # Risk settings
    rr_ratio:    float = Field(2.0,  ge=0.5, le=20.0)
    sl_atr_mult: float = Field(0.5,  ge=0.1, le=5.0)
    risk_pct:    float = Field(1.0,  ge=0.1, le=10.0,
                               description="% of account risked per trade")

    # Advanced
    zone_tolerance: float = Field(0.001, ge=0.0001, le=0.05)
    lookback:       int   = Field(20,    ge=5,       le=100)
    max_bars:       int   = Field(50,    ge=5,       le=500)

    @validator("timeframe")
    def valid_timeframe(cls, v):
        if v not in TIMEFRAMES:
            raise ValueError(f"timeframe must be one of {list(TIMEFRAMES.keys())}")
        return v

    @validator("confluences", each_item=True)
    def valid_confluence(cls, v):
        if v not in CONFLUENCE_REGISTRY:
            raise ValueError(f"Unknown confluence: {v}. Valid: {list(CONFLUENCE_REGISTRY.keys())}")
        return v

    @validator("directions", each_item=True)
    def valid_direction(cls, v):
        if v not in ("BULLISH", "BEARISH"):
            raise ValueError("directions must be 'BULLISH' or 'BEARISH'")
        return v


class TradeOut(BaseModel):
    trade_id:      int
    direction:     str
    entry_bar:     int
    entry_ts:      int
    entry_price:   float
    stop_loss:     float
    take_profit:   float
    risk_pips:     float
    reward_pips:   float
    rr_ratio:      float
    exit_bar:      Optional[int]
    exit_ts:       Optional[int]
    exit_price:    Optional[float]
    outcome:       Optional[str]
    bars_held:     Optional[int]
    pnl_r:         Optional[float]
    entry_type:    Optional[str]
    confluence_score: int
    confluence_max:   int
    confluence_items: Dict[str, bool]


class StatsOut(BaseModel):
    total_trades:    int
    wins:            int
    losses:          int
    timeouts:        int
    win_rate:        float
    avg_rr:          float
    total_r:         float
    avg_r_per_trade: float
    max_drawdown_r:  float
    profit_factor:   float
    avg_bars_held:   float
    longest_win_streak:  int
    longest_loss_streak: int
    confluence_breakdown: Dict[str, int]


class BacktestResponse(BaseModel):
    job_id:           str
    symbol:           str
    timeframe:        str
    window:           str
    trades:           List[TradeOut]
    stats:            StatsOut
    detector_summary: Dict[str, int]
    config:           Dict
    generated_at:     int   # unix ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_job_id(req: BacktestRequest) -> str:
    payload = req.json(sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def _result_to_response(
    result:    BacktestResult,
    job_id:    str,
    req:       BacktestRequest,
) -> BacktestResponse:
    trades_out = []
    for t in result.trades:
        trades_out.append(TradeOut(
            trade_id         = t.trade_id,
            direction        = t.direction,
            entry_bar        = t.entry_bar,
            entry_ts         = t.entry_ts,
            entry_price      = t.entry_price,
            stop_loss        = t.stop_loss,
            take_profit      = t.take_profit,
            risk_pips        = t.risk_pips,
            reward_pips      = t.reward_pips,
            rr_ratio         = t.rr_ratio,
            exit_bar         = t.exit_bar,
            exit_ts          = t.exit_ts,
            exit_price       = t.exit_price,
            outcome          = t.outcome,
            bars_held        = t.bars_held,
            pnl_r            = t.pnl_r,
            entry_type       = t.entry_type,
            confluence_score = t.confluence_score,
            confluence_max   = t.confluence_max,
            confluence_items = t.confluence_items,
        ))

    s = result.stats
    stats_out = StatsOut(
        total_trades         = s.total_trades,
        wins                 = s.wins,
        losses               = s.losses,
        timeouts             = s.timeouts,
        win_rate             = round(s.win_rate, 4),
        avg_rr               = round(s.avg_rr, 4),
        total_r              = s.total_r,
        avg_r_per_trade      = s.avg_r_per_trade,
        max_drawdown_r       = s.max_drawdown_r,
        profit_factor        = s.profit_factor if not math.isinf(s.profit_factor) else 999.0,
        avg_bars_held        = s.avg_bars_held,
        longest_win_streak   = s.longest_win_streak,
        longest_loss_streak  = s.longest_loss_streak,
        confluence_breakdown = s.confluence_breakdown,
    )

    return BacktestResponse(
        job_id           = job_id,
        symbol           = req.symbol,
        timeframe        = req.timeframe,
        window           = req.window,
        trades           = trades_out,
        stats            = stats_out,
        detector_summary = result.detector_summary,
        config           = result.config,
        generated_at     = int(time.time() * 1000),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check."""
    return {
        "status":  "ok",
        "version": "1.0.0",
        "engine":  "wicks-smc",
    }


@app.get("/symbols")
def list_symbols():
    """Return supported instruments grouped by category."""
    return SYMBOLS


@app.get("/confluences")
def list_confluences():
    """Return all supported confluence items with labels."""
    return {
        k: {"label": v, "id": k}
        for k, v in CONFLUENCE_REGISTRY.items()
    }


@app.get("/timeframes")
def list_timeframes():
    return list(TIMEFRAMES.keys())


@app.post("/backtest", response_model=BacktestResponse)
def run_backtest_endpoint(req: BacktestRequest):
    """
    Run a full SMC backtest.

    Fetches OHLCV data from Polygon.io, runs all detectors,
    evaluates confluence conditions at every bar, and returns
    a complete trade log with statistics.
    """
    job_id = _make_job_id(req)

    # Return cached result if available
    if job_id in _result_cache:
        return _result_cache[job_id]

    # ── Fetch OHLCV data ─────────────────────────────────────────────────
    try:
        ohlcv = fetch_ohlcv(
            symbol    = req.symbol,
            timeframe = req.timeframe,
            months    = WINDOWS.get(req.window, 6),
        )
    except Exception as e:
        raise HTTPException(status_code=502,
                            detail=f"Data fetch failed: {str(e)}")

    if len(ohlcv["closes"]) < 100:
        raise HTTPException(status_code=422,
                            detail=f"Insufficient data: only {len(ohlcv['closes'])} bars returned. "
                                   "Try a longer window or different symbol/timeframe.")

    # ── Run backtest engine ──────────────────────────────────────────────
    try:
        result = run_backtest(
            opens      = ohlcv["opens"],
            highs      = ohlcv["highs"],
            lows       = ohlcv["lows"],
            closes     = ohlcv["closes"],
            timestamps = ohlcv["timestamps"],

            selected_confluences = req.confluences,
            min_required         = req.min_required,
            directions           = req.directions,
            sessions             = req.sessions,
            rr_ratio             = req.rr_ratio,
            sl_atr_mult          = req.sl_atr_mult,
            zone_tolerance       = req.zone_tolerance,
            lookback             = req.lookback,
            max_bars_in_trade    = req.max_bars,
        )
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Backtest engine error: {str(e)}")

    response = _result_to_response(result, job_id, req)

    # Cache result
    _result_cache[job_id] = response

    # Evict oldest if cache gets too large
    if len(_result_cache) > 100:
        oldest_key = next(iter(_result_cache))
        del _result_cache[oldest_key]

    return response


@app.get("/backtest/{job_id}", response_model=BacktestResponse)
def get_cached_result(job_id: str):
    """Retrieve a previously computed backtest result by job_id."""
    if job_id not in _result_cache:
        raise HTTPException(status_code=404,
                            detail=f"No cached result for job_id={job_id}")
    return _result_cache[job_id]

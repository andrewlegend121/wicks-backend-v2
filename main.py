"""
main.py
───────
FastAPI entry point for the Wicks backtesting engine.

Routes:
  GET  /health          → liveness probe
  POST /backtest        → run full backtest
  POST /scan            → confluence scan only (no trade simulation)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from backtest_runner import run_backtest
from confluence_scorer import score_confluences, summarise
from backtest_runner import _fetch_candles


app = FastAPI(title="Wicks Backtest Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class RiskConfig(BaseModel):
    rr:          float = 2.0
    risk_pct:    float = 1.0
    sl_type:     str   = "fixed_pips"
    sl_pips:     float = 15.0
    max_trades:  Optional[int] = None
    session:     Optional[str] = None


class BacktestRequest(BaseModel):
    symbol:    str = "C:EURUSD"
    window:    str = "1mo"
    params:    Dict[str, Dict[str, Any]] = {}
    min_conf:  int = 2
    risk:      RiskConfig = RiskConfig()


class ScanRequest(BaseModel):
    symbol:    str = "C:EURUSD"
    window:    str = "1mo"
    params:    Dict[str, Dict[str, Any]] = {}
    min_conf:  int = 2


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "engine": "wicks-v2"}


@app.post("/backtest")
async def backtest(req: BacktestRequest):
    try:
        body = req.dict()
        body["risk"] = req.risk.dict()
        result = await run_backtest(body)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scan")
async def scan(req: ScanRequest):
    """
    Run the confluence scanner only — no trade simulation.
    Returns all confluence setups with scores.
    """
    try:
        df     = _fetch_candles(req.symbol, req.window)
        setups = score_confluences(df, req.params, req.min_conf)
        stats  = summarise(setups)
        return {
            "symbol":  req.symbol,
            "window":  req.window,
            "setups": [
                {
                    "timestamp":        str(s.timestamp),
                    "direction":        s.direction,
                    "confluences_hit":  s.confluences_hit,
                    "confluence_count": s.confluence_count,
                    "confluence_score": s.confluence_score,
                    "entry_price":      s.entry_price,
                }
                for s in setups
            ],
            "stats": stats,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

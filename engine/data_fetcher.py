"""
data_fetcher.py
===============
Wicks SMC Backtesting Engine — Layer 0

Fetches OHLCV data from Polygon.io and returns it in the format
expected by all detectors (oldest-first lists).

Environment Variables
---------------------
  POLYGON_API_KEY : required

Supported asset classes
-----------------------
  Forex   — prefix "C:"  e.g. "C:EURUSD"
  Crypto  — prefix "X:"  e.g. "X:BTCUSD"
  Stocks  — no prefix    e.g. "AAPL"
  Indices — prefix "I:"  e.g. "I:SPX"
"""

from __future__ import annotations

import os
import time
import datetime
from typing import Dict, List

import requests


POLYGON_BASE = "https://api.polygon.io/v2/aggs/ticker"

TIMEFRAME_MAP = {
    "1m":  ("minute", 1),
    "5m":  ("minute", 5),
    "15m": ("minute", 15),
    "1h":  ("hour",   1),
    "4h":  ("hour",   4),
    "1d":  ("day",    1),
}


def fetch_ohlcv(
    symbol:    str,
    timeframe: str = "1h",
    months:    int = 6,
    api_key:   str = None,
) -> Dict[str, List]:
    """
    Fetch OHLCV bars from Polygon.io.

    Parameters
    ----------
    symbol    : Polygon ticker e.g. "C:EURUSD", "X:BTCUSD", "I:SPX"
    timeframe : one of "1m", "5m", "15m", "1h", "4h", "1d"
    months    : how many months of history to fetch
    api_key   : Polygon API key (falls back to POLYGON_API_KEY env var)

    Returns
    -------
    dict with keys: timestamps, opens, highs, lows, closes, volumes
    All lists are oldest-first.

    Raises
    ------
    ValueError  : bad timeframe or missing API key
    RuntimeError: API returned an error
    """
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(f"Unsupported timeframe {timeframe!r}. "
                         f"Use one of: {list(TIMEFRAME_MAP.keys())}")

    key = api_key or os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY")
    if not key:
        raise ValueError("POLYGON_API_KEY not set. "
                         "Set the environment variable or pass api_key=.")

    span, multiplier = TIMEFRAME_MAP[timeframe]

    end_dt   = datetime.date.today()
    # Add buffer for weekends/holidays
    start_dt = end_dt - datetime.timedelta(days=int(months * 31))

    from_str = start_dt.strftime("%Y-%m-%d")
    to_str   = end_dt.strftime("%Y-%m-%d")

    url = (
        f"{POLYGON_BASE}/{symbol}/range/{multiplier}/{span}"
        f"/{from_str}/{to_str}"
    )

    timestamps: List[int]   = []
    opens:      List[float] = []
    highs:      List[float] = []
    lows:       List[float] = []
    closes:     List[float] = []
    volumes:    List[float] = []

    # Polygon paginates — follow next_url
    next_url: str = url
    params = {
        "adjusted": "true",
        "sort":     "asc",
        "limit":    50000,
        "apiKey":   key,
    }

    while next_url:
        resp = requests.get(next_url, params=params if next_url == url else None,
                            timeout=30)

        if resp.status_code == 429:
            # Rate limited — wait and retry
            time.sleep(12)
            resp = requests.get(next_url,
                                params=params if next_url == url else None,
                                timeout=30)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Polygon API error {resp.status_code}: {resp.text[:200]}"
            )

        data = resp.json()

        if data.get("status") not in ("OK", "DELAYED"):
            raise RuntimeError(
                f"Polygon returned status={data.get('status')}: "
                f"{data.get('error', data.get('message', ''))}"
            )

        results = data.get("results", [])
        for bar in results:
            timestamps.append(bar["t"])          # unix ms
            opens.append(float(bar["o"]))
            highs.append(float(bar["h"]))
            lows.append(float(bar["l"]))
            closes.append(float(bar["c"]))
            volumes.append(float(bar.get("v", 0)))

        next_url = data.get("next_url")
        # next_url from Polygon already includes apiKey
        params = None

    if not timestamps:
        raise RuntimeError(
            f"No data returned for {symbol} {timeframe} "
            f"({from_str} → {to_str}). "
            "Check the ticker symbol and ensure your Polygon plan supports it."
        )

    return {
        "timestamps": timestamps,
        "opens":      opens,
        "highs":      highs,
        "lows":       lows,
        "closes":     closes,
        "volumes":    volumes,
        "symbol":     symbol,
        "timeframe":  timeframe,
        "bar_count":  len(timestamps),
    }


def fetch_ohlcv_mock(
    symbol:    str = "C:EURUSD",
    timeframe: str = "1h",
    bars:      int = 500,
) -> Dict[str, List]:
    """
    Generate synthetic OHLCV data for testing without a Polygon API key.
    Produces a realistic-looking trending market with pullbacks.
    """
    import math
    import random

    random.seed(42)
    ts_base = 1_700_000_000_000
    interval_ms = {
        "1m":  60_000,
        "5m":  300_000,
        "15m": 900_000,
        "1h":  3_600_000,
        "4h":  14_400_000,
        "1d":  86_400_000,
    }.get(timeframe, 3_600_000)

    base = 1.2000 if symbol.startswith("C:") else 100.0
    mid  = []
    price = base
    for i in range(bars):
        trend_component  = 0.0001 * math.sin(2 * math.pi * i / (bars * 0.4))
        cycle_component  = 0.005  * math.sin(2 * math.pi * i / 30)
        noise_component  = random.gauss(0, 0.001)
        price += trend_component + cycle_component * 0.1 + noise_component * 0.001
        mid.append(price)

    timestamps = [ts_base + i * interval_ms for i in range(bars)]
    atr_est    = 0.001

    opens  = [m - random.uniform(0, atr_est * 0.3) for m in mid]
    highs  = [m + random.uniform(atr_est * 0.3, atr_est) for m in mid]
    lows   = [m - random.uniform(atr_est * 0.3, atr_est) for m in mid]
    closes = [m + random.uniform(-atr_est * 0.2, atr_est * 0.2) for m in mid]
    volumes = [random.uniform(1000, 10000) for _ in range(bars)]

    return {
        "timestamps": timestamps,
        "opens":      opens,
        "highs":      highs,
        "lows":       lows,
        "closes":     closes,
        "volumes":    volumes,
        "symbol":     symbol,
        "timeframe":  timeframe,
        "bar_count":  bars,
        "mock":       True,
    }

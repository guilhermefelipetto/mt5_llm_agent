"""
Fonte de dados via yfinance.

Usada como fallback quando MT5 não está disponível.
"""

import pandas as pd
import yfinance as yf

from data.cache import cache

_TTL    = {"1m": 60, "5m": 300, "1h": 3600, "4h": 14400, "1d": 86400}
_PERIOD = {"1m": "1d",  "5m": "5d", "1h": "60d", "1d": "2y"}

# 4h não existe no yfinance - derivado via resample de 1h
_RESAMPLE_FROM = {"4h": ("1h", "4h")}


def _fetch_native(yf_symbol: str, tf: str) -> pd.DataFrame | None:
    cache_key = f"yf_{yf_symbol}_{tf}"
    cached = cache.get(cache_key, _TTL[tf])
    if cached is not None:
        return cached

    try:
        df = yf.download(
            yf_symbol,
            period=_PERIOD[tf],
            interval=tf,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            print(f"[!] yfinance: sem dados para {yf_symbol} {tf}")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index = pd.to_datetime(df.index, utc=True)
        cache.set(cache_key, df)
        return df
    except Exception as e:
        print(f"[!] yfinance erro {yf_symbol} {tf}: {e}")
        return None


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    return df.resample(rule).agg(agg).dropna(subset=["Open", "High", "Low", "Close"])


def get_ohlcv_yf(yf_symbol: str, timeframes: list[str]) -> dict[str, pd.DataFrame]:
    needed_native: set[str] = set()
    for tf in timeframes:
        if tf in _RESAMPLE_FROM:
            needed_native.add(_RESAMPLE_FROM[tf][0])
        elif tf in _PERIOD:
            needed_native.add(tf)
        else:
            print(f"[!] yfinance: timeframe '{tf}' não suportado")

    native: dict[str, pd.DataFrame] = {}
    for tf in needed_native:
        df = _fetch_native(yf_symbol, tf)
        if df is not None:
            native[tf] = df

    result: dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        if tf in _RESAMPLE_FROM:
            source_tf, rule = _RESAMPLE_FROM[tf]
            if source_tf in native:
                result[tf] = _resample(native[source_tf], rule)
        elif tf in native:
            result[tf] = native[tf]

    return result

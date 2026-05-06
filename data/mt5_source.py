"""
Fonte de dados via MetaTrader 5 Python API.

Linux (Wine):
    Requer mt5linux + bridge rodando em segundo plano.
    Ver README.md → seção "Configuração MT5 no Linux".

Windows:
    Usa MetaTrader5 nativo - nenhuma configuração extra.

Vantagem sobre yfinance:
    - Dados do próprio broker (sem divergência de preço)
    - Histórico muito maior (ex: 5000 barras 1h ≈ 3 anos)
    - Tick volume real (se disponível)
"""

import pandas as pd

from config import settings
from data.cache import cache


# Quantas barras buscar por timeframe - muito mais que o yfinance permite
_BARS_COUNT = {
    "1m":   1440,   # 1 dia (evitar sobrecarga)
    "5m":   2880,   # 10 dias
    "1h":   5000,   # ≈ 3 anos de barras horárias
    "4h":   1500,   # ≈ 2.5 anos
    "1d":   1500,   # ≈ 6 anos
}

_TTL = {"1m": 60, "5m": 300, "1h": 3600, "4h": 14400, "1d": 86400}


def get_mt5_client():
    """
    Retorna objeto/módulo MT5 compatível.

    Tenta mt5linux (Linux/Wine) primeiro; cai para MetaTrader5 nativo (Windows).
    Ambos expõem a mesma API: initialize(), copy_rates_from_pos(), shutdown(), etc.
    """
    try:
        from mt5linux import MetaTrader5
        return MetaTrader5(host=settings.mt5_host, port=settings.mt5_port)
    except ImportError:
        pass
    try:
        import MetaTrader5 as mt5
        return mt5
    except ImportError:
        return None


def _tf_const(mt5_obj, tf: str):
    """Mapeia string de timeframe para constante MT5."""
    mapping = {
        "1m":  mt5_obj.TIMEFRAME_M1,
        "5m":  mt5_obj.TIMEFRAME_M5,
        "1h":  mt5_obj.TIMEFRAME_H1,
        "4h":  mt5_obj.TIMEFRAME_H4,
        "1d":  mt5_obj.TIMEFRAME_D1,
    }
    return mapping.get(tf)


def _fetch_rates(mt5_obj, symbol: str, tf: str) -> pd.DataFrame | None:
    cache_key = f"mt5_{symbol}_{tf}"
    cached = cache.get(cache_key, _TTL.get(tf, 300))
    if cached is not None:
        return cached

    tf_const = _tf_const(mt5_obj, tf)
    if tf_const is None:
        print(f"[!] MT5: timeframe '{tf}' não suportado")
        return None

    count = _BARS_COUNT.get(tf, 500)
    rates = mt5_obj.copy_rates_from_pos(symbol, tf_const, 0, count)
    if rates is None or len(rates) == 0:
        print(f"[!] MT5: sem dados para {symbol} {tf} - {mt5_obj.last_error()}")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df = df.rename(columns={
        "open":        "Open",
        "high":        "High",
        "low":         "Low",
        "close":       "Close",
        "tick_volume": "Volume",
    })
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols]

    cache.set(cache_key, df)
    return df


def get_ohlcv_mt5(symbol: str, timeframes: list[str]) -> dict[str, pd.DataFrame] | None:
    """
    Busca OHLCV do MT5 para os timeframes solicitados.

    Retorna None se MT5 não estiver disponível, símbolo não existir,
    ou bridge falhar. Fallback para yfinance é feito em fetcher.py.
    """
    mt5 = get_mt5_client()
    if mt5 is None:
        print("[!] MT5: pacote não instalado", flush=True)
        return None

    if not mt5.initialize():
        print(f"[!] MT5: initialize falhou - {mt5.last_error()}", flush=True)
        return None

    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            print(f"[!] MT5: símbolo '{symbol}' não existe - verifique SYMBOL no .env", flush=True)
            return None

        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                print(f"[!] MT5: falha ao selecionar '{symbol}'", flush=True)
                return None

        result: dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            df = _fetch_rates(mt5, symbol, tf)
            if df is not None:
                result[tf] = df
    finally:
        mt5.shutdown()

    if not result:
        return None

    n_bars = {tf: len(df) for tf, df in result.items()}
    print(f"[+] MT5 dados ({symbol}): {n_bars}", flush=True)
    return result

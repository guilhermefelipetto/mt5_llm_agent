"""
Camada de aquisição de dados - interface unificada MT5 + yfinance.

DATA_SOURCE=mt5      → tenta MT5 primeiro; fallback automático para yfinance
DATA_SOURCE=yfinance → usa apenas yfinance (sem dependência de MT5)

Vantagem do MT5:
  - Dados do próprio broker (sem gap de preço entre análise e execução)
  - Histórico maior (ex: 5000 barras de 1h vs. ~1400 do yfinance)
"""

from config import settings
from data.mt5_source import get_ohlcv_mt5
from data.yf_source import get_ohlcv_yf

import pandas as pd


def get_ohlcv(yf_symbol: str, timeframes: list[str]) -> dict[str, pd.DataFrame]:
    """
    Retorna dict {tf: DataFrame OHLCV} para os timeframes solicitados.

    Prioriza MT5 quando DATA_SOURCE=mt5; cai para yfinance se MT5 falhar.
    """
    if settings.data_source == "mt5":
        result = get_ohlcv_mt5(settings.symbol, timeframes)
        if result:
            return result
        print("[!] MT5 indisponível - usando yfinance como fallback", flush=True)

    return get_ohlcv_yf(yf_symbol, timeframes)

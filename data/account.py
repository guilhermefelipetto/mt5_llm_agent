"""
Leitura de equity da conta MT5 - usada pelo position sizing dinâmico
e pelo circuit breaker para calibrar limites como % do capital.

Quando MT5 não está disponível (DATA_SOURCE=yfinance ou bridge offline)
cai para `ACCOUNT_EQUITY_FALLBACK` do .env. Isso evita que a falta de
conexão pause o agente - o sizing fica conservador e segue funcionando.
"""

from config import settings
from data.mt5_source import get_mt5_client


def get_account_equity() -> float:
    """Retorna equity (saldo flutuante) da conta MT5, ou fallback do .env."""
    if settings.data_source != "mt5":
        return settings.account_equity_fallback

    mt5 = get_mt5_client()
    if mt5 is None or not mt5.initialize():
        return settings.account_equity_fallback

    try:
        info = mt5.account_info()
        if info is None:
            return settings.account_equity_fallback
        return float(info.equity)
    finally:
        mt5.shutdown()

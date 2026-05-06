import datetime

import pandas as pd

from config import settings
from features.garch import fit_garch
from features.har_rv import fit_har_rv
from features.indicators import compute as compute_indicators
from features.statistics import compute_stats

_SESSIONS = [
    ("Tokyo",    0,  9),
    ("London",   8, 17),
    ("New York", 13, 22),
]

# Timeframes que recebem análise estatística (custosa) - apenas os maiores
_STATS_TIMEFRAMES = {"1h", "4h", "1d"}


def _active_sessions(utc_hour: int) -> list[str]:
    active = [name for name, start, end in _SESSIONS if start <= utc_hour < end]
    return active or ["Off-hours"]


def _volatility_regime(atr_pips: float) -> str:
    if atr_pips < 5:
        return "baixa"
    if atr_pips > 15:
        return "alta"
    return "normal"


def build_context(ohlcv: dict[str, pd.DataFrame], symbol: str) -> dict:
    now = datetime.datetime.now(datetime.timezone.utc)
    tf_data: dict[str, dict] = {}
    latest_price: float | None = None
    ref_atr_pips: float | None = None

    for tf, df in ohlcv.items():
        ind = compute_indicators(df)
        if not ind:
            continue

        if tf in _STATS_TIMEFRAMES:
            stats = compute_stats(df)
            if stats:
                ind["statistics"] = stats

        tf_data[tf] = ind

        # Preço de referência: prefere o menor TF disponível (1m ou 5m)
        if tf in ("1m", "5m") or latest_price is None:
            latest_price = ind["price"]
        # ATR de referência para SL/TP: prefere 1h
        if tf == "1h" or ref_atr_pips is None:
            ref_atr_pips = ind["atr_pips"]

    # Modelos de volatilidade condicional - usam série 1h (mais longa e estável)
    vol_forecast: dict[str, dict] = {}
    df_1h = ohlcv.get("1h")
    if df_1h is not None:
        garch_result = fit_garch(df_1h, pip_size=settings.pip_size)
        if garch_result:
            vol_forecast["garch"] = garch_result

        har_result = fit_har_rv(df_1h, pip_size=settings.pip_size)
        if har_result:
            vol_forecast["har_rv"] = har_result

    return {
        "symbol": symbol,
        "current_price": latest_price,
        "sessions": _active_sessions(now.hour),
        "volatility_regime": _volatility_regime(ref_atr_pips or 0),
        "vol_forecast": vol_forecast,
        "timeframes": tf_data,
        "timestamp": now.isoformat(),
    }

"""
GARCH(1,1) - estimativa de volatilidade condicional para SL/TP.

Modelo:
    r_t = σ_t · ε_t,  ε_t ~ N(0,1)
    σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}

Persistência = α + β:
    > 0.95 → clusters de alta volatilidade duram muito
    < 0.85 → volatilidade reverte rápido

Usa a biblioteca `arch` (GARCH canônico com MLE).
"""

import numpy as np
import pandas as pd


def fit_garch(df: pd.DataFrame, pip_size: float = 0.0001, min_n: int = 200) -> dict | None:
    """
    Ajusta GARCH(1,1) nos log-retornos de `df` e retorna σ_{t+1} previsto.

    Parâmetros:
        df       : DataFrame OHLCV com coluna Close
        pip_size : tamanho do pip em unidades de preço (EURUSD → 0.0001)
        min_n    : mínimo de observações; retorna None se insuficiente

    Retorna:
        {
            "sigma_pips"  : float  - desvio-padrão previsto em pips
            "alpha"       : float  - coeficiente ARCH (α)
            "beta"        : float  - coeficiente GARCH (β)
            "persistence" : float  - α + β
        }
        ou None se ajuste falhar.
    """
    try:
        from arch import arch_model
    except ImportError:
        return None

    close = df["Close"].squeeze()
    if len(close) < min_n:
        return None

    # Escalar para % - melhora estabilidade numérica do MLE
    log_ret = np.log(close).diff().dropna() * 100

    try:
        am = arch_model(log_ret, vol="Garch", p=1, q=1, dist="Normal", rescale=False)
        res = am.fit(disp="off", show_warning=False)
    except Exception as e:
        print(f"[!] GARCH ajuste falhou: {e}")
        return None

    try:
        forecast = res.forecast(horizon=1, reindex=False)
        var_next = float(forecast.variance.iloc[-1, 0])
    except Exception as e:
        print(f"[!] GARCH forecast falhou: {e}")
        return None

    # De volta para escala de preço e pips
    sigma_price = np.sqrt(var_next) / 100
    sigma_pips = sigma_price / pip_size

    # Extrai parâmetros α e β pelo nome (arch indexa por string)
    pnames = list(res.params.index)
    alpha = float(res.params[[n for n in pnames if n.startswith("alpha")][0]]) \
        if any(n.startswith("alpha") for n in pnames) else 0.0
    beta = float(res.params[[n for n in pnames if n.startswith("beta")][0]]) \
        if any(n.startswith("beta") for n in pnames) else 0.0

    return {
        "sigma_pips": round(float(sigma_pips), 2),
        "alpha": round(float(alpha), 4),
        "beta": round(float(beta), 4),
        "persistence": round(float(alpha + beta), 4),
    }

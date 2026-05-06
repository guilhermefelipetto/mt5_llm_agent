"""
HAR-RV (Heterogeneous Autoregressive Realized Volatility) - Corsi (2009).

Modelo original usa RV diária computada a partir de retornos de alta frequência:
    RV_{t+1} = c + β_d·RV_t + β_w·RV^{(w)}_t + β_m·RV^{(m)}_t + ε_t

Adaptação para dados 1h (sem tick ou 1m intraday disponível):
    RV_proxy_t = r_t²   (quadrado do log-retorno no período t)
    RV^{(d)}_t = média dos últimos d_lag períodos   (default 1)
    RV^{(w)}_t = média dos últimos w_lag períodos   (default 5h ≈ 1 sessão)
    RV^{(m)}_t = média dos últimos m_lag períodos   (default 22h ≈ 1 dia)

A heterogeneidade temporal captura a influência de participantes de curto prazo
(1h), intraday (5h) e diário (22h) - análoga à estrutura HAC original.
"""

import numpy as np
import pandas as pd


def fit_har_rv(
    df: pd.DataFrame,
    pip_size: float = 0.0001,
    d_lag: int = 1,
    w_lag: int = 5,
    m_lag: int = 22,
    min_n: int = 60,
) -> dict | None:
    """
    Ajusta HAR-RV via OLS e retorna σ_{t+1} previsto.

    Parâmetros:
        df       : DataFrame OHLCV com coluna Close
        pip_size : tamanho do pip em unidades de preço (EURUSD → 0.0001)
        d_lag    : horizonte "diário" em nº de candles
        w_lag    : horizonte "semanal" em nº de candles
        m_lag    : horizonte "mensal" em nº de candles
        min_n    : mínimo de observações válidas para ajuste OLS

    Retorna:
        {
            "sigma_pips" : float  - desvio-padrão previsto em pips
            "beta_d"     : float  - coeficiente componente de curto prazo
            "beta_w"     : float  - coeficiente componente semanal
            "beta_m"     : float  - coeficiente componente mensal
            "rv_forecast": float  - RV prevista na escala original (r²)
        }
        ou None se insuficiente.
    """
    close = df["Close"].squeeze()
    if len(close) < min_n + m_lag + 1:
        return None

    log_ret = np.log(close).diff().dropna()
    rv = log_ret ** 2  # proxy de RV: quadrado do log-retorno

    df_rv = pd.DataFrame({"rv": rv.values})
    df_rv["rv_d"] = df_rv["rv"].rolling(d_lag).mean()
    df_rv["rv_w"] = df_rv["rv"].rolling(w_lag).mean()
    df_rv["rv_m"] = df_rv["rv"].rolling(m_lag).mean()
    df_rv["rv_next"] = df_rv["rv"].shift(-1)
    df_rv = df_rv.dropna()

    if len(df_rv) < min_n:
        return None

    X = np.column_stack([
        np.ones(len(df_rv)),
        df_rv["rv_d"].values,
        df_rv["rv_w"].values,
        df_rv["rv_m"].values,
    ])
    y = df_rv["rv_next"].values

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception as e:
        print(f"[!] HAR-RV OLS falhou: {e}")
        return None

    c, beta_d, beta_w, beta_m = coeffs

    # Forecast para próximo período usando os valores mais recentes
    n = len(rv)
    last_rv_d = float(rv.iloc[-d_lag:].mean()) if n >= d_lag else float(rv.iloc[-1])
    last_rv_w = float(rv.iloc[-w_lag:].mean()) if n >= w_lag else last_rv_d
    last_rv_m = float(rv.iloc[-m_lag:].mean()) if n >= m_lag else last_rv_w

    rv_forecast = float(c + beta_d * last_rv_d + beta_w * last_rv_w + beta_m * last_rv_m)
    rv_forecast = max(rv_forecast, 1e-12)  # garante não-negativo

    sigma_pips = np.sqrt(rv_forecast) / pip_size

    return {
        "sigma_pips": round(float(sigma_pips), 2),
        "beta_d": round(float(beta_d), 4),
        "beta_w": round(float(beta_w), 4),
        "beta_m": round(float(beta_m), 4),
        "rv_forecast": round(rv_forecast, 10),
    }

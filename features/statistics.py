"""
Testes estatísticos clássicos para séries temporais financeiras.

Implementa:
    - ADF (Augmented Dickey-Fuller): estacionariedade
    - Ljung-Box: autocorrelação serial
    - Expoente de Hurst (R/S analysis): persistência vs reversão à média
    - Variance Ratio Test (Lo-MacKinlay, 1988): diagnóstico de random walk
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


def adf_test(series, regression: str = "c", min_n: int = 50) -> dict | None:
    """
    Augmented Dickey-Fuller - testa raiz unitária.
        H0: série tem raiz unitária (não-estacionária)
        H1: série é estacionária
    `regression`: 'c' (constante), 'ct' (constante + tendência), 'n' (nada).
    """
    s = pd.Series(series).dropna()
    if len(s) < min_n:
        return None
    try:
        stat, pval, lags, *_ = adfuller(s, regression=regression, autolag="AIC")
        return {
            "statistic": round(float(stat), 4),
            "pvalue": round(float(pval), 4),
            "stationary": bool(pval < 0.05),
            "lags": int(lags),
        }
    except Exception as e:
        print(f"[!] ADF falhou: {e}")
        return None


def ljung_box_test(series, lags: int = 10, min_n: int = 30) -> dict | None:
    """
    Ljung-Box - testa autocorrelação conjunta até `lags`.
        H0: não há autocorrelação significativa (ruído branco)
        H1: há autocorrelação
    Aplicar tipicamente em retornos (ou resíduos de modelos).
    """
    s = pd.Series(series).dropna()
    if len(s) < min_n + lags:
        return None
    try:
        result = acorr_ljungbox(s, lags=[lags], return_df=True)
        return {
            "statistic": round(float(result["lb_stat"].iloc[0]), 4),
            "pvalue": round(float(result["lb_pvalue"].iloc[0]), 4),
            "autocorrelated": bool(result["lb_pvalue"].iloc[0] < 0.05),
            "lags": lags,
        }
    except Exception as e:
        print(f"[!] Ljung-Box falhou: {e}")
        return None


def hurst_exponent(series, min_lag: int = 4, max_lag: int = 64,
                   min_n: int = 200) -> dict | None:
    """
    Expoente de Hurst via R/S analysis clássico (Mandelbrot/Wallis).

    Para janelas de tamanho n:
        R(n) = max(cumsum(deviations)) - min(cumsum(deviations))
        S(n) = std(window)
        E[R(n)/S(n)] ~ c · n^H

    Interpretação (sobre preços):
        H ≈ 0.5  → random walk (Browniano)
        H > 0.5  → persistente / trending
        H < 0.5  → reversão à média / anti-persistente
    """
    arr = np.asarray(pd.Series(series).dropna(), dtype=float)
    if len(arr) < min_n:
        return None

    lags = np.unique(np.logspace(np.log10(min_lag), np.log10(max_lag), 12).astype(int))
    lags = [n for n in lags if 2 <= n < len(arr) // 4]
    if len(lags) < 4:
        return None

    rs_values = []
    for n in lags:
        m = len(arr) // n
        if m < 2:
            continue
        chunks_rs = []
        for i in range(m):
            chunk = arr[i * n:(i + 1) * n]
            cumdev = np.cumsum(chunk - chunk.mean())
            R = cumdev.max() - cumdev.min()
            S = chunk.std(ddof=1)
            if S > 0:
                chunks_rs.append(R / S)
        if chunks_rs:
            rs_values.append((n, float(np.mean(chunks_rs))))

    if len(rs_values) < 4:
        return None

    log_n = np.log([v[0] for v in rs_values])
    log_rs = np.log([v[1] for v in rs_values])
    slope, _ = np.polyfit(log_n, log_rs, 1)
    h = float(slope)

    if h > 0.55:
        regime = "persistente"
    elif h < 0.45:
        regime = "reversao_a_media"
    else:
        regime = "random_walk"

    return {"hurst": round(h, 3), "regime": regime}


def variance_ratio_test(series, q: int = 5, min_n: int = 50) -> dict | None:
    """
    Variance Ratio Test (Lo-MacKinlay, 1988) - testa hipótese de random walk.

    VR(q) = Var(r_t + r_{t-1} + ... + r_{t-q+1}) / (q · Var(r_t))

    Sob RW: VR = 1.
    VR > 1 → autocorrelação positiva (momentum/tendência).
    VR < 1 → autocorrelação negativa (reversão à média).

    Z-estatística heteroskedástica consistente (HC) de Lo-MacKinlay eq. 18.
    """
    s = pd.Series(series).dropna()
    if len(s) < min_n + q:
        return None

    log_ret = np.log(s).diff().dropna().values
    n = len(log_ret)
    if n < min_n:
        return None

    try:
        mu = log_ret.mean()
        var1 = float(((log_ret - mu) ** 2).sum() / (n - 1))
        if var1 < 1e-16:
            return None

        # Retornos de q períodos com overlapping
        ret_q = np.array([log_ret[i:i + q].sum() for i in range(n - q + 1)])
        var_q = float(((ret_q - q * mu) ** 2).sum() / (len(ret_q) * q))

        vr = var_q / var1

        # θ heteroskedástico (Lo-MacKinlay, 1988)
        denom = float(((log_ret - mu) ** 2).sum() ** 2)
        theta = 0.0
        for j in range(1, q):
            w_j = (2 * (q - j) / q) ** 2
            cross = float(
                ((log_ret[j:] - mu) ** 2 * (log_ret[:n - j] - mu) ** 2).sum()
            )
            theta += w_j * (cross / denom if denom > 0 else 0)

        z = (vr - 1) / (np.sqrt(theta / n) + 1e-12)
        pvalue = float(2 * norm.sf(abs(z)))

        if vr > 1.05:
            regime = "trending"
        elif vr < 0.95:
            regime = "mean_reverting"
        else:
            regime = "random_walk"

        return {
            "vr": round(float(vr), 4),
            "z_stat": round(float(z), 4),
            "pvalue": round(pvalue, 4),
            "regime": regime,
            "q": q,
        }
    except Exception as e:
        print(f"[!] VRT falhou: {e}")
        return None


def compute_stats(df: pd.DataFrame) -> dict:
    """
    Calcula o pacote padrão de testes para um DataFrame OHLCV.
    ADF e Hurst aplicados nos preços de fechamento; Ljung-Box e VRT nos log-retornos.
    """
    if df is None or len(df) < 50:
        return {}

    close = df["Close"].squeeze()
    log_returns = np.log(close).diff().dropna()

    return {
        "adf": adf_test(close),
        "ljung_box": ljung_box_test(log_returns),
        "hurst": hurst_exponent(close.values),
        "vrt": variance_ratio_test(close, q=5),
    }

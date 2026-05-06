"""
Estimadores de volatilidade plugáveis para cálculo de SL/TP.

Ordem de preferência configurável via VOL_ESTIMATOR:
    "garch"  → GARCH(1,1) σ_{t+1} (forward-looking, modelado)
    "har_rv" → HAR-RV σ_{t+1} (forward-looking, baseado em RV histórica)
    "atr"    → ATR 1h puro (backward-looking, robusto)

Se o estimador preferido não estiver disponível (dados insuficientes, falha),
faz fallback automático em cascata: garch → har_rv → atr.
"""

from dataclasses import dataclass

from config import settings


@dataclass
class VolatilityEstimate:
    sigma: float        # em unidades de preço (ex: 0.00082 para EURUSD)
    sigma_pips: float   # em pips (ex: 8.2)
    method: str         # "garch" | "har_rv" | "atr"


def from_context(context: dict) -> VolatilityEstimate:
    """
    Seleciona o melhor estimador disponível no contexto para SL/TP.

    `context["vol_forecast"]` é populado por context_builder com GARCH e HAR-RV.
    Se ausente ou vazio, cai para ATR 1h (sempre disponível).
    """
    preferred = settings.vol_estimator
    pip_size = settings.pip_size
    vf = context.get("vol_forecast", {})

    # Tenta o estimador preferido, depois o alternativo, depois ATR
    order = ["garch", "har_rv"] if preferred != "har_rv" else ["har_rv", "garch"]

    if preferred != "atr":
        for method in order:
            entry = vf.get(method)
            if entry and entry.get("sigma_pips") and entry["sigma_pips"] > 0:
                sigma_pips = float(entry["sigma_pips"])
                return VolatilityEstimate(
                    sigma=round(sigma_pips * pip_size, 6),
                    sigma_pips=sigma_pips,
                    method=method,
                )

    # Fallback ATR: prefere 1h, depois 5m, depois 1d
    tf = context.get("timeframes", {})
    atr_raw = (
        tf.get("1h", {}).get("atr")
        or tf.get("5m", {}).get("atr")
        or tf.get("1d", {}).get("atr")
    )
    atr_val = float(atr_raw) if atr_raw else (10.0 * pip_size)
    return VolatilityEstimate(
        sigma=atr_val,
        sigma_pips=round(atr_val / pip_size, 2),
        method="atr",
    )

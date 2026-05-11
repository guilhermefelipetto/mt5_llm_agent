"""
Constrói o prompt enviado ao LLM, com hierarquia condicional de TFs por
horizonte e suporte a múltiplas posições simultâneas (v1.7).

Princípio fundamental: o TF dominante depende do horizonte da operação,
não é fixo. Para cada posição aberta, o prompt destaca qual TF usar como
referência primária e quais TFs são apenas viés ou ruído.
"""

from config import HORIZON_PROFILES, settings
from data.positions import OpenPosition


_TF_ORDER = ["1d", "4h", "1h", "5m", "1m"]
_TF_LABELS = {
    "1d": "Macro tendência",
    "4h": "Tendência intermediária",
    "1h": "Confirmação de momentum",
    "5m": "Timing de entrada",
    "1m": "Microestrutura",
}


def _system_prompt() -> str:
    """System prompt construído dinamicamente para refletir os limites
    do agente (max_positions, perfis de horizonte ativos)."""
    profiles_lines = []
    for h, prof in HORIZON_PROFILES.items():
        profiles_lines.append(
            f"  - {h}: TF dominante {prof['dominant_tfs']}; SL≈{prof['sl_mult']}σ, "
            f"TP≈{prof['tp_mult']}σ; time exit {prof['max_age_hours']}h"
        )
    profiles_block = "\n".join(profiles_lines)

    return (
        "Você é um analista quantitativo de mercados forex com formação em "
        "estatística e séries temporais.\n\n"

        "FORMATO DE RESPOSTA (JSON estrito):\n"
        '  - "action": "OPEN_LONG" | "OPEN_SHORT" | "HOLD" | "CLOSE" | "TIGHTEN_STOP"\n'
        '  - "intended_horizon": "scalp" | "intraday" | "swing" | null\n'
        '  - "position_id": ticket da posição-alvo (CLOSE/TIGHTEN_STOP) ou null\n'
        '  - "confidence": 0.0 a 1.0\n'
        '  - "reasoning": justificativa técnica curta (1-2 frases)\n'
        '  - "new_sl": novo SL para TIGHTEN_STOP, senão null\n\n'

        "VOCABULÁRIO:\n"
        "- OPEN_LONG / OPEN_SHORT: abre nova posição. intended_horizon "
        "obrigatório. Sistema calcula SL/TP automaticamente conforme o "
        "horizonte escolhido.\n"
        "- HOLD: não fazer nada. Slot vazio é melhor que slot com trade ruim - "
        "não opere só porque há slots disponíveis.\n"
        "- CLOSE: encerra uma posição existente antes de SL/TP. Use quando a "
        "tese original quebrou. Com múltiplas posições abertas, position_id "
        "é obrigatório.\n"
        "- TIGHTEN_STOP: aperta o SL para travar lucro acumulado. Só "
        "permitido movendo na direção do preço (long: subir SL; short: "
        "descer SL). REGRA QUANTITATIVA: só faz sentido apertar quando o "
        "trade já percorreu pelo menos 50% do caminho até o TP — antes "
        "disso, o ruído normal de mercado é maior que o lucro e apertar "
        "transforma o trade num scalp de baixa expectativa. Quando "
        "apertar, deixe pelo menos 30% do lucro acumulado como margem; "
        "colar o SL próximo do preço atual significa fechar no próximo "
        "tick contrário. Sistema rejeita tanto afrouxamentos quanto "
        "apertos prematuros.\n\n"

        f"PERFIS DE HORIZONTE:\n{profiles_block}\n\n"

        "HIERARQUIA CONDICIONAL DE TIMEFRAMES:\n"
        "Cada horizonte tem TF DOMINANTE diferente. NÃO aplique a mesma "
        "regra para todos os trades:\n"
        "- Para SWING (D1 dominante): D1 e 4H são primários; H1 é contexto; "
        "M5 e M1 são RUÍDO - não use como veto.\n"
        "- Para INTRADAY (H1 dominante): H1 e M30 são primários; 4H/D1 dão "
        "viés macro (mas não vetam); M5 dá timing de entrada; M1 é ruído.\n"
        "- Para SCALP (M5 dominante): M5 e M1 são primários; M15/H1 dão "
        "estrutura de curto prazo; D1 é apenas viés muito amplo (não veta).\n"
        "Nunca recuse um trade de scalp porque D1 está em direção contrária - "
        "para scalp, D1 é ruído na escala em que você opera.\n\n"

        f"MULTI-POSITION (até {settings.max_positions} simultâneas):\n"
        "- Permitido abrir múltiplas posições, MAS uma por (lado, horizonte). "
        "Não dobre aposta no mesmo setup - se já há LONG intraday aberto, "
        "novo OPEN_LONG só se for de outro horizonte (scalp ou swing).\n"
        f"- Risco máximo agregado = {settings.max_positions} × "
        f"{settings.risk_per_trade_pct}% = {settings.max_positions * settings.risk_per_trade_pct:.1f}% "
        "do equity se todas baterem SL juntas.\n\n"

        "REGRAS DE ANÁLISE:\n"
        "1. Considere os regimes estatísticos: Hurst (persistência global) e "
        "VRT (autocorrelação local). Em regime persistente (Hurst > 0.55), "
        "favoreça continuação; em reversão à média (Hurst < 0.45), favoreça "
        "reversão; em random walk, prefira HOLD salvo confluência forte.\n"
        "2. Use as previsões de volatilidade (GARCH/HAR-RV) para contextualizar "
        "o risco. Persistência GARCH alta (α+β > 0.95) indica que a volatilidade "
        "atual tende a continuar.\n"
        "3. GESTÃO DE POSIÇÃO ABERTA: avalie se a tese original ainda vale. "
        "Considere CLOSE se o regime virou contra; TIGHTEN_STOP se acumulou "
        "lucro relevante e quer travar parte; HOLD se a tese segue íntegra. "
        "Use o TF dominante do horizonte da posição como referência primária.\n"
        "4. CALIBRE-SE PELO TRACK RECORD quando disponível: se a seção "
        "[SEU TRACK RECORD] mostrar baixo win rate (<40%) num bucket "
        "(regime, horizonte, lado) específico, RECONSIDERE entrar nesse "
        "bucket — talvez o lado oposto, outro horizonte ou HOLD. Win rate "
        "alto (>60%) com amostra >= 10 é validação parcial. Amostras n<10 "
        "são ruído.\n"
        "5. Não force operação - HOLD é resposta válida e frequentemente "
        "correta."
    )


def _format_statistics(stats: dict) -> str:
    parts = []
    if stats.get("hurst"):
        h = stats["hurst"]
        parts.append(f"Hurst={h['hurst']} ({h['regime']})")
    if stats.get("vrt"):
        v = stats["vrt"]
        parts.append(f"VR({v['q']})={v['vr']} ({v['regime']}, p={v['pvalue']})")
    if stats.get("adf"):
        ss = "estacionária" if stats["adf"]["stationary"] else "não-estacionária"
        parts.append(f"ADF p={stats['adf']['pvalue']} ({ss})")
    if stats.get("ljung_box"):
        lb = "autocorrel." if stats["ljung_box"]["autocorrelated"] else "ruído branco"
        parts.append(f"LB p={stats['ljung_box']['pvalue']} ({lb})")
    return " | ".join(parts)


def _format_vol_forecast(vol_forecast: dict) -> str:
    lines = []
    if "garch" in vol_forecast:
        g = vol_forecast["garch"]
        lines.append(
            f"  GARCH(1,1): σ_{{t+1}} = {g['sigma_pips']} pips "
            f"(α={g['alpha']}, β={g['beta']}, persist.={g['persistence']})"
        )
    if "har_rv" in vol_forecast:
        h = vol_forecast["har_rv"]
        lines.append(
            f"  HAR-RV:     σ_{{t+1}} = {h['sigma_pips']} pips "
            f"(β_d={h['beta_d']}, β_w={h['beta_w']}, β_m={h['beta_m']})"
        )
    return "\n".join(lines)


def _format_positions(positions: list[OpenPosition]) -> list[str]:
    if not positions:
        return [
            f"[POSIÇÕES ATUAIS] Nenhuma — você está flat. "
            f"Pode abrir até {settings.max_positions} posições "
            f"(uma por horizonte).",
            "",
        ]
    lines = [f"[POSIÇÕES ATUAIS] {len(positions)} aberta(s):"]
    for i, p in enumerate(positions, 1):
        sign = "+" if p.pnl_pips >= 0 else ""
        prof = HORIZON_PROFILES[p.intended_horizon]
        inferred_tag = "  (horizonte inferido)" if p.horizon_inferred else ""
        lines.append(
            f"  #{i} ticket={p.ticket}  {p.side} {p.intended_horizon}"
            f"  TF dominante: {prof['dominant_tfs']}{inferred_tag}"
        )
        lines.append(
            f"     Entrada: {p.entry_price} → Atual: {p.current_price}  "
            f"P&L: {sign}{p.pnl_pips} pips ({sign}{p.pnl_pct:.3f}%)"
        )
        lines.append(
            f"     SL: {p.sl_price}  TP: {p.tp_price}  "
            f"Idade: {p.age_minutes:.0f}min "
            f"(time exit em {prof['max_age_hours']}h)"
        )
        if p.open_reasoning:
            lines.append(f'     Tese original: "{p.open_reasoning}"')
    lines.append(
        "  Para CLOSE/TIGHTEN_STOP, informe position_id (ticket) da posição-alvo."
    )
    lines.append("")
    return lines


def _current_regime(context: dict) -> str | None:
    """Hurst regime do 1h no contexto atual — chave para casar com track record."""
    tfs = context.get("timeframes") or {}
    h1 = tfs.get("1h") or {}
    stats = h1.get("statistics") or {}
    return (stats.get("hurst") or {}).get("regime")


def build_prompt(
    context: dict,
    positions: list[OpenPosition],
    calibration: dict | None = None,
) -> tuple[str, str]:
    lines = [
        "=== Contexto de Mercado ===",
        f"Ativo: {context['symbol']}  |  Preço: {context['current_price']}  |  "
        f"Sessões ativas: {', '.join(context['sessions'])}",
        f"Regime de volatilidade (ATR 1H): {context['volatility_regime']}",
        "",
    ]

    if calibration is not None:
        from analytics.calibration import format_for_prompt
        lines.extend(format_for_prompt(calibration, _current_regime(context)))

    lines.extend(_format_positions(positions))

    vol_forecast = context.get("vol_forecast", {})
    if vol_forecast:
        lines.append("[VOLATILIDADE PREVISTA — modelos condicionais]")
        lines.append(_format_vol_forecast(vol_forecast))
        lines.append("")

    tf_data = context.get("timeframes", {})
    for tf in _TF_ORDER:
        if tf not in tf_data:
            continue
        data = tf_data[tf]
        label = _TF_LABELS.get(tf, tf.upper())
        lines.append(f"[{tf.upper()} — {label}]")
        lines.append(f"  Tendência (EMA20/50): {data['trend']}")
        lines.append(f"  RSI: {data['rsi']} → {data['rsi_signal']}")
        lines.append(f"  MACD: {data['macd_signal']} (hist {data['macd_hist']})")
        lines.append(f"  Bollinger position: {data['bb_position']:.0%}")
        lines.append(f"  ATR: {data['atr_pips']} pips")
        lines.append(
            f"  Retornos: {data['ret_5']:+.4f}% (5c)  |  {data['ret_20']:+.4f}% (20c)"
        )
        stats = data.get("statistics")
        if stats:
            stats_line = _format_statistics(stats)
            if stats_line:
                lines.append(f"  Estatística: {stats_line}")
        lines.append("")

    lines.append(
        'Retorne sua decisão como JSON: '
        '{"action": "...", "intended_horizon": "...", "position_id": null, '
        '"confidence": 0.0, "reasoning": "...", "new_sl": null}'
    )

    return _system_prompt(), "\n".join(lines)

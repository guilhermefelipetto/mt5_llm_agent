"""
Constrói o prompt enviado ao LLM, com hierarquia top-down dos timeframes.

Princípio: timeframes maiores definem direção macro; menores definem timing.
A ordem do prompt e a instrução do system prompt reforçam essa hierarquia.

A partir da v1.4 o prompt também injeta a posição aberta atual quando existe,
permitindo decisões stateful (CLOSE antes de SL/TP, TIGHTEN_STOP, HOLD).
"""

from data.positions import OpenPosition

# Ordem hierárquica top-down (maior → menor TF)
_TF_ORDER = ["1d", "4h", "1h", "5m", "1m"]

_TF_LABELS = {
    "1d": "Macro tendência",
    "4h": "Tendência intermediária",
    "1h": "Confirmação de momentum",
    "5m": "Timing de entrada",
    "1m": "Microestrutura",
}

SYSTEM_PROMPT = (
    "Você é um analista quantitativo de mercados forex com formação em "
    "estatística e séries temporais. Analise o contexto multi-timeframe abaixo "
    "e produza um objeto JSON com os campos:\n"
    '  - "action": "OPEN_LONG" | "OPEN_SHORT" | "HOLD" | "CLOSE" | "TIGHTEN_STOP"\n'
    '  - "confidence": número entre 0.0 e 1.0\n'
    '  - "reasoning": justificativa técnica curta (1-2 frases)\n'
    '  - "new_sl": preço absoluto do novo SL (apenas para TIGHTEN_STOP, '
    'caso contrário null)\n\n'
    "VOCABULÁRIO DE AÇÕES:\n"
    "- OPEN_LONG / OPEN_SHORT: abre nova posição. Se já houver posição na "
    "direção oposta, o sistema fecha e inverte automaticamente.\n"
    "- HOLD: não fazer nada. Se há posição aberta, deixa o trade correr até "
    "SL/TP. Se está flat, não opera.\n"
    "- CLOSE: encerra a posição atual ANTES de SL/TP - use quando a tese "
    "original do trade quebrou (regime mudou, tendência inverteu, contexto "
    "macro virou contra).\n"
    "- TIGHTEN_STOP: aperta o SL para travar lucro acumulado. Só permitido "
    "MOVENDO na direção do preço (long: subir SL; short: descer SL). "
    "O sistema rejeita afrouxamentos automaticamente.\n\n"
    "REGRAS DE ANÁLISE:\n"
    "1. Tendências em timeframes MAIORES (D1, 4H) DOMINAM sinais de timeframes menores. "
    "Não opere contra a tendência macro a menos que haja evidência muito forte.\n"
    "2. Use timeframes menores apenas para timing - não para definir direção.\n"
    "3. Considere os regimes estatísticos: Hurst (persistência global) e VRT (autocorrelação local). "
    "Em regime persistente (Hurst > 0.55, VR > 1), favoreça estratégias de continuação; "
    "em reversão à média (Hurst < 0.45, VR < 1), favoreça reversão; "
    "em random walk, prefira HOLD salvo confluência forte.\n"
    "4. Use as previsões de volatilidade (GARCH/HAR-RV) para contextualizar o risco. "
    "Alta persistência GARCH (α+β > 0.95) indica que a volatilidade atual tende a continuar.\n"
    "5. Se houver conflito entre timeframes adjacentes, prefira HOLD.\n"
    "6. GESTÃO DE POSIÇÃO ABERTA: quando a seção [POSIÇÃO ATUAL] estiver presente, "
    "avalie se a tese original ainda vale. Considere CLOSE se o regime virou; "
    "TIGHTEN_STOP se o trade já acumulou lucro relevante e quer travar parte; "
    "HOLD se a tese segue íntegra. Não force operação - HOLD é resposta válida."
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


def _format_position(
    pos: OpenPosition, original_reasoning: str | None = None
) -> list[str]:
    pnl_sign = "+" if pos.pnl_pips >= 0 else ""
    lines = [
        "[POSIÇÃO ATUAL]",
        f"  Lado: {pos.side}  |  Lote: {pos.lot}",
        f"  Entrada: {pos.entry_price}  →  Atual: {pos.current_price}",
        f"  P&L: {pnl_sign}{pos.pnl_pips} pips ({pnl_sign}{pos.pnl_pct:.3f}%)",
        f"  SL atual: {pos.sl_price}   TP atual: {pos.tp_price}",
        f"  Aberta há: {pos.age_minutes:.0f} min",
    ]
    if original_reasoning:
        # Auto-crítica: o LLM revisita a tese que ele mesmo articulou na abertura.
        lines.append(f'  Tese original: "{original_reasoning}"')
    lines.append(
        "  Ações disponíveis: HOLD (deixa correr), CLOSE (sair agora), "
        "TIGHTEN_STOP (apertar SL), OPEN_oposto (reverter)."
    )
    lines.append("")
    return lines


def build_prompt(
    context: dict,
    position: OpenPosition | None = None,
    original_reasoning: str | None = None,
) -> tuple[str, str]:
    lines = [
        "=== Contexto de Mercado ===",
        f"Ativo: {context['symbol']}  |  Preço: {context['current_price']}  |  "
        f"Sessões ativas: {', '.join(context['sessions'])}",
        f"Regime de volatilidade (ATR 1H): {context['volatility_regime']}",
        "",
    ]

    if position is not None:
        lines.extend(_format_position(position, original_reasoning))
    else:
        lines.append("[POSIÇÃO ATUAL] Nenhuma - você está flat.")
        lines.append("")

    vol_forecast = context.get("vol_forecast", {})
    if vol_forecast:
        lines.append("[VOLATILIDADE PREVISTA - modelos condicionais]")
        lines.append(_format_vol_forecast(vol_forecast))
        lines.append("")

    tf_data = context.get("timeframes", {})
    for tf in _TF_ORDER:
        if tf not in tf_data:
            continue
        data = tf_data[tf]
        label = _TF_LABELS.get(tf, tf.upper())
        lines.append(f"[{tf.upper()} - {label}]")
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
        '{"action": "...", "confidence": 0.0, "reasoning": "...", "new_sl": null}'
    )

    return SYSTEM_PROMPT, "\n".join(lines)

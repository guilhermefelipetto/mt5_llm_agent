"""
Position sizing dinâmico - substitui o `DEFAULT_LOT` fixo da v1.0–v1.5.

O lote é calculado para que **a perda no SL** atinja exatamente o risco-alvo:

    risco_$ = equity × (RISK_PER_TRADE_PCT/100) × confiança
    lote    = risco_$ / (pip_value_per_lot × distância_SL_em_pips)

Combina natural e simultaneamente:
  - **Confiança** → lote escala linearmente com a confiança do LLM
  - **Volatilidade inversa** → SL maior (alta vol via GARCH/HAR-RV) reduz lote
  - **Risco constante** → cada trade arrisca no máximo a mesma fração do capital

Limitado por MIN_LOT/MAX_LOT e arredondado para o LOT_STEP do broker.
"""

from config import settings


def compute_lot(
    equity: float,
    confidence: float,
    entry_price: float,
    sl_price: float,
) -> float:
    """Lote arredondado para o passo do broker, dentro dos bounds."""
    sl_distance_price = abs(entry_price - sl_price)
    if sl_distance_price <= 0:
        return settings.min_lot

    sl_distance_pips = sl_distance_price / settings.pip_size
    risk_money = equity * (settings.risk_per_trade_pct / 100.0) * confidence

    raw_lot = risk_money / (settings.pip_value_per_lot * sl_distance_pips)

    # Bound antes do arredondamento
    bounded = max(settings.min_lot, min(settings.max_lot, raw_lot))

    # Arredonda para o passo (ex: 0.01) - depois reaplica MIN/MAX para
    # cobrir o caso de raw_lot < min_lot virando 0 no round.
    steps = round(bounded / settings.lot_step)
    final = steps * settings.lot_step
    final = max(settings.min_lot, min(settings.max_lot, final))
    return round(final, 2)

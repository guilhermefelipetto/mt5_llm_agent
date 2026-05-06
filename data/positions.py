"""
Leitura de posições abertas via MT5 - viabiliza decisões stateful do agente.

O LLM precisa enxergar a posição atual para decidir CLOSE, TIGHTEN_STOP
ou inferir que está na direção certa (HOLD) vs. precisa reverter.

Retorna None quando DATA_SOURCE != mt5 ou bridge indisponível: nesse caso
o agente opera stateless (comportamento pré-v1.4).
"""

import datetime
from dataclasses import dataclass, asdict

from config import settings
from data.mt5_source import get_mt5_client


@dataclass
class OpenPosition:
    ticket: int
    symbol: str
    side: str               # "LONG" | "SHORT"
    entry_price: float
    current_price: float
    sl_price: float
    tp_price: float
    lot: float
    opened_at: datetime.datetime
    age_minutes: float
    pnl_pips: float
    pnl_pct: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["opened_at"] = self.opened_at.isoformat()
        return d


def get_open_position(symbol: str) -> OpenPosition | None:
    """Retorna a primeira posição aberta para o símbolo, ou None.

    O EA mantém no máximo 1 posição por símbolo (HasOurPosition guard),
    então a primeira é suficiente.
    """
    if settings.data_source != "mt5":
        return None

    mt5 = get_mt5_client()
    if mt5 is None:
        return None

    if not mt5.initialize():
        return None

    try:
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return None

        p = positions[0]
        side = "LONG" if p.type == mt5.POSITION_TYPE_BUY else "SHORT"

        if side == "LONG":
            pnl_price = p.price_current - p.price_open
        else:
            pnl_price = p.price_open - p.price_current

        pip_size = settings.pip_size
        pnl_pips = pnl_price / pip_size if pip_size else 0.0
        pnl_pct = (pnl_price / p.price_open) * 100 if p.price_open else 0.0

        opened_at = datetime.datetime.fromtimestamp(p.time, tz=datetime.timezone.utc)
        age_min = (
            datetime.datetime.now(datetime.timezone.utc) - opened_at
        ).total_seconds() / 60

        return OpenPosition(
            ticket=int(p.ticket),
            symbol=p.symbol,
            side=side,
            entry_price=float(p.price_open),
            current_price=float(p.price_current),
            sl_price=float(p.sl),
            tp_price=float(p.tp),
            lot=float(p.volume),
            opened_at=opened_at,
            age_minutes=round(age_min, 1),
            pnl_pips=round(pnl_pips, 1),
            pnl_pct=round(pnl_pct, 4),
        )
    finally:
        mt5.shutdown()

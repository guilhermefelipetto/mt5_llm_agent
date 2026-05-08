"""
Leitura de posições abertas via MT5 - viabiliza decisões stateful do agente.

A v1.7 introduz multi-position: o agente pode ter até MAX_POSITIONS abertas
simultaneamente, com a regra de não colidir (lado, horizonte). O horizonte
de cada posição é recuperado a partir do `intended_horizon` do sinal de
abertura em signals.jsonl, casado por proximidade temporal.

Quando MT5 não está disponível, retorna lista vazia: o agente opera stateless
nesse caso (comportamento pré-v1.4 como fallback gracioso).
"""

import datetime
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from config import HORIZON_PROFILES, DEFAULT_HORIZON, settings
from data.mt5_source import get_mt5_client


_SIGNALS_LOG = Path("logs") / "signals.jsonl"


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
    intended_horizon: str   # "scalp" | "intraday" | "swing"
    horizon_inferred: bool  # True se não veio de sinal nosso (default applied)
    open_reasoning: str | None  # tese original do trade, se identificada

    def to_dict(self) -> dict:
        d = asdict(self)
        d["opened_at"] = self.opened_at.isoformat()
        return d

    @property
    def max_age_minutes(self) -> float:
        return HORIZON_PROFILES[self.intended_horizon]["max_age_hours"] * 60.0

    @property
    def cadence_seconds(self) -> int:
        return HORIZON_PROFILES[self.intended_horizon]["cadence_s"]


def _read_signals_log() -> list[dict]:
    if not _SIGNALS_LOG.exists():
        return []
    out = []
    with _SIGNALS_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _match_open_signal(opened_at: datetime.datetime, side: str) -> dict | None:
    """Sinal mais recente cujo lado bate com o trade aberto e cujo created_at
    é anterior à abertura por no máximo 2 × analysis_interval.
    """
    target_action = "OPEN_LONG" if side == "LONG" else "OPEN_SHORT"
    max_lag = datetime.timedelta(seconds=2 * settings.analysis_interval)

    candidates = []
    for s in _read_signals_log():
        if s.get("action") != target_action:
            continue
        try:
            created = datetime.datetime.fromisoformat(s["created_at"])
        except (KeyError, ValueError):
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=datetime.timezone.utc)
        if created > opened_at:
            continue
        if opened_at - created > max_lag:
            continue
        candidates.append((created, s))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _build_position(p, mt5) -> OpenPosition:
    """Constrói OpenPosition a partir de uma estrutura position do MT5,
    recuperando horizonte e tese a partir do signals.jsonl.
    """
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

    sig = _match_open_signal(opened_at, side)
    horizon = (sig or {}).get("intended_horizon")
    inferred = horizon is None
    if horizon not in HORIZON_PROFILES:
        horizon = DEFAULT_HORIZON
        inferred = True

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
        intended_horizon=horizon,
        horizon_inferred=inferred,
        open_reasoning=(sig or {}).get("reasoning"),
    )


def get_open_positions(symbol: str) -> list[OpenPosition]:
    """Lista todas as posições abertas para o símbolo (multi-position v1.7)."""
    if settings.data_source != "mt5":
        return []

    mt5 = get_mt5_client()
    if mt5 is None or not mt5.initialize():
        return []

    try:
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return []
        return [_build_position(p, mt5) for p in positions]
    finally:
        mt5.shutdown()

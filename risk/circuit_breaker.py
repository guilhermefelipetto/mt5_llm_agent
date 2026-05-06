"""
Circuit breaker - bloqueia novas aberturas quando o agente está em maré ruim.

Dois gatilhos, ambos avaliados sobre trades fechados **hoje** (UTC):
  1. Drawdown diário ≥ DAILY_DRAWDOWN_PCT × equity
  2. ≥ MAX_CONSECUTIVE_LOSSES trades perdedores em sequência

Apenas OPEN_LONG/OPEN_SHORT são bloqueados - CLOSE e TIGHTEN_STOP nunca
são, porque reduzem risco. HOLD passa direto. O lock auto-reseta na
virada do dia UTC, evitando deadlock permanente.
"""

import datetime
import json
from dataclasses import dataclass
from pathlib import Path

from config import settings


_TRADES_LOG = Path("logs") / "trades.jsonl"


@dataclass
class CircuitState:
    blocked: bool
    reason: str | None
    daily_pnl: float
    consecutive_losses: int


def _read_today_trades() -> list[dict]:
    if not _TRADES_LOG.exists():
        return []

    today_utc = datetime.datetime.now(datetime.timezone.utc).date()
    out: list[dict] = []
    with _TRADES_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            closed = t.get("closed_at")
            if not closed:
                continue
            try:
                dt = datetime.datetime.fromisoformat(closed)
            except ValueError:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            if dt.date() == today_utc:
                out.append(t)
    return out


def _trailing_losses(trades: list[dict]) -> int:
    """Conta trades perdedores na cauda mais recente (do mais recente pra trás)."""
    sorted_trades = sorted(trades, key=lambda t: t.get("closed_at", ""), reverse=True)
    count = 0
    for t in sorted_trades:
        if (t.get("pnl_money") or 0) < 0:
            count += 1
        else:
            break
    return count


def evaluate(equity: float) -> CircuitState:
    """Estado atual do circuit breaker dado a equity da conta."""
    today = _read_today_trades()
    daily_pnl = sum(float(t.get("pnl_money") or 0.0) for t in today)
    consec_losses = _trailing_losses(today)

    dd_limit = -equity * (settings.daily_drawdown_pct / 100.0)
    if daily_pnl <= dd_limit:
        return CircuitState(
            blocked=True,
            reason=(
                f"drawdown diário {daily_pnl:.2f} ≤ limite {dd_limit:.2f} "
                f"({settings.daily_drawdown_pct}% de {equity:.2f})"
            ),
            daily_pnl=daily_pnl,
            consecutive_losses=consec_losses,
        )

    if consec_losses >= settings.max_consecutive_losses:
        return CircuitState(
            blocked=True,
            reason=f"{consec_losses} perdas consecutivas hoje (limite {settings.max_consecutive_losses})",
            daily_pnl=daily_pnl,
            consecutive_losses=consec_losses,
        )

    return CircuitState(
        blocked=False, reason=None,
        daily_pnl=daily_pnl, consecutive_losses=consec_losses,
    )

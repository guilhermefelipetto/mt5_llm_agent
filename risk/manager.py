import datetime
from dataclasses import dataclass

from config import settings
from data.account import get_account_equity
from data.positions import OpenPosition
from risk.circuit_breaker import CircuitState, evaluate as evaluate_circuit
from risk.sizing import compute_lot
from risk.volatility import from_context as get_vol_estimate


# Vocabulário de ações enviado pro EA. Mantemos enum estável aqui e no
# schema do LLM - qualquer divergência vira HOLD por sanity check.
_VALID_ACTIONS = {
    "OPEN_LONG",
    "OPEN_SHORT",
    "HOLD",
    "CLOSE",
    "TIGHTEN_STOP",
}


@dataclass
class Signal:
    signal_id: str
    symbol: str
    action: str          # OPEN_LONG | OPEN_SHORT | HOLD | CLOSE | TIGHTEN_STOP
    confidence: float
    reasoning: str
    entry_price: float | None
    sl_price: float | None
    tp_price: float | None
    new_sl: float | None
    lot: float
    vol_method: str
    equity: float                # equity no momento da decisão (auditoria)
    circuit_state: dict          # snapshot do circuit breaker
    created_at: datetime.datetime
    expires_at: datetime.datetime

    def is_valid(self) -> bool:
        return datetime.datetime.now(datetime.timezone.utc) < self.expires_at

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "action": self.action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "entry_price": self.entry_price,
            "sl_price": self.sl_price,
            "tp_price": self.tp_price,
            "new_sl": self.new_sl,
            "lot": self.lot,
            "vol_method": self.vol_method,
            "equity": self.equity,
            "circuit_state": self.circuit_state,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "valid": self.is_valid(),
        }


def _downgrade(reason: str, original_reasoning: str) -> tuple[str, str]:
    """Força HOLD com prefixo explicando por quê - útil pro log post-mortem."""
    return "HOLD", f"[downgrade: {reason}] {original_reasoning}"


def _validate_action(
    action: str,
    new_sl: float | None,
    reasoning: str,
    position: OpenPosition | None,
) -> tuple[str, str, float | None]:
    """Guardrails state-aware (v1.4) - coerência ação ↔ estado de posição."""
    if action not in _VALID_ACTIONS:
        a, r = _downgrade(f"action inválida {action!r}", reasoning)
        return a, r, None

    if action in ("OPEN_LONG", "OPEN_SHORT"):
        if position is not None:
            same_side = (
                (action == "OPEN_LONG" and position.side == "LONG")
                or (action == "OPEN_SHORT" and position.side == "SHORT")
            )
            if same_side:
                a, r = _downgrade(f"já estamos {position.side}", reasoning)
                return a, r, None
        return action, reasoning, None

    if action == "CLOSE":
        if position is None:
            a, r = _downgrade("CLOSE sem posição aberta", reasoning)
            return a, r, None
        return action, reasoning, None

    if action == "TIGHTEN_STOP":
        if position is None:
            a, r = _downgrade("TIGHTEN_STOP sem posição", reasoning)
            return a, r, None
        if new_sl is None:
            a, r = _downgrade("TIGHTEN_STOP sem new_sl", reasoning)
            return a, r, None
        if position.side == "LONG" and new_sl <= position.sl_price:
            a, r = _downgrade(
                f"SL não aperta (atual={position.sl_price}, proposto={new_sl})",
                reasoning,
            )
            return a, r, None
        if position.side == "SHORT" and new_sl >= position.sl_price:
            a, r = _downgrade(
                f"SL não aperta (atual={position.sl_price}, proposto={new_sl})",
                reasoning,
            )
            return a, r, None
        margin = settings.pip_size
        if position.side == "LONG" and new_sl >= position.current_price - margin:
            a, r = _downgrade(
                f"SL ultrapassa preço atual ({position.current_price})", reasoning,
            )
            return a, r, None
        if position.side == "SHORT" and new_sl <= position.current_price + margin:
            a, r = _downgrade(
                f"SL ultrapassa preço atual ({position.current_price})", reasoning,
            )
            return a, r, None
        return action, reasoning, round(new_sl, 5)

    return action, reasoning, None


def _apply_time_exit(
    action: str,
    reasoning: str,
    position: OpenPosition | None,
) -> tuple[str, str]:
    """Força CLOSE em trades pendurados há mais que MAX_TRADE_AGE_HOURS.

    Só sobrescreve quando a ação manteria o trade vivo (HOLD/TIGHTEN_STOP).
    Se o LLM já está pedindo CLOSE ou inversão, deixa passar.
    """
    if position is None or settings.max_trade_age_hours <= 0:
        return action, reasoning
    max_age_min = settings.max_trade_age_hours * 60
    if position.age_minutes < max_age_min:
        return action, reasoning
    if action in ("HOLD", "TIGHTEN_STOP"):
        return "CLOSE", (
            f"[time exit: trade aberto há {position.age_minutes:.0f}min "
            f"(limite {max_age_min:.0f}min)] {reasoning}"
        )
    return action, reasoning


def _apply_circuit_breaker(
    action: str,
    reasoning: str,
    circuit: CircuitState,
) -> tuple[str, str]:
    """Bloqueia novas aberturas quando o circuit breaker dispara.

    CLOSE e TIGHTEN_STOP NUNCA são bloqueados - eles reduzem risco.
    HOLD passa direto. Apenas OPEN_LONG/OPEN_SHORT viram HOLD.
    """
    if not circuit.blocked:
        return action, reasoning
    if action in ("OPEN_LONG", "OPEN_SHORT"):
        return "HOLD", f"[circuit breaker: {circuit.reason}] {reasoning}"
    return action, reasoning


def build_signal(
    llm_response: dict,
    context: dict,
    position: OpenPosition | None,
) -> Signal:
    raw_action = str(llm_response.get("action", "HOLD")).upper().strip()
    reasoning = str(llm_response.get("reasoning", ""))

    raw_new_sl = llm_response.get("new_sl")
    try:
        new_sl_in = float(raw_new_sl) if raw_new_sl is not None else None
    except (TypeError, ValueError):
        new_sl_in = None

    try:
        confidence = float(llm_response.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    # 1. Coerência ação ↔ estado (v1.4)
    action, reasoning, new_sl = _validate_action(
        raw_action, new_sl_in, reasoning, position
    )

    # 2. Time-based exit (v1.6) - antes do circuit breaker, porque CLOSE
    #    forçado por idade não deve ser bloqueado por DD do dia.
    action, reasoning = _apply_time_exit(action, reasoning, position)

    # 3. Circuit breaker (v1.6) - equity lida fresh do MT5 (ou fallback).
    equity = get_account_equity()
    circuit = evaluate_circuit(equity)
    action, reasoning = _apply_circuit_breaker(action, reasoning, circuit)

    # 4. Filtro de confiança (mantido da v1.4) - ações que mexem em
    #    risco precisam de confiança mínima; HOLD passa sempre.
    actionable = action in ("OPEN_LONG", "OPEN_SHORT", "CLOSE", "TIGHTEN_STOP")
    if actionable and confidence < settings.min_confidence:
        action, reasoning = _downgrade(
            f"confiança insuficiente ({confidence:.0%})", reasoning
        )
        new_sl = None

    price = context.get("current_price") or 0.0
    vol = get_vol_estimate(context)

    entry_price: float | None = None
    sl_price: float | None = None
    tp_price: float | None = None
    lot = settings.min_lot

    if action in ("OPEN_LONG", "OPEN_SHORT") and price:
        entry_price = price
        sl_dist = vol.sigma * settings.atr_sl_multiplier
        tp_dist = vol.sigma * settings.atr_tp_multiplier
        if action == "OPEN_LONG":
            sl_price = round(price - sl_dist, 5)
            tp_price = round(price + tp_dist, 5)
        else:
            sl_price = round(price + sl_dist, 5)
            tp_price = round(price - tp_dist, 5)

        # 5. Position sizing dinâmico (v1.6) - substitui DEFAULT_LOT fixo.
        lot = compute_lot(equity, confidence, entry_price, sl_price)

    now = datetime.datetime.now(datetime.timezone.utc)
    return Signal(
        signal_id=now.strftime("%Y%m%dT%H%M%S%f"),
        symbol=context.get("symbol", settings.symbol),
        action=action,
        confidence=confidence,
        reasoning=reasoning,
        entry_price=entry_price,
        sl_price=sl_price,
        tp_price=tp_price,
        new_sl=new_sl,
        lot=lot,
        vol_method=vol.method,
        equity=round(equity, 2),
        circuit_state={
            "blocked": circuit.blocked,
            "daily_pnl": round(circuit.daily_pnl, 2),
            "consecutive_losses": circuit.consecutive_losses,
            "reason": circuit.reason,
        },
        created_at=now,
        expires_at=now + datetime.timedelta(seconds=settings.signal_ttl),
    )

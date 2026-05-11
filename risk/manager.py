import datetime
from dataclasses import dataclass

from config import HORIZON_PROFILES, HORIZONS, settings
from data.account import get_account_equity
from data.positions import OpenPosition
from risk.circuit_breaker import CircuitState, evaluate as evaluate_circuit
from risk.sizing import compute_lot
from risk.volatility import from_context as get_vol_estimate


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
    action: str
    intended_horizon: str | None    # scalp | intraday | swing | None
    position_id: int | None         # ticket da posição-alvo (CLOSE/TIGHTEN_STOP)
    confidence: float
    reasoning: str
    entry_price: float | None
    sl_price: float | None
    tp_price: float | None
    new_sl: float | None
    lot: float
    vol_method: str
    equity: float
    circuit_state: dict
    created_at: datetime.datetime
    expires_at: datetime.datetime

    def is_valid(self) -> bool:
        return datetime.datetime.now(datetime.timezone.utc) < self.expires_at

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "action": self.action,
            "intended_horizon": self.intended_horizon,
            "position_id": self.position_id,
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
    return "HOLD", f"[downgrade: {reason}] {original_reasoning}"


def _resolve_target_position(
    position_id: int | None, positions: list[OpenPosition]
) -> OpenPosition | None:
    """Resolve qual posição é alvo de CLOSE/TIGHTEN_STOP.

    Se position_id veio: retorna a que tem esse ticket (ou None se não existe).
    Se position_id é None e há só 1 aberta: assume essa.
    Se position_id é None e há múltiplas: ambíguo - retorna None (caller faz HOLD).
    """
    if position_id is not None:
        for p in positions:
            if p.ticket == position_id:
                return p
        return None
    if len(positions) == 1:
        return positions[0]
    return None


def _validate_action(
    action: str,
    intended_horizon: str | None,
    position_id: int | None,
    new_sl: float | None,
    reasoning: str,
    positions: list[OpenPosition],
) -> tuple[str, str | None, int | None, float | None, str]:
    """Guardrails state-aware (v1.4 + v1.7 multi-position).

    Retorna (action, intended_horizon, position_id, new_sl, reasoning) ajustados.
    """
    if action not in _VALID_ACTIONS:
        a, r = _downgrade(f"action inválida {action!r}", reasoning)
        return a, None, None, None, r

    # ---- OPEN_LONG / OPEN_SHORT ----
    if action in ("OPEN_LONG", "OPEN_SHORT"):
        if intended_horizon not in HORIZONS:
            a, r = _downgrade(
                f"OPEN sem intended_horizon válido (recebido: {intended_horizon!r})",
                reasoning,
            )
            return a, None, None, None, r

        if len(positions) >= settings.max_positions:
            a, r = _downgrade(
                f"MAX_POSITIONS atingido ({len(positions)}/{settings.max_positions})",
                reasoning,
            )
            return a, None, None, None, r

        target_side = "LONG" if action == "OPEN_LONG" else "SHORT"
        for p in positions:
            if p.side == target_side and p.intended_horizon == intended_horizon:
                a, r = _downgrade(
                    f"já há posição {target_side} de horizonte {intended_horizon} "
                    f"(ticket {p.ticket}) - regra de não-colisão",
                    reasoning,
                )
                return a, None, None, None, r

        return action, intended_horizon, None, None, reasoning

    # ---- CLOSE ----
    if action == "CLOSE":
        if not positions:
            a, r = _downgrade("CLOSE sem posições abertas", reasoning)
            return a, None, None, None, r
        target = _resolve_target_position(position_id, positions)
        if target is None:
            if position_id is None:
                a, r = _downgrade(
                    f"CLOSE ambíguo: {len(positions)} posições abertas, "
                    "position_id obrigatório",
                    reasoning,
                )
            else:
                a, r = _downgrade(
                    f"CLOSE com position_id {position_id} inexistente",
                    reasoning,
                )
            return a, None, None, None, r
        return action, None, target.ticket, None, reasoning

    # ---- TIGHTEN_STOP ----
    if action == "TIGHTEN_STOP":
        if not positions:
            a, r = _downgrade("TIGHTEN_STOP sem posições abertas", reasoning)
            return a, None, None, None, r
        target = _resolve_target_position(position_id, positions)
        if target is None:
            if position_id is None:
                a, r = _downgrade(
                    f"TIGHTEN_STOP ambíguo: {len(positions)} posições abertas, "
                    "position_id obrigatório",
                    reasoning,
                )
            else:
                a, r = _downgrade(
                    f"TIGHTEN_STOP com position_id {position_id} inexistente",
                    reasoning,
                )
            return a, None, None, None, r
        if new_sl is None:
            a, r = _downgrade("TIGHTEN_STOP sem new_sl", reasoning)
            return a, None, None, None, r
        if target.side == "LONG" and new_sl <= target.sl_price:
            a, r = _downgrade(
                f"SL não aperta (atual={target.sl_price}, proposto={new_sl})",
                reasoning,
            )
            return a, None, None, None, r
        if target.side == "SHORT" and new_sl >= target.sl_price:
            a, r = _downgrade(
                f"SL não aperta (atual={target.sl_price}, proposto={new_sl})",
                reasoning,
            )
            return a, None, None, None, r
        margin = settings.pip_size
        if target.side == "LONG" and new_sl >= target.current_price - margin:
            a, r = _downgrade(
                f"SL ultrapassa preço atual ({target.current_price})",
                reasoning,
            )
            return a, None, None, None, r
        if target.side == "SHORT" and new_sl <= target.current_price + margin:
            a, r = _downgrade(
                f"SL ultrapassa preço atual ({target.current_price})",
                reasoning,
            )
            return a, None, None, None, r

        # Guardrails anti-aperto-prematuro (v1.8.1):
        # (A) progresso mínimo até o TP antes de permitir apertar
        # (B) margem mínima entre novo SL e preço atual em relação ao lucro
        if target.side == "LONG":
            tp_dist = target.tp_price - target.entry_price
            current_gain = target.current_price - target.entry_price
            sl_from_current = target.current_price - new_sl
        else:  # SHORT
            tp_dist = target.entry_price - target.tp_price
            current_gain = target.entry_price - target.current_price
            sl_from_current = new_sl - target.current_price

        if tp_dist > 0 and current_gain > 0:
            progress = current_gain / tp_dist
            if progress < settings.min_tighten_progress:
                a, r = _downgrade(
                    f"TIGHTEN precoce: progresso {progress:.0%} < mínimo "
                    f"{settings.min_tighten_progress:.0%} do caminho ao TP",
                    reasoning,
                )
                return a, None, None, None, r

            trail_buffer = sl_from_current / current_gain
            if trail_buffer < settings.min_trail_buffer:
                a, r = _downgrade(
                    f"SL muito colado: buffer {trail_buffer:.0%} < mínimo "
                    f"{settings.min_trail_buffer:.0%} do lucro acumulado",
                    reasoning,
                )
                return a, None, None, None, r

        return action, None, target.ticket, round(new_sl, 5), reasoning

    # HOLD
    return action, None, None, None, reasoning


def _apply_time_exit(
    action: str,
    position_id: int | None,
    reasoning: str,
    positions: list[OpenPosition],
) -> tuple[str, int | None, str]:
    """Força CLOSE em qualquer posição que excedeu seu limite de idade
    (per-horizon: scalp 4h, intraday 24h, swing 14d).

    Tem precedência sobre HOLD/TIGHTEN_STOP do LLM. Se o LLM já está pedindo
    CLOSE ou inversão, deixa passar.
    """
    if action in ("OPEN_LONG", "OPEN_SHORT", "CLOSE"):
        return action, position_id, reasoning

    # Procura primeira posição expirada por idade.
    for p in positions:
        if p.age_minutes >= p.max_age_minutes:
            return "CLOSE", p.ticket, (
                f"[time exit: {p.intended_horizon} aberta há "
                f"{p.age_minutes:.0f}min (limite {p.max_age_minutes:.0f}min, "
                f"ticket {p.ticket})] {reasoning}"
            )
    return action, position_id, reasoning


def _apply_circuit_breaker(
    action: str,
    reasoning: str,
    circuit: CircuitState,
) -> tuple[str, str]:
    if not circuit.blocked:
        return action, reasoning
    if action in ("OPEN_LONG", "OPEN_SHORT"):
        return "HOLD", f"[circuit breaker: {circuit.reason}] {reasoning}"
    return action, reasoning


def build_signal(
    llm_response: dict,
    context: dict,
    positions: list[OpenPosition],
) -> Signal:
    raw_action = str(llm_response.get("action", "HOLD")).upper().strip()
    reasoning = str(llm_response.get("reasoning", ""))

    raw_horizon = llm_response.get("intended_horizon")
    intended_horizon_in = raw_horizon if raw_horizon in HORIZONS else None

    raw_pos_id = llm_response.get("position_id")
    try:
        position_id_in = int(raw_pos_id) if raw_pos_id is not None else None
    except (TypeError, ValueError):
        position_id_in = None

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

    # 1. Coerência ação ↔ estado (v1.4 + v1.7 multi-position)
    action, intended_horizon, position_id, new_sl, reasoning = _validate_action(
        raw_action, intended_horizon_in, position_id_in, new_sl_in,
        reasoning, positions,
    )

    # 2. Time-based exit per-horizon (v1.7) - antes do circuit breaker.
    action, position_id, reasoning = _apply_time_exit(
        action, position_id, reasoning, positions
    )

    # 3. Circuit breaker (v1.6).
    equity = get_account_equity()
    circuit = evaluate_circuit(equity)
    action, reasoning = _apply_circuit_breaker(action, reasoning, circuit)

    # 4. Filtro de confiança (v1.4).
    actionable = action in ("OPEN_LONG", "OPEN_SHORT", "CLOSE", "TIGHTEN_STOP")
    if actionable and confidence < settings.min_confidence:
        action, reasoning = _downgrade(
            f"confiança insuficiente ({confidence:.0%})", reasoning
        )
        new_sl = None
        intended_horizon = None
        position_id = None

    # 5. Compute SL/TP and lot for opens, scaled by horizon.
    price = context.get("current_price") or 0.0
    vol = get_vol_estimate(context)

    entry_price: float | None = None
    sl_price: float | None = None
    tp_price: float | None = None
    lot = settings.min_lot

    if action in ("OPEN_LONG", "OPEN_SHORT") and price and intended_horizon:
        profile = HORIZON_PROFILES[intended_horizon]
        entry_price = price
        sl_dist = vol.sigma * profile["sl_mult"]
        tp_dist = vol.sigma * profile["tp_mult"]
        if action == "OPEN_LONG":
            sl_price = round(price - sl_dist, 5)
            tp_price = round(price + tp_dist, 5)
        else:
            sl_price = round(price + sl_dist, 5)
            tp_price = round(price - tp_dist, 5)
        lot = compute_lot(equity, confidence, entry_price, sl_price)

    now = datetime.datetime.now(datetime.timezone.utc)
    return Signal(
        signal_id=now.strftime("%Y%m%dT%H%M%S%f"),
        symbol=context.get("symbol", settings.symbol),
        action=action,
        intended_horizon=intended_horizon,
        position_id=position_id,
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

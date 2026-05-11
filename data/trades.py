"""
Captura do lifecycle real de trades fechados.

Lê o histórico de deals do MT5 (`history_deals_get`), reconstrói trades
completos (par OUT-IN) e persiste em `logs/trades.jsonl` com link ao
`signal_id` que originou a abertura - viabilizando análise de outcomes
real (TP, SL, CLOSE manual, REVERSE) sem depender da simulação post-hoc.

Cada entrada do log tem o ciclo completo:
  - signal_id_open  : sinal que abriu o trade (match temporal)
  - opened_at       : timestamp da abertura
  - closed_at       : timestamp do fechamento
  - close_reason    : "tp" | "sl" | "manual" | "reverse" | "unknown"
  - pnl_money       : P&L em moeda da conta (já com spread/swap/comissão)
  - pnl_pips        : P&L em pips
  - duration_min    : duração total
"""

import datetime
import json
from pathlib import Path

from config import settings
from data.mt5_source import get_mt5_client


_LOG_FILE = Path("logs") / "trades.jsonl"
_STATE_FILE = Path("logs") / "trades_state.json"


def _read_state() -> dict:
    if not _STATE_FILE.exists():
        return {"last_deal_time": 0, "captured_position_ids": []}
    try:
        return json.loads(_STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {"last_deal_time": 0, "captured_position_ids": []}


def _write_state(state: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state))


def _read_signals_log() -> list[dict]:
    log = Path("logs") / "signals.jsonl"
    if not log.exists():
        return []
    out = []
    with log.open("r", encoding="utf-8") as f:
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
    """Casa um trade aberto com o sinal que provavelmente o originou.

    Heurística: sinal mais recente cujo `action` bate com o lado do trade
    (OPEN_LONG↔LONG, OPEN_SHORT↔SHORT) e cujo `created_at` é anterior à
    abertura por no máximo 2 × analysis_interval (margem para latência
    de execução + slippage entre poll do EA e fill do broker).
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


def _signals_for_position(
    position_id: int,
    opened_at: datetime.datetime,
    closed_at: datetime.datetime,
    action_filter: str,
) -> list[dict]:
    """Sinais do tipo `action_filter` que tinham position_id alvo dentro
    do ciclo de vida do trade."""
    out = []
    for s in _read_signals_log():
        if s.get("action") != action_filter:
            continue
        if s.get("position_id") != position_id:
            continue
        try:
            ts = datetime.datetime.fromisoformat(s["created_at"])
        except (KeyError, ValueError):
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=datetime.timezone.utc)
        if ts < opened_at or ts > closed_at:
            continue
        out.append((ts, s))
    return [s for _, s in sorted(out, key=lambda t: t[0])]


def _classify_close(
    position_close_price: float,
    sl_open: float,
    tp_open: float,
    side: str,
    position_id: int,
    opened_at: datetime.datetime,
    closed_at: datetime.datetime,
) -> str:
    """Infere motivo do fechamento cruzando preço de saída com SL/TP originais,
    com sinais de TIGHTEN_STOP/CLOSE emitidos durante a vida do trade.

    Taxonomia:
      tp            - bateu o TP original
      sl            - bateu o SL original (não houve aperto)
      sl_tightened  - bateu um SL apertado por TIGHTEN_STOP
      close         - fechado por ação CLOSE explícita do agente
      manual        - fechado fora do controle do agente (intervenção humana
                      no terminal, desconexão do broker, etc)
    """
    tol = settings.pip_size * 2  # 2 pips de tolerância (slippage)

    if sl_open and abs(position_close_price - sl_open) <= tol:
        return "sl"
    if tp_open and abs(position_close_price - tp_open) <= tol:
        return "tp"

    # Sinal de CLOSE explícito durante o trade → fechamento discricionário
    if position_id and _signals_for_position(
        position_id, opened_at, closed_at, "CLOSE"
    ):
        return "close"

    # Algum TIGHTEN_STOP aceito? Usa o SL apertado mais recente como referência.
    tightens = (
        _signals_for_position(position_id, opened_at, closed_at, "TIGHTEN_STOP")
        if position_id else []
    )
    last_new_sl = next(
        (s.get("new_sl") for s in reversed(tightens) if s.get("new_sl")),
        None,
    )
    if last_new_sl and abs(position_close_price - float(last_new_sl)) <= tol:
        return "sl_tightened"

    return "manual"


def capture_closed_trades() -> int:
    """Varre deals novos desde a última captura e persiste trades fechados.

    Retorna número de trades novos persistidos.
    """
    if settings.data_source != "mt5":
        return 0

    mt5 = get_mt5_client()
    if mt5 is None or not mt5.initialize():
        return 0

    try:
        state = _read_state()
        # Janela: desde o último deal capturado, ou últimas 7 dias na primeira run.
        last_seen = state.get("last_deal_time", 0)
        if last_seen == 0:
            since = datetime.datetime.now() - datetime.timedelta(days=7)
        else:
            since = datetime.datetime.fromtimestamp(last_seen)

        deals = mt5.history_deals_get(since, datetime.datetime.now())
        if not deals:
            return 0

        # Agrupa deals por position_id - abertura (entry) + fechamento (out).
        by_pos: dict[int, list] = {}
        for d in deals:
            by_pos.setdefault(int(d.position_id), []).append(d)

        captured = set(state.get("captured_position_ids", []))
        new_entries: list[dict] = []
        max_time = last_seen

        for pos_id, ds in by_pos.items():
            if pos_id in captured:
                continue
            ds_sorted = sorted(ds, key=lambda d: d.time)
            entry = next((d for d in ds_sorted if d.entry == mt5.DEAL_ENTRY_IN), None)
            exit_d = next((d for d in ds_sorted if d.entry == mt5.DEAL_ENTRY_OUT), None)
            if not entry or not exit_d:
                # Trade ainda aberto - pula, captura quando fechar.
                continue

            side = "LONG" if entry.type == mt5.DEAL_TYPE_BUY else "SHORT"
            opened_at = datetime.datetime.fromtimestamp(
                entry.time, tz=datetime.timezone.utc
            )
            closed_at = datetime.datetime.fromtimestamp(
                exit_d.time, tz=datetime.timezone.utc
            )

            sig = _match_open_signal(opened_at, side)
            sig_open_id = sig.get("signal_id") if sig else None
            sl_open = sig.get("sl_price") if sig else None
            tp_open = sig.get("tp_price") if sig else None

            close_reason = _classify_close(
                position_close_price=float(exit_d.price),
                sl_open=sl_open or 0,
                tp_open=tp_open or 0,
                side=side,
                position_id=pos_id,
                opened_at=opened_at,
                closed_at=closed_at,
            )

            entry_price = float(entry.price)
            exit_price = float(exit_d.price)
            if side == "LONG":
                pnl_pips = (exit_price - entry_price) / settings.pip_size
            else:
                pnl_pips = (entry_price - exit_price) / settings.pip_size

            new_entries.append({
                "position_id": pos_id,
                "symbol": entry.symbol,
                "side": side,
                "lot": float(entry.volume),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "opened_at": opened_at.isoformat(),
                "closed_at": closed_at.isoformat(),
                "duration_min": round(
                    (closed_at - opened_at).total_seconds() / 60, 1
                ),
                "close_reason": close_reason,
                "pnl_money": float(exit_d.profit) + float(getattr(exit_d, "commission", 0)) + float(getattr(exit_d, "swap", 0)),
                "pnl_pips": round(pnl_pips, 1),
                "signal_id_open": sig_open_id,
                "open_reasoning": (sig or {}).get("reasoning"),
                "model_version": (sig or {}).get("model_version"),
            })
            captured.add(pos_id)
            max_time = max(max_time, int(exit_d.time))

        if new_entries:
            _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with _LOG_FILE.open("a", encoding="utf-8") as f:
                for e in new_entries:
                    f.write(json.dumps(e) + "\n")

        # Bound do crescimento do estado: guarda apenas IDs recentes.
        state["last_deal_time"] = max_time
        state["captured_position_ids"] = list(captured)[-1000:]
        _write_state(state)

        return len(new_entries)
    finally:
        mt5.shutdown()


def find_open_signal(opened_at: datetime.datetime, side: str) -> dict | None:
    """Wrapper público pra busca de sinal de abertura - usado pra injetar
    a razão original no prompt quando há posição viva.
    """
    return _match_open_signal(opened_at, side)

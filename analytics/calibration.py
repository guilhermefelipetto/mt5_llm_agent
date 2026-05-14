"""
Self-calibration (v1.8): o agente vê seu próprio histórico de desempenho
condicionado ao contexto atual. Junta signals.jsonl ↔ trades.jsonl via
signal_id_open e agrega por (regime_hurst, horizonte, lado).

Princípio: o LLM passa a saber, antes de decidir, "em contextos similares
nas últimas N decisões fechadas, eu acertei X% e meu P&L médio foi Y pips".
Isso fecha o loop de aprendizado sem fine-tuning — é metaprompting com
métricas reais.

Limitação importante: amostras pequenas (n<3) não geram win rate; são
mostradas só como contagem. O prompt instrui o LLM a tratar números com
amostra pequena como ruído, não regra.
"""

import datetime
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from config import HORIZONS


_SIGNALS = Path("logs") / "signals.jsonl"
_TRADES = Path("logs") / "trades.jsonl"

# Mínimo de trades num bucket pra calcular win rate. Abaixo disso é ruído.
_MIN_SAMPLES = 3


@dataclass
class CalibStats:
    n_trades: int
    n_wins: int
    win_rate_pct: float | None     # None quando n < MIN_SAMPLES
    avg_pips: float
    total_pips: float
    avg_pnl_money: float
    profit_factor: float | None    # None quando não há perdas

    def to_dict(self) -> dict:
        return {
            "n_trades": self.n_trades,
            "n_wins": self.n_wins,
            "win_rate_pct": self.win_rate_pct,
            "avg_pips": self.avg_pips,
            "total_pips": self.total_pips,
            "avg_pnl_money": self.avg_pnl_money,
            "profit_factor": self.profit_factor,
        }


def _compute(items: list[dict]) -> CalibStats:
    n = len(items)
    pips = [e["pnl_pips"] for e in items]
    monies = [e["pnl_money"] for e in items]
    wins = sum(1 for p in pips if p > 0)
    gains = sum(p for p in pips if p > 0)
    losses = sum(-p for p in pips if p < 0)

    return CalibStats(
        n_trades=n,
        n_wins=wins,
        win_rate_pct=round(wins / n * 100, 1) if n >= _MIN_SAMPLES else None,
        avg_pips=round(sum(pips) / n, 2) if n else 0.0,
        total_pips=round(sum(pips), 1),
        avg_pnl_money=round(sum(monies) / n, 2) if n else 0.0,
        profit_factor=round(gains / losses, 2) if losses > 0 else None,
    )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _signal_regime(sig: dict) -> str | None:
    """Hurst regime do 1h no momento da decisão de abertura."""
    ctx = sig.get("context_summary") or {}
    tfs = ctx.get("timeframes") or {}
    h1 = tfs.get("1h") or {}
    stats = h1.get("statistics") or {}
    return (stats.get("hurst") or {}).get("regime")


def _signal_session(sig: dict) -> str | None:
    """Sessão dominante (primeira da lista) no momento da decisão."""
    ctx = sig.get("context_summary") or {}
    sessions = ctx.get("sessions") or []
    return sessions[0] if sessions else None


def build_calibration(lookback_days: int = 30) -> dict:
    """Constrói calibração a partir dos logs.

    Retorna dict com agregações em três níveis de especificidade:
      - by_full_key: (regime, horizonte, lado) — mais específico
      - by_horizon_side: (horizonte, lado) — agrega regimes
      - by_regime_side: (regime, lado) — agrega horizontes
      - overall: tudo junto

    Trades sem signal de origem casado (manuais ou abertos por agente
    pré-v1.7 sem horizonte) são descartados — não dá pra atribuir contexto.
    """
    signals = _read_jsonl(_SIGNALS)
    trades = _read_jsonl(_TRADES)

    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=lookback_days
    )
    sig_by_id = {s["signal_id"]: s for s in signals if s.get("signal_id")}

    enriched: list[dict] = []
    for t in trades:
        try:
            closed = datetime.datetime.fromisoformat(t["closed_at"])
        except (KeyError, ValueError):
            continue
        if closed.tzinfo is None:
            closed = closed.replace(tzinfo=datetime.timezone.utc)
        if closed < cutoff:
            continue

        sig = sig_by_id.get(t.get("signal_id_open"))
        if not sig:
            continue
        regime = _signal_regime(sig)
        horizon = sig.get("intended_horizon")
        if not regime or horizon not in HORIZONS:
            continue

        enriched.append({
            "regime": regime,
            "horizon": horizon,
            "side": t.get("side"),
            "pnl_money": float(t.get("pnl_money") or 0),
            "pnl_pips": float(t.get("pnl_pips") or 0),
        })

    by_full: dict[tuple, list] = defaultdict(list)
    by_hs: dict[tuple, list] = defaultdict(list)
    by_rs: dict[tuple, list] = defaultdict(list)
    for e in enriched:
        by_full[(e["regime"], e["horizon"], e["side"])].append(e)
        by_hs[(e["horizon"], e["side"])].append(e)
        by_rs[(e["regime"], e["side"])].append(e)

    return {
        "by_full_key": {k: _compute(v) for k, v in by_full.items()},
        "by_horizon_side": {k: _compute(v) for k, v in by_hs.items()},
        "by_regime_side": {k: _compute(v) for k, v in by_rs.items()},
        "overall": _compute(enriched) if enriched else None,
        "n_trades_total": len(enriched),
        "lookback_days": lookback_days,
        "n_trades_unmatched": len(trades) - len(enriched),
    }


def _fmt_stats(stats: CalibStats) -> str:
    """Formata uma estatística numa linha curta para o prompt."""
    wr = f"{stats.win_rate_pct}%" if stats.win_rate_pct is not None else "n<3"
    pf_part = f", PF {stats.profit_factor}" if stats.profit_factor is not None else ""
    return (
        f"{stats.n_trades} trades, {wr} wins, "
        f"{stats.avg_pips:+.1f} pips médio (Σ {stats.total_pips:+.1f}){pf_part}"
    )


def format_for_prompt(calibration: dict, current_regime: str | None) -> list[str]:
    """Renderiza a seção [SEU TRACK RECORD] para o prompt do LLM.

    Estratégia: foca no regime atual primeiro (mais relevante), depois mostra
    agregados por (horizonte, lado) ignorando regime, pra dar visão geral.
    """
    n = calibration["n_trades_total"]
    if n == 0:
        return [
            "[SEU TRACK RECORD] Sem trades fechados com contexto associado ainda. "
            "Decida sem viés histórico — está em fase de coleta.",
            "",
        ]

    lines = [
        f"[SEU TRACK RECORD — últimos {calibration['lookback_days']}d, "
        f"{n} trades com contexto associado]",
    ]

    # Bloco 1: foco no regime atual (sinal mais útil pra decidir agora)
    if current_regime:
        matching = {
            (h, s): stats
            for (r, h, s), stats in calibration["by_full_key"].items()
            if r == current_regime
        }
        if matching:
            lines.append(f"  Em regime '{current_regime}' (atual):")
            for (h, side), stats in sorted(matching.items()):
                lines.append(f"    {h} {side}: {_fmt_stats(stats)}")
        else:
            lines.append(
                f"  Em regime '{current_regime}' (atual): sem trades passados — "
                "explore com cautela."
            )

    # Bloco 2: agregado por (horizonte, lado), todos regimes
    if calibration["by_horizon_side"]:
        lines.append("  Agregado por (horizonte, lado), todos regimes:")
        for (h, side), stats in sorted(calibration["by_horizon_side"].items()):
            lines.append(f"    {h} {side}: {_fmt_stats(stats)}")

    # Bloco 3: aviso operacional pro modelo interpretar com sabedoria
    lines.append(
        "  ATENÇÃO: amostras < 10 são ruído estatístico. Win rate < 40% num "
        "bucket = considere inverter ou HOLD."
    )
    lines.append("")
    return lines


def posterior_win_rate(
    calibration: dict,
    regime: str | None,
    horizon: str,
    side: str,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[float, int, str]:
    """Posterior do win rate sob prior Beta(alpha, beta) conjugado a Bernoulli.

    Para w vitórias em n trades:  E[p | dados] = (alpha + w) / (alpha + beta + n).
    Prior default Beta(1,1) é a Regra de Sucessão de Laplace - não informativo,
    puxa o estimador pra 0.5 quando n é pequeno (efeito de regressão à média).

    Faz cascata de fallback no bucket: (regime, horizonte, lado) -> (horizonte,
    lado) -> overall -> só-prior. Retorna (p_posterior, n_efetivo, bucket_usado).
    """
    if regime is not None:
        stats = calibration["by_full_key"].get((regime, horizon, side))
        if stats and stats.n_trades > 0:
            p = (alpha + stats.n_wins) / (alpha + beta + stats.n_trades)
            return p, stats.n_trades, "full"

    stats = calibration["by_horizon_side"].get((horizon, side))
    if stats and stats.n_trades > 0:
        p = (alpha + stats.n_wins) / (alpha + beta + stats.n_trades)
        return p, stats.n_trades, "horizon_side"

    overall = calibration.get("overall")
    if overall and overall.n_trades > 0:
        p = (alpha + overall.n_wins) / (alpha + beta + overall.n_trades)
        return p, overall.n_trades, "overall"

    return alpha / (alpha + beta), 0, "prior_only"


def calibration_to_json(calibration: dict) -> dict:
    """Serializa o calibration dict pra JSON (chaves de tupla viram strings).

    Usado pelo dashboard pra renderizar a mesma informação que o LLM vê.
    """
    def _key_to_str(k):
        return " | ".join(str(x) for x in k)

    return {
        "by_full_key": {_key_to_str(k): v.to_dict() for k, v in calibration["by_full_key"].items()},
        "by_horizon_side": {_key_to_str(k): v.to_dict() for k, v in calibration["by_horizon_side"].items()},
        "by_regime_side": {_key_to_str(k): v.to_dict() for k, v in calibration["by_regime_side"].items()},
        "overall": calibration["overall"].to_dict() if calibration["overall"] else None,
        "n_trades_total": calibration["n_trades_total"],
        "n_trades_unmatched": calibration["n_trades_unmatched"],
        "lookback_days": calibration["lookback_days"],
    }

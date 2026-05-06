"""
Agregação de métricas a partir de sinais com outcomes simulados.
"""

import statistics
from collections import Counter

from analytics.simulator import _signal_side


def aggregate(signals_with_outcomes: list[dict]) -> dict:
    """
    Recebe uma lista de dicts onde cada um tem os campos do sinal logado +
    um campo `outcome` produzido pelo simulator.
    """
    n = len(signals_with_outcomes)
    if n == 0:
        return {"total_signals": 0}

    actions = Counter(s["action"] for s in signals_with_outcomes)

    trades = [s for s in signals_with_outcomes if _signal_side(s) is not None]
    resolved = [
        s for s in trades
        if s["outcome"]["resolution"] in ("tp_hit", "sl_hit")
    ]

    tp_hits = [s for s in resolved if s["outcome"]["resolution"] == "tp_hit"]
    sl_hits = [s for s in resolved if s["outcome"]["resolution"] == "sl_hit"]
    expired = [s for s in trades if s["outcome"]["resolution"] == "expired"]
    no_data = [s for s in trades if s["outcome"]["resolution"] == "no_data"]

    metrics: dict = {
        "total_signals": n,
        "action_distribution": dict(actions),
        "trades_attempted": len(trades),
        "trades_resolved": len(resolved),
        "tp_hits": len(tp_hits),
        "sl_hits": len(sl_hits),
        "expired": len(expired),
        "no_data": len(no_data),
    }

    if not resolved:
        return metrics

    pnl = [s["outcome"]["pnl_pips"] for s in resolved]
    wins = [p for p in pnl if p > 0]
    losses = [abs(p) for p in pnl if p < 0]

    metrics["win_rate_pct"] = round(len(tp_hits) / len(resolved) * 100, 1)
    metrics["total_pips"] = round(sum(pnl), 1)
    metrics["avg_pips"] = round(statistics.mean(pnl), 2)
    metrics["best_trade_pips"] = round(max(pnl), 1)
    metrics["worst_trade_pips"] = round(min(pnl), 1)

    if losses:
        metrics["profit_factor"] = round(sum(wins) / sum(losses), 2)
    elif wins:
        metrics["profit_factor"] = float("inf")

    if len(pnl) > 1:
        sd = statistics.stdev(pnl)
        metrics["pnl_std_pips"] = round(sd, 2)
        if sd > 0:
            metrics["sharpe_per_trade"] = round(statistics.mean(pnl) / sd, 3)

    durations = [s["outcome"].get("duration_seconds", 0) for s in resolved]
    if durations:
        metrics["avg_duration_min"] = round(statistics.mean(durations) / 60, 1)

    return metrics


def aggregate_latency(signals: list[dict]) -> dict | None:
    """Métricas de latência do LLM (úteis para comparar provedores)."""
    latencies = [s.get("llm_latency_ms") for s in signals if s.get("llm_latency_ms")]
    if not latencies:
        return None
    return {
        "avg_ms": round(statistics.mean(latencies), 0),
        "median_ms": round(statistics.median(latencies), 0),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 0)
                  if len(latencies) >= 20 else None,
        "max_ms": max(latencies),
        "n": len(latencies),
    }

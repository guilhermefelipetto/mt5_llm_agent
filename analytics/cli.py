"""
CLI para análise dos logs de sinais. Roda a simulação de outcomes e imprime
um relatório agregado por `model_version` (e opcionalmente por provider).

Uso:
    python -m analytics.cli
    python -m analytics.cli --version v1.4.0
    python -m analytics.cli --by provider          # agrupa por provider+model
    python -m analytics.cli --max-hours 12         # horizonte de resolução
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from analytics.metrics import aggregate, aggregate_latency
from analytics.simulator import fetch_market_data, simulate_outcome


def load_signals(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    out = []
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _print_section(title: str, metrics: dict, latency: dict | None):
    print(f"\n┌─ {title} " + "─" * max(0, 50 - len(title)))
    print(f"│ Total de sinais: {metrics['total_signals']}")
    print(f"│ Ações:           {metrics.get('action_distribution', {})}")

    attempted = metrics.get("trades_attempted", 0)
    if attempted:
        print(f"│")
        print(f"│ Trades simulados: {metrics['trades_resolved']}/{attempted} "
              f"(no_data={metrics.get('no_data', 0)}, expired={metrics.get('expired', 0)})")
        print(f"│   TP hits: {metrics.get('tp_hits', 0)}")
        print(f"│   SL hits: {metrics.get('sl_hits', 0)}")

    if "win_rate_pct" in metrics:
        print(f"│")
        print(f"│ Win rate:        {metrics['win_rate_pct']}%")
        print(f"│ Total pips:      {metrics['total_pips']:+.1f}")
        print(f"│ Pips médios:     {metrics['avg_pips']:+.2f}")
        print(f"│ Profit factor:   {metrics.get('profit_factor', 'n/a')}")
        print(f"│ Melhor trade:    {metrics['best_trade_pips']:+.1f} pips")
        print(f"│ Pior trade:      {metrics['worst_trade_pips']:+.1f} pips")
        if "sharpe_per_trade" in metrics:
            print(f"│ Sharpe/trade:    {metrics['sharpe_per_trade']}")
        if "avg_duration_min" in metrics:
            print(f"│ Duração média:   {metrics['avg_duration_min']} min")

    if latency:
        print(f"│")
        print(f"│ Latência LLM:    avg={latency['avg_ms']}ms  med={latency['median_ms']}ms  "
              f"max={latency['max_ms']}ms  (n={latency['n']})")
    print("└" + "─" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Análise de sinais logados - simula outcomes e agrega métricas."
    )
    parser.add_argument("--log", default="logs/signals.jsonl",
                        help="Caminho do arquivo .jsonl")
    parser.add_argument("--version", help="Filtra por uma model_version específica")
    parser.add_argument("--by", choices=["version", "provider"], default="version",
                        help="Critério de agrupamento (default: version)")
    parser.add_argument("--max-hours", type=int, default=24,
                        help="Horizonte máximo (em horas) para resolver um trade")
    parser.add_argument("--symbol", default="EURUSD=X",
                        help="Símbolo yfinance para buscar dados de simulação")
    args = parser.parse_args()

    signals = load_signals(Path(args.log))
    if not signals:
        print(f"[!] Nenhum sinal encontrado em {args.log}")
        return

    if args.version:
        signals = [s for s in signals if s.get("model_version") == args.version]
        if not signals:
            print(f"[!] Nenhum sinal para a versão '{args.version}'.")
            return

    print(f"[*] {len(signals)} sinais carregados de {args.log}")
    print(f"[*] Buscando dados de mercado ({args.symbol}, 1m, 7d)...")
    market_data = fetch_market_data(args.symbol)
    if market_data.empty:
        print("[!] Não foi possível obter dados de mercado para simulação.")
        return

    print(f"[*] Simulando outcomes (horizonte: {args.max_hours}h)...")
    for s in signals:
        s["outcome"] = simulate_outcome(s, market_data, max_hours=args.max_hours)

    # Agrupamento
    if args.by == "provider":
        def key(s):
            return f"{s.get('provider', '?')}/{s.get('model', '?')} ({s.get('model_version', '?')})"
    else:
        def key(s):
            return s.get("model_version", "unknown")

    groups: dict[str, list[dict]] = defaultdict(list)
    for s in signals:
        groups[key(s)].append(s)

    for group_name in sorted(groups.keys()):
        group_signals = groups[group_name]
        metrics = aggregate(group_signals)
        latency = aggregate_latency(group_signals)
        _print_section(group_name, metrics, latency)
    print()


if __name__ == "__main__":
    main()

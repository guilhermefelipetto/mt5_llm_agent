#!/usr/bin/env python3
"""
Zera os logs de runtime do agente - útil para começar uma nova janela
de coleta limpa (ex: depois de resetar o equity da conta demo).

Uso:
    python reset_logs.py           # interativo, pede confirmação digitando "SIM"
    python reset_logs.py --yes     # pula confirmação (scripts/CI)

Apaga:
    logs/signals.jsonl       - decisões do LLM
    logs/trades.jsonl        - ciclo de vida de trades fechados
    logs/trades_state.json   - estado da última varredura de history_deals

NÃO toca em:
    .env, código, configurações, paper/, ROADMAP.md, posições no MT5.
"""

import argparse
import sys
from pathlib import Path


_LOG_DIR = Path(__file__).parent / "logs"
_TARGETS = [
    "signals.jsonl",
    "trades.jsonl",
    "trades_state.json",
]


def _human(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.1f} KB"
    return f"{n_bytes / 1024 ** 2:.1f} MB"


def _line_count(path: Path) -> int | None:
    """Conta linhas só pra .jsonl - JSON não-linear não tem 'linhas' significativas."""
    if path.suffix != ".jsonl":
        return None
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _existing_targets() -> list[Path]:
    return [_LOG_DIR / name for name in _TARGETS if (_LOG_DIR / name).exists()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Zera signals.jsonl, trades.jsonl e trades_state.json."
    )
    parser.add_argument(
        "--yes", "-y", action="store_true",
        help="Pula a confirmação interativa.",
    )
    args = parser.parse_args()

    existing = _existing_targets()
    if not existing:
        print("Nada a fazer - logs já estão zerados.")
        return 0

    print("Os seguintes arquivos serão APAGADOS:\n")
    total = 0
    for p in existing:
        size = p.stat().st_size
        total += size
        n = _line_count(p)
        suffix = f" - {n} linhas" if n is not None else ""
        print(f"  {p}  ({_human(size)}){suffix}")
    print(f"\nTotal: {_human(total)} em {len(existing)} arquivo(s).\n")

    if not args.yes:
        # 'SIM' explícito (não Y/n) para evitar muscle memory de aceitar tudo.
        resp = input("Confirma? digite SIM para apagar: ").strip()
        if resp != "SIM":
            print("Cancelado - nada foi apagado.")
            return 1

    for p in existing:
        p.unlink()
        print(f"  apagado: {p.name}")

    print(
        "\nLogs zerados. Reinicie o uvicorn para limpar o `current_signal` "
        "em memória do servidor."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

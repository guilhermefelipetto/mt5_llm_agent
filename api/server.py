import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from agent.llm_manager import LLMManager
from agent.prompt_builder import build_prompt
from api.dashboard import router as dashboard_router
from config import settings
from data.fetcher import get_ohlcv
from data.positions import OpenPosition, get_open_position
from data.trades import capture_closed_trades, find_open_signal
from features.context_builder import build_context
from risk.manager import Signal, build_signal


def _build_pipeline(
    ohlcv: dict,
    position: OpenPosition | None,
    original_reasoning: str | None,
) -> tuple[dict, str, str]:
    """Pipeline CPU-bound (statsmodels) + montagem de prompt - fica no executor."""
    ctx = build_context(ohlcv, settings.symbol)
    sys_p, usr_p = build_prompt(ctx, position, original_reasoning)
    return ctx, sys_p, usr_p


_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "signals.jsonl"

llm = LLMManager()
current_signal: Signal | None = None


def _log(
    signal: Signal,
    context: dict,
    position: OpenPosition | None,
    llm_latency_ms: int,
):
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "model_version": settings.model_version,
        "provider": llm.provider,
        "model": llm.model,
        "llm_latency_ms": llm_latency_ms,
        "yf_symbol": settings.yf_symbol,
        "position_at_decision": position.to_dict() if position else None,
        **signal.to_dict(),
        "context_summary": {
            "sessions": context.get("sessions"),
            "volatility_regime": context.get("volatility_regime"),
            "vol_forecast": context.get("vol_forecast"),
            "timeframes": {
                tf: {
                    k: v for k, v in data.items()
                    if k in ("rsi", "macd_signal", "trend", "atr_pips", "statistics")
                }
                for tf, data in context.get("timeframes", {}).items()
            },
        },
    }
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _next_interval(position: OpenPosition | None) -> int:
    """Intervalo adaptativo: revisita posição aberta mais rápido pra
    poder reagir com CLOSE/TIGHTEN_STOP antes de SL/TP bater."""
    if position is not None:
        return max(60, settings.analysis_interval // 2)
    return settings.analysis_interval


async def run_analysis() -> int:
    """Executa uma análise. Retorna o intervalo (s) recomendado até a próxima."""
    global current_signal
    print(f"\n[~] Analisando {settings.symbol}...", flush=True)

    loop = asyncio.get_event_loop()

    # Captura de trades fechados desde a última varredura - alimenta trades.jsonl
    # com outcomes reais. Roda em paralelo com o resto, é independente.
    n_closed = await loop.run_in_executor(None, capture_closed_trades)
    if n_closed:
        print(f"[+] {n_closed} trade(s) fechado(s) capturado(s) em logs/trades.jsonl", flush=True)

    ohlcv = await loop.run_in_executor(
        None, get_ohlcv, settings.yf_symbol, ["1d", "4h", "1h", "5m", "1m"]
    )
    if not ohlcv:
        print("[!] Sem dados - análise abortada.")
        return _next_interval(None)

    position = await loop.run_in_executor(None, get_open_position, settings.symbol)
    original_reasoning: str | None = None
    if position:
        sig = await loop.run_in_executor(
            None, find_open_signal, position.opened_at, position.side
        )
        original_reasoning = (sig or {}).get("reasoning")
        print(
            f"[~] Posição aberta: {position.side} @ {position.entry_price} "
            f"({position.pnl_pips:+.1f} pips, {position.age_minutes:.0f}min)",
            flush=True,
        )

    context, system_prompt, user_prompt = await loop.run_in_executor(
        None, lambda: _build_pipeline(ohlcv, position, original_reasoning)
    )

    t0 = time.time()
    llm_response = await loop.run_in_executor(
        None, lambda: llm.get_decision(user_prompt, system_prompt)
    )
    llm_latency_ms = int((time.time() - t0) * 1000)

    signal = build_signal(llm_response, context, position)

    current_signal = signal
    _log(signal, context, position, llm_latency_ms)
    print(
        f"[+] {signal.action} | conf: {signal.confidence:.0%} | "
        f"latency: {llm_latency_ms}ms | {signal.reasoning}"
    )

    return _next_interval(position)


async def _analysis_loop():
    delay = await run_analysis()
    while True:
        await asyncio.sleep(delay)
        try:
            delay = await run_analysis()
        except Exception as e:
            print(f"[!] Erro no loop de análise: {e}")
            delay = settings.analysis_interval


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_analysis_loop())
    yield
    task.cancel()


app = FastAPI(title="MT5 LLM Agent", version="1.6.0", lifespan=lifespan)
app.include_router(dashboard_router)


@app.get("/signal")
def get_signal():
    if current_signal is None:
        return {"action": "HOLD", "reason": "Aguardando primeira análise."}
    if not current_signal.is_valid():
        return {"action": "HOLD", "reason": "Sinal expirado."}
    return current_signal.to_dict()


@app.get("/health")
def health():
    return {
        "status": "online",
        "model_version": settings.model_version,
        "provider": llm.provider,
        "model": llm.model,
        "symbol": settings.symbol,
        "analysis_interval_s": settings.analysis_interval,
        "last_action": current_signal.action if current_signal else None,
        "signal_valid": current_signal.is_valid() if current_signal else False,
    }


@app.post("/analyze")
async def force_analyze():
    """Força uma análise imediata (útil para debug)."""
    await run_analysis()
    return current_signal.to_dict() if current_signal else {"action": "HOLD"}

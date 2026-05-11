import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from agent.llm_manager import LLMManager
from agent.prompt_builder import build_prompt
from analytics.calibration import build_calibration, calibration_to_json
from api.dashboard import router as dashboard_router
from config import settings
from data import fetcher
from data.fetcher import get_ohlcv
from data.positions import OpenPosition, get_open_positions
from data.trades import capture_closed_trades
from features.context_builder import build_context
from risk.manager import Signal, build_signal


def _build_pipeline(
    ohlcv: dict, positions: list[OpenPosition]
) -> tuple[dict, str, str, dict]:
    ctx = build_context(ohlcv, settings.symbol)
    calibration = build_calibration(lookback_days=30)
    sys_p, usr_p = build_prompt(ctx, positions, calibration)
    return ctx, sys_p, usr_p, calibration


_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "signals.jsonl"

llm = LLMManager()
current_signal: Signal | None = None


def _log(
    signal: Signal,
    context: dict,
    positions: list[OpenPosition],
    llm_latency_ms: int,
    calibration: dict,
):
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "model_version": settings.model_version,
        "provider": llm.provider,
        "model": llm.model,
        "llm_latency_ms": llm_latency_ms,
        "yf_symbol": settings.yf_symbol,
        "data_source_used": fetcher.last_source,
        "calibration_n_trades": calibration.get("n_trades_total", 0),
        "positions_at_decision": [p.to_dict() for p in positions],
        **signal.to_dict(),
        "context_summary": {
            "current_price": context.get("current_price"),
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


def _next_interval(positions: list[OpenPosition]) -> int:
    """Cadência adaptativa por horizonte (v1.7).

    Sem posições: usa analysis_interval do .env (flat default).
    Com posições: pega a menor cadência entre os horizontes ativos -
    o horizonte mais curto dita o ritmo (scalp puxa pra 90s,
    intraday pra 180s, swing pra 600s). Você sempre pode revisitar
    swing na cadência de scalp; o contrário não funciona.
    """
    if not positions:
        return settings.analysis_interval
    return min(p.cadence_seconds for p in positions)


async def run_analysis() -> int:
    """Executa uma análise. Retorna o intervalo (s) recomendado até a próxima."""
    global current_signal
    print(f"\n[~] Analisando {settings.symbol}...", flush=True)

    loop = asyncio.get_event_loop()

    n_closed = await loop.run_in_executor(None, capture_closed_trades)
    if n_closed:
        print(
            f"[+] {n_closed} trade(s) fechado(s) capturado(s) em logs/trades.jsonl",
            flush=True,
        )

    ohlcv = await loop.run_in_executor(
        None, get_ohlcv, settings.yf_symbol, ["1d", "4h", "1h", "5m", "1m"]
    )
    if not ohlcv:
        print("[!] Sem dados — análise abortada.")
        return _next_interval([])

    positions = await loop.run_in_executor(
        None, get_open_positions, settings.symbol
    )
    if positions:
        descs = ", ".join(
            f"{p.side}/{p.intended_horizon} ({p.pnl_pips:+.1f}pips, "
            f"{p.age_minutes:.0f}min)"
            for p in positions
        )
        print(f"[~] {len(positions)} posição(ões) aberta(s): {descs}", flush=True)

    context, system_prompt, user_prompt, calibration = await loop.run_in_executor(
        None, lambda: _build_pipeline(ohlcv, positions)
    )

    t0 = time.time()
    llm_response = await loop.run_in_executor(
        None, lambda: llm.get_decision(user_prompt, system_prompt)
    )
    llm_latency_ms = int((time.time() - t0) * 1000)

    signal = build_signal(llm_response, context, positions)
    current_signal = signal
    _log(signal, context, positions, llm_latency_ms, calibration)

    horizon_tag = (
        f" [{signal.intended_horizon}]" if signal.intended_horizon else ""
    )
    target_tag = (
        f" #{signal.position_id}" if signal.position_id else ""
    )
    print(
        f"[+] {signal.action}{horizon_tag}{target_tag} | "
        f"conf: {signal.confidence:.0%} | "
        f"latency: {llm_latency_ms}ms | {signal.reasoning}"
    )
    return _next_interval(positions)


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


app = FastAPI(title="MT5 LLM Agent", version="1.8.1", lifespan=lifespan)
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
        "max_positions": settings.max_positions,
        "analysis_interval_s": settings.analysis_interval,
        "last_action": current_signal.action if current_signal else None,
        "signal_valid": current_signal.is_valid() if current_signal else False,
    }


@app.post("/analyze")
async def force_analyze():
    """Força uma análise imediata (útil para debug)."""
    await run_analysis()
    return current_signal.to_dict() if current_signal else {"action": "HOLD"}

"""
Simulação post-hoc de outcomes a partir dos sinais logados.

Estratégia: o servidor sabe entry, SL e TP de cada sinal. Buscamos as velas 1m
posteriores e determinamos qual nível bateu primeiro. Mede o MODELO puro,
sem ruído de slippage/spread/comissão - exatamente o que importa para o paper.

Limitação: yfinance só tem 7 dias de histórico em 1m. Sinais mais antigos
retornam "no_data" e devem ser analisados com 5m (60 dias) - TODO futuro.
"""

import datetime

import pandas as pd
import yfinance as yf


def fetch_market_data(yf_symbol: str = "EURUSD=X", period: str = "7d",
                      interval: str = "1m") -> pd.DataFrame:
    """Busca uma janela única de dados 1m para reuso em simulações em batch."""
    df = yf.download(
        yf_symbol, period=period, interval=interval,
        progress=False, auto_adjust=True,
    )
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def _ensure_utc(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def _signal_side(signal: dict) -> str | None:
    """Lado direcional do sinal, ou None para ações não-direcionais.

    Apenas OPEN_LONG/OPEN_SHORT abrem trades simuláveis;
    HOLD/CLOSE/TIGHTEN_STOP não têm entry+SL+TP próprios.
    """
    action = signal.get("action")
    if action == "OPEN_LONG":
        return "LONG"
    if action == "OPEN_SHORT":
        return "SHORT"
    return None


def _build_outcome(resolution: str, exit_price: float, exit_ts,
                   signal: dict, created_utc: datetime.datetime) -> dict:
    side = _signal_side(signal)
    entry = signal["entry_price"]

    if side == "LONG":
        pnl_pips = (exit_price - entry) * 10_000
    else:
        pnl_pips = (entry - exit_price) * 10_000

    duration = int((exit_ts.to_pydatetime() - created_utc).total_seconds())

    return {
        "resolution": resolution,
        "exit_price": round(float(exit_price), 5),
        "exit_at": exit_ts.isoformat(),
        "pnl_pips": round(float(pnl_pips), 1),
        "duration_seconds": duration,
    }


def simulate_outcome(signal: dict, market_data: pd.DataFrame,
                     max_hours: int = 24) -> dict:
    """
    Determina o resultado de um sinal individual contra um DataFrame de velas.

    Resoluções possíveis:
      - "no_trade"  → ação não-direcional (HOLD, CLOSE, TIGHTEN_STOP)
      - "no_levels" → SL ou TP ausentes (não deveria acontecer em produção)
      - "no_data"   → sem velas no horizonte (sinal muito recente ou muito antigo)
      - "tp_hit"    → take profit atingido
      - "sl_hit"    → stop loss atingido (conservador: ganha em caso de empate)
      - "expired"   → max_hours atingido sem hit
    """
    side = _signal_side(signal)
    if side is None:
        # HOLD, CLOSE, TIGHTEN_STOP - sem trade direcional novo a simular
        return {"resolution": "no_trade"}
    if signal.get("sl_price") is None or signal.get("tp_price") is None:
        return {"resolution": "no_levels"}

    created_at_str = signal.get("created_at")
    if not created_at_str:
        return {"resolution": "missing_created_at"}

    created = _ensure_utc(datetime.datetime.fromisoformat(created_at_str))
    end = created + datetime.timedelta(hours=max_hours)
    window = market_data[(market_data.index > created) & (market_data.index <= end)]

    if window.empty:
        return {"resolution": "no_data"}

    sl = float(signal["sl_price"])
    tp = float(signal["tp_price"])

    for ts, row in window.iterrows():
        high = float(row["High"])
        low = float(row["Low"])

        if side == "LONG":
            sl_hit = low <= sl
            tp_hit = high >= tp
        else:
            sl_hit = high >= sl
            tp_hit = low <= tp

        # Conservador: se ambos batem na mesma vela, assume SL primeiro.
        # (sem dados intra-vela, é a hipótese mais segura para o paper.)
        if sl_hit:
            return _build_outcome("sl_hit", sl, ts, signal, created)
        if tp_hit:
            return _build_outcome("tp_hit", tp, ts, signal, created)

    last_ts = window.index[-1]
    return {
        "resolution": "expired",
        "duration_seconds": int((last_ts.to_pydatetime() - created).total_seconds()),
    }

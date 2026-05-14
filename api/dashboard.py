"""
Dashboard de runtime - visualização agregada dos logs de sinais e trades.

Endpoints:
  GET /            → HTML estático (Chart.js via CDN, autoatualiza)
  GET /api/dashboard → JSON com tudo que os gráficos precisam

Lê `logs/signals.jsonl` (decisões do LLM) e `logs/trades.jsonl` (lifecycle
real de trades fechados). Sem dependência adicional: pandas/matplotlib não
entram aqui - agregação em Python puro, plotagem do lado do navegador.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

from config import settings


router = APIRouter()

_LOG_DIR = Path("logs")
_SIGNALS = _LOG_DIR / "signals.jsonl"
_TRADES = _LOG_DIR / "trades.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
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


def _signal_regime(s: dict) -> str:
    """Regime estatístico associado ao sinal - usa Hurst do 1h.

    Fallback: 'desconhecido' se contexto faltar.
    """
    ctx = s.get("context_summary", {})
    tfs = ctx.get("timeframes", {})
    h1 = tfs.get("1h", {})
    stats = h1.get("statistics") or {}
    hurst = stats.get("hurst") or {}
    return hurst.get("regime") or "desconhecido"


def _summary(signals: list[dict], trades: list[dict]) -> dict:
    last = signals[-1] if signals else None
    open_positions = (last or {}).get("positions_at_decision") or []

    pnl_money = round(sum(t.get("pnl_money", 0.0) for t in trades), 2)
    pnl_pips = round(sum(t.get("pnl_pips", 0.0) for t in trades), 1)
    wins = sum(1 for t in trades if t.get("pnl_money", 0) > 0)

    last_ctx = (last or {}).get("context_summary") or {}
    last_source = (last or {}).get("data_source_used")
    fallback_active = (
        last_source == "yfinance"
        and settings.data_source == "mt5"
    )
    return {
        "model_version": settings.model_version,
        "symbol": settings.symbol,
        "max_positions": settings.max_positions,
        "current_price": last_ctx.get("current_price"),
        "data_source_configured": settings.data_source,
        "data_source_used": last_source,
        "fallback_active": fallback_active,
        "last_action": (last or {}).get("action"),
        "last_horizon": (last or {}).get("intended_horizon"),
        "last_position_id": (last or {}).get("position_id"),
        "last_confidence": (last or {}).get("confidence"),
        "last_reasoning": (last or {}).get("reasoning"),
        "last_at": (last or {}).get("created_at"),
        "last_lot": (last or {}).get("lot"),
        "last_equity": (last or {}).get("equity"),
        "circuit_state": (last or {}).get("circuit_state"),
        "open_positions": open_positions,
        "total_signals": len(signals),
        "total_trades": len(trades),
        "wins": wins,
        "win_rate_pct": round(wins / len(trades) * 100, 1) if trades else None,
        "total_pnl_money": pnl_money,
        "total_pnl_pips": pnl_pips,
    }


def _equity_curve(trades: list[dict]) -> list[dict]:
    """Equity curve cumulativa por trade (em moeda da conta)."""
    sorted_trades = sorted(trades, key=lambda t: t.get("closed_at", ""))
    cum = 0.0
    out = []
    for t in sorted_trades:
        cum += float(t.get("pnl_money") or 0.0)
        out.append({
            "t": t.get("closed_at"),
            "pnl": round(cum, 2),
            "side": t.get("side"),
            "reason": t.get("close_reason"),
            "trade_pnl_pips": t.get("pnl_pips"),
        })
    return out


def _action_counts(signals: list[dict]) -> dict:
    return dict(Counter(s.get("action", "UNKNOWN") for s in signals))


def _close_reasons(trades: list[dict]) -> dict:
    return dict(Counter(t.get("close_reason", "unknown") for t in trades))


def _latency_series(signals: list[dict]) -> list[dict]:
    """Latência do LLM ao longo do tempo (ms)."""
    out = []
    for s in signals:
        ms = s.get("llm_latency_ms")
        t = s.get("created_at")
        if ms is None or t is None:
            continue
        out.append({"t": t, "ms": int(ms)})
    return out


def _regime_winrate(signals: list[dict], trades: list[dict]) -> dict:
    """Win rate agregado por regime de Hurst no momento da abertura.

    Junta trades.jsonl ↔ signals.jsonl via signal_id_open.
    """
    sig_by_id: dict[str, dict] = {
        s.get("signal_id"): s for s in signals if s.get("signal_id")
    }
    agg: dict[str, dict] = defaultdict(lambda: {"wins": 0, "trades": 0, "pnl": 0.0})
    for t in trades:
        sid = t.get("signal_id_open")
        sig = sig_by_id.get(sid)
        regime = _signal_regime(sig) if sig else "desconhecido"
        agg[regime]["trades"] += 1
        agg[regime]["pnl"] += float(t.get("pnl_money") or 0.0)
        if (t.get("pnl_money") or 0) > 0:
            agg[regime]["wins"] += 1
    return {
        regime: {
            "trades": d["trades"],
            "wins": d["wins"],
            "win_rate_pct": round(d["wins"] / d["trades"] * 100, 1) if d["trades"] else 0,
            "pnl_money": round(d["pnl"], 2),
        }
        for regime, d in agg.items()
    }


def _horizon_winrate(signals: list[dict], trades: list[dict]) -> dict:
    """Win rate agregado pelo intended_horizon do sinal de abertura."""
    sig_by_id = {s.get("signal_id"): s for s in signals if s.get("signal_id")}
    agg: dict[str, dict] = defaultdict(lambda: {"wins": 0, "trades": 0, "pnl": 0.0})
    for t in trades:
        sig = sig_by_id.get(t.get("signal_id_open"))
        h = (sig or {}).get("intended_horizon") or "desconhecido"
        agg[h]["trades"] += 1
        agg[h]["pnl"] += float(t.get("pnl_money") or 0.0)
        if (t.get("pnl_money") or 0) > 0:
            agg[h]["wins"] += 1
    return {
        h: {
            "trades": d["trades"],
            "wins": d["wins"],
            "win_rate_pct": round(d["wins"] / d["trades"] * 100, 1) if d["trades"] else 0,
            "pnl_money": round(d["pnl"], 2),
        }
        for h, d in agg.items()
    }


def _duration_buckets(trades: list[dict]) -> dict:
    """Histograma de duração de trades em buckets pré-definidos (minutos)."""
    buckets = {"<5min": 0, "5-30min": 0, "30min-2h": 0, "2-6h": 0, "6-24h": 0, ">24h": 0}
    for t in trades:
        d = float(t.get("duration_min") or 0)
        if d < 5: buckets["<5min"] += 1
        elif d < 30: buckets["5-30min"] += 1
        elif d < 120: buckets["30min-2h"] += 1
        elif d < 360: buckets["2-6h"] += 1
        elif d < 1440: buckets["6-24h"] += 1
        else: buckets[">24h"] += 1
    return buckets


@router.get("/api/dashboard")
def api_dashboard() -> JSONResponse:
    signals = _read_jsonl(_SIGNALS)
    trades = _read_jsonl(_TRADES)
    return JSONResponse({
        "summary": _summary(signals, trades),
        "equity_curve": _equity_curve(trades),
        "actions": _action_counts(signals),
        "close_reasons": _close_reasons(trades),
        "latency": _latency_series(signals)[-200:],   # últimos 200 pontos
        "regimes": _regime_winrate(signals, trades),
        "horizons": _horizon_winrate(signals, trades),
        "durations": _duration_buckets(trades),
    })


@router.get("/api/signals_history")
def api_signals_history(limit: int = 50) -> JSONResponse:
    """Histórico de decisões para o modal de "Ver histórico".

    Retorna em ordem inversa (mais recente primeiro), com campos enxutos.
    """
    signals = _read_jsonl(_SIGNALS)
    recent = list(reversed(signals[-limit:]))
    out = []
    for s in recent:
        ctx = s.get("context_summary") or {}
        out.append({
            "signal_id": s.get("signal_id"),
            "created_at": s.get("created_at"),
            "action": s.get("action"),
            "intended_horizon": s.get("intended_horizon"),
            "position_id": s.get("position_id"),
            "confidence": s.get("confidence"),
            "reasoning": s.get("reasoning"),
            "lot": s.get("lot"),
            "entry_price": s.get("entry_price"),
            "sl_price": s.get("sl_price"),
            "tp_price": s.get("tp_price"),
            "current_price": ctx.get("current_price"),
            "llm_latency_ms": s.get("llm_latency_ms"),
        })
    return JSONResponse({"signals": out, "total": len(signals)})


@router.get("/api/price_chart")
def api_price_chart(hours: int = 24) -> JSONResponse:
    """Série de preço da última N horas + marcadores de trades fechados.

    Cache do fetcher (5min TTL) absorve calls repetidos sem custo extra.
    """
    import datetime
    from data.fetcher import get_ohlcv

    # Usa 5m candles para ~24h (288 pontos), ou 1h para ranges mais longos.
    tf = "5m" if hours <= 48 else "1h"
    ohlcv = get_ohlcv(settings.yf_symbol, [tf])
    df = (ohlcv or {}).get(tf)

    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)

    prices: list[dict] = []
    if df is not None and not df.empty:
        recent_df = df[df.index >= cutoff]
        for ts, row in recent_df.iterrows():
            prices.append({
                "t": ts.isoformat(),
                "o": round(float(row["Open"]), 5),
                "h": round(float(row["High"]), 5),
                "l": round(float(row["Low"]), 5),
                "c": round(float(row["Close"]), 5),
            })

    trades = _read_jsonl(_TRADES)
    markers: list[dict] = []
    for t in trades:
        try:
            opened = datetime.datetime.fromisoformat(t["opened_at"])
            closed = datetime.datetime.fromisoformat(t["closed_at"])
        except (KeyError, ValueError):
            continue
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=datetime.timezone.utc)
        if closed.tzinfo is None:
            closed = closed.replace(tzinfo=datetime.timezone.utc)
        if closed < cutoff:
            continue
        markers.append({
            "opened_at": opened.isoformat(),
            "closed_at": closed.isoformat(),
            "side": t.get("side"),
            "horizon": (t.get("open_reasoning") and "—") or None,  # placeholder
            "entry_price": t.get("entry_price"),
            "exit_price": t.get("exit_price"),
            "pnl_money": t.get("pnl_money"),
            "pnl_pips": t.get("pnl_pips"),
            "close_reason": t.get("close_reason"),
        })

    return JSONResponse({
        "prices": prices,
        "trades": markers,
        "timeframe": tf,
        "hours": hours,
    })


@router.get("/api/calibration")
def api_calibration(lookback_days: int = 30) -> JSONResponse:
    """Mesma calibração que o LLM vê, exposta pro dashboard.

    Útil pra o usuário verificar de forma transparente quais buckets
    estão influenciando as decisões do agente.
    """
    from analytics.calibration import build_calibration, calibration_to_json
    cal = build_calibration(lookback_days=lookback_days)
    return JSONResponse(calibration_to_json(cal))


@router.get("/", response_class=HTMLResponse)
def dashboard_page() -> HTMLResponse:
    return HTMLResponse(_HTML)


# ---------------------------------------------------------------------------
# HTML inline - Chart.js via CDN, sem build pipeline.
# ---------------------------------------------------------------------------
_HTML = r"""<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<title>MT5 LLM Agent - Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/luxon@3.4.4/build/global/luxon.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.3.1"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial@0.2.1/dist/chartjs-chart-financial.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --border: #30363d; --text: #c9d1d9;
    --muted: #8b949e; --accent: #58a6ff; --green: #3fb950; --red: #f85149;
    --yellow: #d29922;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    background: var(--bg); color: var(--text); padding: 20px;
  }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .subtitle { color: var(--muted); font-size: 12px; }
  .header {
    display: flex; justify-content: space-between; align-items: flex-start;
    margin-bottom: 20px;
  }
  .price {
    color: var(--text); font-weight: 600; font-variant-numeric: tabular-nums;
  }
  .pulse {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--green); margin: 0 6px; vertical-align: middle;
    animation: pulse 2s infinite;
  }
  .pulse.stale { background: var(--yellow); animation: none; }
  .panel-head {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 10px;
  }
  .panel-head h2 { margin: 0; }
  .seg { display: inline-flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
  .seg-btn {
    background: var(--bg); color: var(--muted); border: none;
    padding: 4px 12px; font-family: inherit; font-size: 11px; cursor: pointer;
  }
  .seg-btn:hover { color: var(--text); }
  .seg-btn.active { background: var(--accent); color: white; }
  .alert-fallback {
    background: rgba(248, 81, 73, 0.12); border: 1px solid var(--red);
    color: var(--red); padding: 10px 14px; border-radius: 6px;
    margin-bottom: 16px; font-size: 12px;
  }
  .alert-fallback strong { color: #ff7b72; }
  @keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(63,185,80,0.6); }
    50%      { opacity: 0.7; box-shadow: 0 0 0 6px rgba(63,185,80,0); }
  }
  .btn {
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 14px; cursor: pointer; font-family: inherit;
    font-size: 12px;
  }
  .btn:hover { background: var(--border); }
  .modal {
    position: fixed; inset: 0; background: rgba(0,0,0,0.7);
    display: flex; align-items: center; justify-content: center; z-index: 100;
    padding: 20px;
  }
  .modal[hidden] { display: none !important; }
  .modal-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    width: 100%; max-width: 1100px; max-height: 85vh; display: flex; flex-direction: column;
  }
  .modal-head {
    display: flex; justify-content: space-between; align-items: center;
    padding: 14px 18px; border-bottom: 1px solid var(--border);
  }
  .modal-head h2 { margin: 0; font-size: 14px; }
  .btn-close {
    background: none; border: none; color: var(--muted); font-size: 24px;
    cursor: pointer; line-height: 1;
  }
  .btn-close:hover { color: var(--text); }
  .modal-body { padding: 14px 18px; overflow-y: auto; }
  #history-table td.reason-cell {
    color: var(--muted); font-style: italic; max-width: 400px;
    white-space: normal; word-break: break-word;
  }
  .grid {
    display: grid; gap: 16px;
    grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
  }
  .panel {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px;
  }
  .panel h2 {
    font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px;
    color: var(--muted); margin: 0 0 10px;
  }
  .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
  .kpi { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 10px; }
  .kpi .v { font-size: 18px; font-weight: 600; }
  .kpi .l { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .pos { color: var(--green); }
  .neg { color: var(--red); }
  .neutral { color: var(--text); }
  .last {
    font-size: 12px; line-height: 1.6; word-break: break-word;
  }
  .last .a { font-size: 14px; font-weight: 600; color: var(--accent); }
  .last .reason { color: var(--muted); font-style: italic; margin-top: 4px; }
  canvas { max-height: 240px; }
  .full { grid-column: 1 / -1; }
  table { width: 100%; font-size: 12px; border-collapse: collapse; }
  td, th { padding: 4px 8px; border-bottom: 1px solid var(--border); text-align: left; }
  th { color: var(--muted); font-weight: normal; font-size: 10px; text-transform: uppercase; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>MT5 LLM Agent</h1>
    <div class="subtitle" id="subtitle">carregando...</div>
  </div>
  <button class="btn" id="btn-history" type="button">Ver histórico de decisões</button>
</div>

<div class="alert-fallback" id="alert-fallback" hidden>
  <strong>⚠ Fallback yfinance ativo.</strong>
  Você configurou <code>DATA_SOURCE=mt5</code>, mas o último sinal foi gerado com dados de yfinance —
  a bridge mt5linux pode estar fora ou o MT5 desconectado do broker. Os preços podem divergir do que
  você vê no terminal.
</div>

<div class="grid">

  <div class="panel full">
    <h2>Status</h2>
    <div class="kpis" id="kpis"></div>
  </div>

  <div class="panel full">
    <h2>Última decisão</h2>
    <div class="last" id="last"></div>
  </div>

  <div class="panel full">
    <h2>Equity curve (P&amp;L cumulativo, moeda da conta)</h2>
    <canvas id="equity"></canvas>
  </div>

  <div class="panel">
    <h2>Distribuição de ações</h2>
    <canvas id="actions"></canvas>
  </div>

  <div class="panel">
    <h2>Como os trades fecharam</h2>
    <canvas id="closes"></canvas>
  </div>

  <div class="panel">
    <h2>Duração dos trades</h2>
    <canvas id="durations"></canvas>
  </div>

  <div class="panel">
    <h2>Latência do LLM (ms)</h2>
    <canvas id="latency"></canvas>
  </div>

  <div class="panel full">
    <div class="panel-head">
      <h2>Preço &amp; trades (últimas 24h, M5) <span style="color:var(--muted); font-size:11px; font-weight:normal;">▲▼ entrada · × saída · verde=lucro · vermelho=prejuízo</span></h2>
      <div class="seg">
        <button class="seg-btn" data-mode="candle" type="button">Velas</button>
        <button class="seg-btn" data-mode="line" type="button">Linha</button>
      </div>
    </div>
    <canvas id="pricechart" style="max-height: 320px;"></canvas>
  </div>

  <div class="panel full">
    <h2>Posições abertas (snapshot da última análise)</h2>
    <table id="positions">
      <thead><tr><th>Ticket</th><th>Lado</th><th>Horizonte</th><th>Entrada</th><th>Atual</th><th>P&amp;L</th><th>SL</th><th>TP</th><th>Idade</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>

  <div class="panel">
    <h2>Win rate por horizonte</h2>
    <table id="horizons">
      <thead><tr><th>Horizonte</th><th>Trades</th><th>Wins</th><th>Win rate</th><th>P&amp;L</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>

  <div class="panel">
    <h2>Win rate por regime estatístico (Hurst 1h)</h2>
    <table id="regimes">
      <thead><tr><th>Regime</th><th>Trades</th><th>Wins</th><th>Win rate</th><th>P&amp;L</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>

  <div class="panel full">
    <h2>Calibração histórica <span style="color:var(--muted); font-size:11px; font-weight:normal;">— mesma tabela que o LLM vê na decisão (últimos 30d)</span></h2>
    <table id="calibration">
      <thead><tr><th>Regime</th><th>Horizonte</th><th>Lado</th><th>Trades</th><th>Win rate</th><th>P&amp;L médio (pips)</th><th>Profit factor</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>

</div>

<!-- Modal de histórico de decisões -->
<div class="modal" id="modal-history" hidden>
  <div class="modal-card">
    <div class="modal-head">
      <h2>Histórico de decisões</h2>
      <button class="btn-close" id="btn-close-history" type="button">×</button>
    </div>
    <div class="modal-body">
      <table id="history-table">
        <thead>
          <tr>
            <th>Quando</th>
            <th>Ação</th>
            <th>Horizonte</th>
            <th>Conf.</th>
            <th>Preço</th>
            <th>Lote</th>
            <th>Latência</th>
            <th>Reasoning</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const COLORS = {
  OPEN_LONG: '#3fb950', OPEN_SHORT: '#f85149',
  HOLD: '#8b949e', CLOSE: '#d29922', TIGHTEN_STOP: '#58a6ff',
  tp: '#3fb950', sl: '#f85149', manual: '#d29922', unknown: '#6e7681',
};

let charts = {};

function fmtPnL(v, suffix = '') {
  if (v === null || v === undefined) return '-';
  const cls = v > 0 ? 'pos' : v < 0 ? 'neg' : 'neutral';
  const sign = v > 0 ? '+' : '';
  return `<span class="${cls}">${sign}${v}${suffix}</span>`;
}

function renderKPIs(s) {
  const priceTxt = s.current_price != null ? Number(s.current_price).toFixed(5) : '—';
  // Bolinha pulsa se a última análise foi nos últimos 10min, senão fica amarela "stale".
  let stale = true;
  if (s.last_at) {
    const ageMin = (Date.now() - new Date(s.last_at).getTime()) / 60000;
    stale = ageMin > 10;
  }
  document.getElementById('subtitle').innerHTML =
    `${s.symbol} <span class="price">${priceTxt}</span>` +
    `<span class="pulse${stale ? ' stale' : ''}" title="${stale ? 'sinal antigo (>10min)' : 'sinal recente'}"></span>` +
    `· ${s.model_version} · atualizado ${new Date().toLocaleTimeString()}`;
  const html = `
    <div class="kpi"><div class="l">Sinais</div><div class="v">${s.total_signals}</div></div>
    <div class="kpi"><div class="l">Trades fechados</div><div class="v">${s.total_trades}</div></div>
    <div class="kpi"><div class="l">Win rate</div><div class="v">${s.win_rate_pct ?? '-'}${s.win_rate_pct != null ? '%' : ''}</div></div>
    <div class="kpi"><div class="l">P&L acumulado</div><div class="v">${fmtPnL(s.total_pnl_money, ' $')}<br><span style="font-size:11px">${fmtPnL(s.total_pnl_pips, ' pips')}</span></div></div>
  `;
  document.getElementById('kpis').innerHTML = html;
}

function renderLast(s) {
  const positions = s.open_positions || [];
  const posLine = positions.length
    ? `${positions.length}/${s.max_positions} posições abertas (ver tabela abaixo)`
    : `Flat — pode abrir até ${s.max_positions} posições`;
  const horizonTag = s.last_horizon ? ` <span style="color: var(--accent);">[${s.last_horizon}]</span>` : '';
  const targetTag = s.last_position_id ? ` → ticket #${s.last_position_id}` : '';

  const cs = s.circuit_state;
  let circuitLine = '';
  if (cs) {
    const state = cs.blocked
      ? `<span class="neg">BLOQUEADO</span> - ${cs.reason}`
      : `<span class="pos">livre</span> · DD hoje ${fmtPnL(cs.daily_pnl, ' $')} · perdas seguidas ${cs.consecutive_losses}`;
    circuitLine = `<div style="margin-top: 4px;">Circuit breaker: ${state}</div>`;
  }

  const sizingLine = (s.last_lot != null && s.last_equity != null)
    ? `<div style="margin-top: 4px;">Equity: $${s.last_equity} · Lote sugerido: ${s.last_lot}</div>`
    : '';

  document.getElementById('last').innerHTML = `
    <div><span class="a">${s.last_action ?? '-'}</span>${horizonTag}${targetTag} · confiança ${s.last_confidence != null ? (s.last_confidence * 100).toFixed(0) + '%' : '-'} · ${s.last_at ? new Date(s.last_at).toLocaleString() : '-'}</div>
    <div class="reason">"${s.last_reasoning ?? ''}"</div>
    <div style="margin-top: 8px;">${posLine}</div>
    ${sizingLine}
    ${circuitLine}
  `;
}

function fmtPrice(v) {
  // EURUSD usa 5 decimais. MT5 às vezes devolve floats com lixo de
  // precisão (ex: 1.1776100000000003); o toFixed normaliza pra largura
  // visual consistente da coluna.
  return (v != null) ? Number(v).toFixed(5) : '—';
}

function renderPositions(positions) {
  const tbody = document.querySelector('#positions tbody');
  tbody.innerHTML = '';
  if (!positions || positions.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" style="color: var(--muted); text-align:center;">Nenhuma posição aberta.</td></tr>';
    return;
  }
  for (const p of positions) {
    const horizonTag = p.horizon_inferred ? `${p.intended_horizon} <span style="color:var(--muted);">(inferido)</span>` : p.intended_horizon;
    tbody.innerHTML += `
      <tr>
        <td>${p.ticket}</td>
        <td>${p.side}</td>
        <td>${horizonTag}</td>
        <td class="num">${fmtPrice(p.entry_price)}</td>
        <td class="num">${fmtPrice(p.current_price)}</td>
        <td class="num">${fmtPnL(p.pnl_pips, ' pips')}</td>
        <td class="num">${fmtPrice(p.sl_price)}</td>
        <td class="num">${fmtPrice(p.tp_price)}</td>
        <td class="num">${p.age_minutes.toFixed(0)}min</td>
      </tr>`;
  }
}

function renderCalibration(cal) {
  const tbody = document.querySelector('#calibration tbody');
  tbody.innerHTML = '';
  const entries = Object.entries(cal.by_full_key || {});
  if (!entries.length) {
    tbody.innerHTML = `<tr><td colspan="7" style="color:var(--muted); text-align:center;">Sem trades com contexto associado ainda (${cal.n_trades_unmatched ?? 0} fechados sem signal_id casado).</td></tr>`;
    return;
  }
  // Ordena por (regime, horizonte, lado)
  entries.sort((a, b) => a[0].localeCompare(b[0]));
  for (const [key, d] of entries) {
    const [regime, horizon, side] = key.split(' | ');
    const wr = d.win_rate_pct != null ? `${d.win_rate_pct}%` : `n<3`;
    const wrCls = d.win_rate_pct == null ? 'neutral'
                : d.win_rate_pct >= 50 ? 'pos' : 'neg';
    const pf = d.profit_factor != null ? d.profit_factor : '—';
    tbody.innerHTML += `
      <tr>
        <td>${regime}</td>
        <td>${horizon}</td>
        <td>${side}</td>
        <td class="num">${d.n_trades}</td>
        <td class="num ${wrCls}">${wr}</td>
        <td class="num">${fmtPnL(d.avg_pips, '')}</td>
        <td class="num">${pf}</td>
      </tr>`;
  }
}

function renderFallbackAlert(s) {
  const el = document.getElementById('alert-fallback');
  el.hidden = !s.fallback_active;
}

function renderHorizons(horizons) {
  const tbody = document.querySelector('#horizons tbody');
  tbody.innerHTML = '';
  const entries = Object.entries(horizons);
  if (entries.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" style="color: var(--muted); text-align:center;">Sem trades ainda.</td></tr>';
    return;
  }
  for (const [h, d] of entries) {
    const wrCls = d.win_rate_pct >= 50 ? 'pos' : 'neg';
    tbody.innerHTML += `
      <tr>
        <td>${h}</td>
        <td class="num">${d.trades}</td>
        <td class="num">${d.wins}</td>
        <td class="num ${wrCls}">${d.win_rate_pct}%</td>
        <td class="num">${fmtPnL(d.pnl_money, ' $')}</td>
      </tr>`;
  }
}

function makeOrUpdate(id, type, data, options) {
  if (charts[id]) { charts[id].data = data; charts[id].options = options; charts[id].update(); return; }
  charts[id] = new Chart(document.getElementById(id), { type, data, options });
}

const baseOpts = {
  responsive: true, maintainAspectRatio: false,
  plugins: { legend: { labels: { color: '#c9d1d9', font: { size: 11 } } } },
  scales: {
    x: { ticks: { color: '#8b949e', font: { size: 10 } }, grid: { color: '#21262d' } },
    y: { ticks: { color: '#8b949e', font: { size: 10 } }, grid: { color: '#21262d' } },
  },
};

function renderEquity(curve) {
  const labels = curve.map(p => p.t ? new Date(p.t).toLocaleString() : '');
  const data = curve.map(p => p.pnl);
  makeOrUpdate('equity', 'line', {
    labels,
    datasets: [{
      label: 'P&L cumulativo ($)', data,
      borderColor: '#58a6ff', backgroundColor: 'rgba(88,166,255,0.15)',
      fill: true, tension: 0.2, pointRadius: 3,
    }],
  }, baseOpts);
}

function renderDoughnut(id, obj) {
  const labels = Object.keys(obj);
  const data = Object.values(obj);
  const colors = labels.map(l => COLORS[l] || '#58a6ff');
  makeOrUpdate(id, 'doughnut', {
    labels, datasets: [{ data, backgroundColor: colors, borderColor: '#0d1117' }],
  }, { responsive: true, maintainAspectRatio: false,
       plugins: { legend: { position: 'right', labels: { color: '#c9d1d9', font: { size: 11 } } } } });
}

function renderDurations(buckets) {
  makeOrUpdate('durations', 'bar', {
    labels: Object.keys(buckets),
    datasets: [{ label: 'Trades', data: Object.values(buckets), backgroundColor: '#58a6ff' }],
  }, baseOpts);
}

function renderLatency(series) {
  const labels = series.map(p => new Date(p.t).toLocaleTimeString());
  const data = series.map(p => p.ms);
  makeOrUpdate('latency', 'line', {
    labels,
    datasets: [{
      label: 'ms', data, borderColor: '#d29922',
      backgroundColor: 'rgba(210,153,34,0.15)', fill: true, tension: 0.3, pointRadius: 0,
    }],
  }, baseOpts);
}

function renderRegimes(regimes) {
  const tbody = document.querySelector('#regimes tbody');
  tbody.innerHTML = '';
  const entries = Object.entries(regimes);
  if (entries.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" style="color: var(--muted); text-align:center;">Sem trades ainda.</td></tr>';
    return;
  }
  for (const [regime, d] of entries) {
    const wrCls = d.win_rate_pct >= 50 ? 'pos' : 'neg';
    tbody.innerHTML += `
      <tr>
        <td>${regime}</td>
        <td class="num">${d.trades}</td>
        <td class="num">${d.wins}</td>
        <td class="num ${wrCls}">${d.win_rate_pct}%</td>
        <td class="num">${fmtPnL(d.pnl_money, ' $')}</td>
      </tr>`;
  }
}

// Estado persistente do modo de gráfico (vela vs linha).
let priceMode = localStorage.getItem('priceMode') || 'candle';
let lastPriceData = null;  // pra re-renderizar ao trocar de modo sem refetch

function renderPriceChart(data) {
  lastPriceData = data;
  const prices = data.prices || [];
  const trades = data.trades || [];

  // Markers de trades — comuns aos dois modos.
  const entries = [];
  const exits = [];
  for (const t of trades) {
    const isWin = (t.pnl_money ?? 0) > 0;
    const color = isWin ? '#3fb950' : '#f85149';
    entries.push({
      x: new Date(t.opened_at).getTime(),
      y: t.entry_price,
      color, side: t.side, pnl: t.pnl_money, pips: t.pnl_pips, reason: t.close_reason,
    });
    exits.push({
      x: new Date(t.closed_at).getTime(),
      y: t.exit_price,
      color, side: t.side, pnl: t.pnl_money, pips: t.pnl_pips, reason: t.close_reason,
    });
  }

  const tradeDatasets = [
    {
      label: 'Entradas', type: 'scatter', data: entries,
      pointStyle: 'triangle',
      rotation: ctx => ctx.raw && ctx.raw.side === 'SHORT' ? 180 : 0,
      backgroundColor: ctx => (ctx.raw && ctx.raw.color) || '#58a6ff',
      borderColor: '#0d1117', borderWidth: 1, pointRadius: 9,
    },
    {
      label: 'Saídas', type: 'scatter', data: exits,
      pointStyle: 'crossRot',
      backgroundColor: ctx => (ctx.raw && ctx.raw.color) || '#58a6ff',
      borderColor: ctx => (ctx.raw && ctx.raw.color) || '#58a6ff',
      borderWidth: 2.5, pointRadius: 8,
    },
  ];

  const baseOptions = {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#c9d1d9', font: { size: 11 } } },
      tooltip: {
        callbacks: {
          label: ctx => {
            const r = ctx.raw;
            if (r && r.pnl != null) {
              const sign = r.pnl > 0 ? '+' : '';
              return `${r.side} · ${sign}${r.pips} pips (${sign}${r.pnl} $) · ${r.reason ?? ''}`;
            }
            if (r && r.o != null) {
              return `O ${r.o} · H ${r.h} · L ${r.l} · C ${r.c}`;
            }
            return `${ctx.dataset.label}: ${r.y}`;
          },
          title: ctx => new Date(ctx[0].parsed.x).toLocaleString(),
        },
      },
    },
    scales: {
      x: {
        type: 'time',
        time: { unit: 'hour', displayFormats: { hour: 'HH:mm' } },
        ticks: { color: '#8b949e', font: { size: 10 }, maxTicksLimit: 12 },
        grid: { color: '#21262d' },
      },
      y: {
        ticks: { color: '#8b949e', font: { size: 10 } },
        grid: { color: '#21262d' },
      },
    },
  };

  // Destrói o chart anterior pra trocar de tipo limpo (line ↔ candlestick).
  if (charts.pricechart) {
    charts.pricechart.destroy();
    delete charts.pricechart;
  }

  if (priceMode === 'candle') {
    const candleData = prices.map(p => ({
      x: new Date(p.t).getTime(),
      o: p.o, h: p.h, l: p.l, c: p.c,
    }));
    charts.pricechart = new Chart(document.getElementById('pricechart'), {
      type: 'candlestick',
      data: {
        datasets: [
          {
            label: 'OHLC',
            data: candleData,
            color: { up: '#3fb950', down: '#f85149', unchanged: '#8b949e' },
            borderColor: { up: '#3fb950', down: '#f85149', unchanged: '#8b949e' },
          },
          ...tradeDatasets,
        ],
      },
      options: baseOptions,
    });
  } else {
    const lineData = prices.map(p => ({ x: new Date(p.t).getTime(), y: p.c }));
    charts.pricechart = new Chart(document.getElementById('pricechart'), {
      type: 'line',
      data: {
        datasets: [
          {
            label: 'Preço (close)',
            data: lineData,
            borderColor: '#58a6ff',
            backgroundColor: 'rgba(88,166,255,0.08)',
            fill: true, tension: 0.2, pointRadius: 0, borderWidth: 1.5,
          },
          ...tradeDatasets,
        ],
      },
      options: baseOptions,
    });
  }
}

function setPriceMode(mode) {
  priceMode = mode;
  localStorage.setItem('priceMode', mode);
  document.querySelectorAll('.seg-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
  if (lastPriceData) renderPriceChart(lastPriceData);
}

document.querySelectorAll('.seg-btn').forEach(b => {
  b.addEventListener('click', () => setPriceMode(b.dataset.mode));
});
// Aplica modo persistido ao carregar
document.addEventListener('DOMContentLoaded', () => setPriceMode(priceMode));

function fmtTimeAgo(iso) {
  if (!iso) return '—';
  const ms = Date.now() - new Date(iso).getTime();
  const min = Math.floor(ms / 60000);
  if (min < 1) return 'agora';
  if (min < 60) return `${min}min atrás`;
  const h = Math.floor(min / 60);
  if (h < 24) return `${h}h atrás`;
  return `${Math.floor(h / 24)}d atrás`;
}

async function openHistory() {
  document.getElementById('modal-history').hidden = false;
  const tbody = document.querySelector('#history-table tbody');
  tbody.innerHTML = '<tr><td colspan="8" style="color:var(--muted); text-align:center;">Carregando...</td></tr>';
  try {
    const r = await fetch('/api/signals_history?limit=100', { cache: 'no-store' });
    const d = await r.json();
    if (!d.signals.length) {
      tbody.innerHTML = '<tr><td colspan="8" style="color:var(--muted); text-align:center;">Sem decisões registradas.</td></tr>';
      return;
    }
    tbody.innerHTML = d.signals.map(s => {
      const horizon = s.intended_horizon ? `<span style="color:var(--accent);">${s.intended_horizon}</span>` : '—';
      const conf = s.confidence != null ? `${(s.confidence * 100).toFixed(0)}%` : '—';
      const price = s.current_price != null ? Number(s.current_price).toFixed(5) : '—';
      const lot = s.lot != null ? s.lot : '—';
      const lat = s.llm_latency_ms != null ? `${s.llm_latency_ms}ms` : '—';
      const actionColor = COLORS[s.action] || '#c9d1d9';
      return `
        <tr>
          <td title="${s.created_at}">${fmtTimeAgo(s.created_at)}</td>
          <td><span style="color:${actionColor}; font-weight:600;">${s.action}</span></td>
          <td>${horizon}</td>
          <td class="num">${conf}</td>
          <td class="num">${price}</td>
          <td class="num">${lot}</td>
          <td class="num">${lat}</td>
          <td class="reason-cell">${s.reasoning ?? ''}</td>
        </tr>`;
    }).join('');
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="8" style="color:var(--red); text-align:center;">Erro: ${e.message}</td></tr>`;
  }
}

function closeHistory() {
  document.getElementById('modal-history').hidden = true;
}

document.getElementById('btn-history').addEventListener('click', openHistory);
document.getElementById('btn-close-history').addEventListener('click', closeHistory);
document.getElementById('modal-history').addEventListener('click', e => {
  if (e.target.id === 'modal-history') closeHistory();
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeHistory();
});

async function refreshPriceChart() {
  try {
    const r = await fetch('/api/price_chart?hours=24', { cache: 'no-store' });
    const d = await r.json();
    renderPriceChart(d);
  } catch (e) {
    console.error('price chart erro:', e);
  }
}

async function refreshCalibration() {
  try {
    const r = await fetch('/api/calibration', { cache: 'no-store' });
    const d = await r.json();
    renderCalibration(d);
  } catch (e) {
    console.error('calibration erro:', e);
  }
}

async function refresh() {
  try {
    const r = await fetch('/api/dashboard', { cache: 'no-store' });
    const d = await r.json();
    renderKPIs(d.summary);
    renderLast(d.summary);
    renderFallbackAlert(d.summary);
    renderPositions(d.summary.open_positions);
    renderEquity(d.equity_curve);
    renderDoughnut('actions', d.actions);
    renderDoughnut('closes', d.close_reasons);
    renderDurations(d.durations);
    renderLatency(d.latency);
    renderHorizons(d.horizons);
    renderRegimes(d.regimes);
  } catch (e) {
    document.getElementById('subtitle').textContent = 'erro: ' + e.message;
  }
}

refresh();
refreshPriceChart();
refreshCalibration();
setInterval(refresh, 10000);
// Gráfico de preços e calibração atualizam menos frequentemente
setInterval(refreshPriceChart, 60000);
setInterval(refreshCalibration, 30000);
</script>
</body>
</html>
"""

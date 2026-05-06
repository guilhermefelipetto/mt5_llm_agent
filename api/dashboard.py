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
    open_pos = last.get("position_at_decision") if last else None

    pnl_money = round(sum(t.get("pnl_money", 0.0) for t in trades), 2)
    pnl_pips = round(sum(t.get("pnl_pips", 0.0) for t in trades), 1)
    wins = sum(1 for t in trades if t.get("pnl_money", 0) > 0)

    return {
        "model_version": settings.model_version,
        "symbol": settings.symbol,
        "last_action": (last or {}).get("action"),
        "last_confidence": (last or {}).get("confidence"),
        "last_reasoning": (last or {}).get("reasoning"),
        "last_at": (last or {}).get("created_at"),
        "last_lot": (last or {}).get("lot"),
        "last_equity": (last or {}).get("equity"),
        "circuit_state": (last or {}).get("circuit_state"),
        "open_position": open_pos,
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
        "durations": _duration_buckets(trades),
    })


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
  .subtitle { color: var(--muted); font-size: 12px; margin-bottom: 20px; }
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

<h1>MT5 LLM Agent</h1>
<div class="subtitle" id="subtitle">carregando...</div>

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
    <h2>Win rate por regime estatístico (Hurst 1h)</h2>
    <table id="regimes">
      <thead><tr><th>Regime</th><th>Trades</th><th>Wins</th><th>Win rate</th><th>P&amp;L</th></tr></thead>
      <tbody></tbody>
    </table>
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
  document.getElementById('subtitle').textContent =
    `${s.symbol} · ${s.model_version} · atualizado ${new Date().toLocaleTimeString()}`;
  const html = `
    <div class="kpi"><div class="l">Sinais</div><div class="v">${s.total_signals}</div></div>
    <div class="kpi"><div class="l">Trades fechados</div><div class="v">${s.total_trades}</div></div>
    <div class="kpi"><div class="l">Win rate</div><div class="v">${s.win_rate_pct ?? '-'}${s.win_rate_pct != null ? '%' : ''}</div></div>
    <div class="kpi"><div class="l">P&L acumulado</div><div class="v">${fmtPnL(s.total_pnl_money, ' $')}<br><span style="font-size:11px">${fmtPnL(s.total_pnl_pips, ' pips')}</span></div></div>
  `;
  document.getElementById('kpis').innerHTML = html;
}

function renderLast(s) {
  const pos = s.open_position;
  const posLine = pos
    ? `Posição viva: <strong>${pos.side}</strong> @ ${pos.entry_price} · ${fmtPnL(pos.pnl_pips, ' pips')} · há ${pos.age_minutes.toFixed(0)}min`
    : 'Sem posição aberta';

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
    <div><span class="a">${s.last_action ?? '-'}</span> · confiança ${s.last_confidence != null ? (s.last_confidence * 100).toFixed(0) + '%' : '-'} · ${s.last_at ? new Date(s.last_at).toLocaleString() : '-'}</div>
    <div class="reason">"${s.last_reasoning ?? ''}"</div>
    <div style="margin-top: 8px;">${posLine}</div>
    ${sizingLine}
    ${circuitLine}
  `;
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

async function refresh() {
  try {
    const r = await fetch('/api/dashboard', { cache: 'no-store' });
    const d = await r.json();
    renderKPIs(d.summary);
    renderLast(d.summary);
    renderEquity(d.equity_curve);
    renderDoughnut('actions', d.actions);
    renderDoughnut('closes', d.close_reasons);
    renderDurations(d.durations);
    renderLatency(d.latency);
    renderRegimes(d.regimes);
  } catch (e) {
    document.getElementById('subtitle').textContent = 'erro: ' + e.message;
  }
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""

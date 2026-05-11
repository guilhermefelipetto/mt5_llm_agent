import os
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Perfis de horizonte (v1.7)
#
# Cada trade carrega um `intended_horizon` enum, escolhido pelo LLM no momento
# da abertura. O horizonte parametriza simultaneamente:
#   - sl_mult, tp_mult: multiplicadores de σ (vol prevista) para SL/TP
#   - max_age_hours:    time exit específico do horizonte
#   - cadence_s:        cadência de revisita quando há trade desse horizonte aberto
#
# Quanto maior o horizonte, mais frouxo o stop, mais paciência no time exit
# e mais lenta a cadência de análise.
# ---------------------------------------------------------------------------
HORIZON_PROFILES: dict[str, dict] = {
    "scalp":    {"sl_mult": 1.0, "tp_mult": 1.5, "max_age_hours": 4,   "cadence_s": 90,  "dominant_tfs": "M5/M1"},
    "intraday": {"sl_mult": 1.5, "tp_mult": 2.5, "max_age_hours": 24,  "cadence_s": 180, "dominant_tfs": "H1/M30"},
    "swing":    {"sl_mult": 2.5, "tp_mult": 5.0, "max_age_hours": 336, "cadence_s": 600, "dominant_tfs": "D1/4H"},
}
HORIZONS: tuple[str, ...] = tuple(HORIZON_PROFILES.keys())
DEFAULT_HORIZON: str = "intraday"


class Settings:
    # Versionamento - usado para filtrar logs por iteração do agente
    model_version: str = os.getenv("MODEL_VERSION", "v1.8.1")

    # LLM
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Trading
    symbol: str = os.getenv("SYMBOL", "EURUSD")
    yf_symbol: str = os.getenv("YF_SYMBOL", "EURUSD=X")

    # Analysis
    # analysis_interval = cadência usada quando NÃO há posições abertas (flat).
    # Quando há, a cadência é dinâmica: min(cadence_s) entre as posições ativas.
    analysis_interval: int = int(os.getenv("ANALYSIS_INTERVAL", "300"))
    signal_ttl: int = int(os.getenv("SIGNAL_TTL", "300"))

    # Multi-position (v1.7) - teto de posições simultâneas. Regra adicional:
    # uma posição por (lado, horizonte) — não dobra aposta no mesmo setup.
    # ATENÇÃO: risco máximo agregado = MAX_POSITIONS × RISK_PER_TRADE_PCT.
    max_positions: int = int(os.getenv("MAX_POSITIONS", "3"))

    # Risk
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.6"))

    # Position sizing dinâmico (v1.6) - lot calculado por
    # risco_alvo × confiança / (pip_value × distância_SL).
    risk_per_trade_pct: float = float(os.getenv("RISK_PER_TRADE_PCT", "1.0"))
    pip_value_per_lot: float = float(os.getenv("PIP_VALUE_PER_LOT", "10.0"))  # EURUSD/USD
    min_lot: float = float(os.getenv("MIN_LOT", "0.01"))
    max_lot: float = float(os.getenv("MAX_LOT", "1.0"))
    lot_step: float = float(os.getenv("LOT_STEP", "0.01"))
    # Equity de fallback quando MT5 não fornece (yfinance ou bridge offline).
    account_equity_fallback: float = float(os.getenv("ACCOUNT_EQUITY_FALLBACK", "10000.0"))

    # Circuit breaker (v1.6) - desabilita novas aberturas no dia se algum
    # limite for excedido. CLOSE/TIGHTEN_STOP nunca são bloqueados.
    daily_drawdown_pct: float = float(os.getenv("DAILY_DRAWDOWN_PCT", "2.0"))
    max_consecutive_losses: int = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "3"))

    # TIGHTEN_STOP guardrails (v1.8.1) - evitam aperto prematuro de SL.
    # min_tighten_progress: fração [0,1] do caminho entry->TP que o trade
    #   precisa ter percorrido antes de permitir TIGHTEN_STOP.
    # min_trail_buffer: fração [0,1] do lucro acumulado que o novo SL deve
    #   deixar como margem (impede colar SL no preço atual).
    min_tighten_progress: float = float(os.getenv("MIN_TIGHTEN_PROGRESS", "0.5"))
    min_trail_buffer: float = float(os.getenv("MIN_TRAIL_BUFFER", "0.3"))

    # Volatility estimator - garch | har_rv | atr
    vol_estimator: str = os.getenv("VOL_ESTIMATOR", "garch")
    pip_size: float = float(os.getenv("PIP_SIZE", "0.0001"))  # EURUSD: 0.0001 | USDJPY: 0.01

    # Fonte de dados - mt5 | yfinance
    data_source: str = os.getenv("DATA_SOURCE", "yfinance")

    # MT5 bridge (Linux/Wine) - host e porta do mt5linux proxy
    mt5_host: str = os.getenv("MT5_HOST", "localhost")
    mt5_port: int = int(os.getenv("MT5_PORT", "18812"))


settings = Settings()

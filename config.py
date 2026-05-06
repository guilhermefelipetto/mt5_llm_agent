import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Versionamento - usado para filtrar logs por iteração do agente
    model_version: str = os.getenv("MODEL_VERSION", "v1.6.0")

    # LLM
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Trading
    symbol: str = os.getenv("SYMBOL", "EURUSD")
    yf_symbol: str = os.getenv("YF_SYMBOL", "EURUSD=X")

    # Analysis
    analysis_interval: int = int(os.getenv("ANALYSIS_INTERVAL", "300"))  # seconds
    signal_ttl: int = int(os.getenv("SIGNAL_TTL", "300"))

    # Risk
    atr_sl_multiplier: float = float(os.getenv("ATR_SL_MULTIPLIER", "1.5"))
    atr_tp_multiplier: float = float(os.getenv("ATR_TP_MULTIPLIER", "2.5"))
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

    # Time-based exit (v1.6) - força CLOSE em trades pendurados há mais
    # tempo que o limite. 0 desabilita.
    max_trade_age_hours: float = float(os.getenv("MAX_TRADE_AGE_HOURS", "24"))

    # Volatility estimator - garch | har_rv | atr
    vol_estimator: str = os.getenv("VOL_ESTIMATOR", "garch")
    pip_size: float = float(os.getenv("PIP_SIZE", "0.0001"))  # EURUSD: 0.0001 | USDJPY: 0.01

    # Fonte de dados - mt5 | yfinance
    data_source: str = os.getenv("DATA_SOURCE", "yfinance")

    # MT5 bridge (Linux/Wine) - host e porta do mt5linux proxy
    mt5_host: str = os.getenv("MT5_HOST", "localhost")
    mt5_port: int = int(os.getenv("MT5_PORT", "18812"))


settings = Settings()

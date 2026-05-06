import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, min_periods=slow).mean()
    line = ema_fast - ema_slow
    signal_line = line.ewm(span=signal, min_periods=signal).mean()
    return line, signal_line, line - signal_line


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def bollinger(
    close: pd.Series, period: int = 20, std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(period).mean()
    dev = close.rolling(period).std()
    upper = ma + std * dev
    lower = ma - std * dev
    position = (close - lower) / (upper - lower + 1e-10)
    return upper, ma, lower, position


def ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, min_periods=period).mean()


def compute(df: pd.DataFrame) -> dict:
    """Retorna todos os indicadores para o último candle do DataFrame."""
    if len(df) < 50:
        return {}

    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    rsi_series = rsi(close)
    _, _, macd_hist = macd(close)
    atr_val = atr(high, low, close).iloc[-1]
    _, _, _, bb_pos = bollinger(close)
    ema20 = ema(close, 20)
    ema50 = ema(close, 50)

    price = float(close.iloc[-1])
    rsi_val = float(rsi_series.iloc[-1])
    hist_val = float(macd_hist.iloc[-1])
    hist_prev = float(macd_hist.iloc[-2])

    # Tendência pela EMA20/50
    if ema20.iloc[-1] > ema50.iloc[-1] and ema20.iloc[-2] <= ema50.iloc[-2]:
        trend = "cruzamento_alta"
    elif ema20.iloc[-1] < ema50.iloc[-1] and ema20.iloc[-2] >= ema50.iloc[-2]:
        trend = "cruzamento_baixa"
    elif ema20.iloc[-1] > ema50.iloc[-1]:
        trend = "alta"
    else:
        trend = "baixa"

    # RSI
    if rsi_val >= 70:
        rsi_signal = "sobrecomprado"
    elif rsi_val <= 30:
        rsi_signal = "sobrevendido"
    else:
        rsi_signal = "neutro"

    # MACD
    if hist_val > 0 and hist_prev <= 0:
        macd_signal = "cruzamento_alta"
    elif hist_val < 0 and hist_prev >= 0:
        macd_signal = "cruzamento_baixa"
    elif hist_val > 0:
        macd_signal = "bullish"
    else:
        macd_signal = "bearish"

    ret_5 = (price / float(close.iloc[-6]) - 1) * 100 if len(close) > 5 else 0.0
    ret_20 = (price / float(close.iloc[-21]) - 1) * 100 if len(close) > 20 else 0.0

    return {
        "price": round(price, 5),
        "atr": round(float(atr_val), 5),
        "atr_pips": round(float(atr_val) * 10000, 1),
        "rsi": round(rsi_val, 1),
        "rsi_signal": rsi_signal,
        "macd_signal": macd_signal,
        "macd_hist": round(hist_val, 6),
        "bb_position": round(float(bb_pos.iloc[-1]), 2),
        "trend": trend,
        "ret_5": round(ret_5, 4),
        "ret_20": round(ret_20, 4),
    }

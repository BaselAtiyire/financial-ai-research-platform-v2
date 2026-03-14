import pandas as pd
import yfinance as yf


def get_market_snapshot(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    fast_info = getattr(ticker, "fast_info", {}) or {}

    price = fast_info.get("lastPrice")
    previous_close = fast_info.get("previousClose")
    day_high = fast_info.get("dayHigh")
    day_low = fast_info.get("dayLow")
    market_cap = fast_info.get("marketCap")

    change = None
    change_pct = None

    if price is not None and previous_close not in (None, 0):
        change = price - previous_close
        change_pct = (change / previous_close) * 100

    return {
        "symbol": symbol.upper(),
        "price": price,
        "previous_close": previous_close,
        "change": change,
        "change_pct": change_pct,
        "day_high": day_high,
        "day_low": day_low,
        "market_cap": market_cap,
    }


def get_price_history(symbol: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    return df


def compare_market_performance(symbols: list[str], period: str = "6mo") -> pd.DataFrame:
    frames = []

    for symbol in symbols:
        hist = get_price_history(symbol, period=period, interval="1d")
        if hist.empty or "Close" not in hist.columns:
            continue

        hist = hist[["Date", "Close"]].copy()
        hist["Symbol"] = symbol.upper()

        first_close = hist["Close"].iloc[0]
        if first_close not in (None, 0):
            hist["Normalized"] = (hist["Close"] / first_close) * 100

        frames.append(hist)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def company_name_to_ticker(company_name: str) -> str | None:
    if not company_name:
        return None

    name = company_name.strip().lower()

    ticker_map = {
        "apple": "AAPL",
        "apple inc.": "AAPL",
        "tesla": "TSLA",
        "tesla, inc.": "TSLA",
        "tesla inc.": "TSLA",
        "amazon": "AMZN",
        "amazon.com": "AMZN",
        "amazon.com, inc.": "AMZN",
        "amazon inc.": "AMZN",
        "microsoft": "MSFT",
        "microsoft corporation": "MSFT",
        "alphabet": "GOOGL",
        "alphabet inc.": "GOOGL",
        "google": "GOOGL",
        "meta": "META",
        "meta platforms": "META",
        "meta platforms, inc.": "META",
        "nvidia": "NVDA",
        "nvidia corporation": "NVDA",
    }

    if name in ticker_map:
        return ticker_map[name]

    for company, ticker in ticker_map.items():
        if company in name:
            return ticker

    return None
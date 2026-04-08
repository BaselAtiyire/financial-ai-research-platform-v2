import pandas as pd
import streamlit as st
import yfinance as yf

# ── Ticker map ────────────────────────────────────────────────────────────────
TICKER_MAP: dict[str, str] = {
    "apple": "AAPL", "apple inc": "AAPL", "apple inc.": "AAPL",
    "tesla": "TSLA", "tesla inc": "TSLA", "tesla, inc.": "TSLA",
    "amazon": "AMZN", "amazon.com": "AMZN", "amazon.com, inc.": "AMZN",
    "microsoft": "MSFT", "microsoft corporation": "MSFT",
    "alphabet": "GOOGL", "alphabet inc": "GOOGL", "google": "GOOGL",
    "meta": "META", "meta platforms": "META", "meta platforms, inc.": "META",
    "nvidia": "NVDA", "nvidia corporation": "NVDA",
    "netflix": "NFLX", "netflix, inc.": "NFLX",
    "salesforce": "CRM", "salesforce, inc.": "CRM",
    "berkshire hathaway": "BRK-B",
    "jpmorgan": "JPM", "jpmorgan chase": "JPM", "jp morgan": "JPM",
    "johnson & johnson": "JNJ", "johnson and johnson": "JNJ",
    "visa": "V", "mastercard": "MA",
    "exxon": "XOM", "exxonmobil": "XOM",
    "walmart": "WMT", "coca-cola": "KO", "coca cola": "KO",
    "intel": "INTC",
    "amd": "AMD", "advanced micro devices": "AMD",
    "paypal": "PYPL", "adobe": "ADBE",
    "disney": "DIS", "the walt disney company": "DIS",
    "boeing": "BA", "airbus": "EADSY",
    "uber": "UBER", "lyft": "LYFT",
    "spotify": "SPOT", "twitter": "X",
    "snapchat": "SNAP", "snap": "SNAP",
    "palantir": "PLTR", "palantir technologies": "PLTR",
    "shopify": "SHOP", "square": "SQ", "block": "SQ",
    "coinbase": "COIN", "robinhood": "HOOD",
    "oracle": "ORCL", "ibm": "IBM",
    "qualcomm": "QCOM", "broadcom": "AVGO",
    "taiwan semiconductor": "TSM", "tsmc": "TSM",
    "samsung": "SSNLF", "sony": "SONY",
    "ford": "F", "general motors": "GM", "gm": "GM",
    "chevron": "CVX", "shell": "SHEL",
    "pfizer": "PFE", "moderna": "MRNA",
    "johnson & johnson": "JNJ", "abbvie": "ABBV",
    "unitedhealth": "UNH", "unitedhealth group": "UNH",
    "goldman sachs": "GS", "morgan stanley": "MS",
    "bank of america": "BAC", "citigroup": "C", "citi": "C",
    "wells fargo": "WFC",
    "caterpillar": "CAT", "deere": "DE", "john deere": "DE",
    "3m": "MMM", "honeywell": "HON",
    "starbucks": "SBUX", "mcdonald's": "MCD", "mcdonalds": "MCD",
    "nike": "NKE", "adidas": "ADDYY",
    "at&t": "T", "verizon": "VZ", "t-mobile": "TMUS",
}


@st.cache_data(ttl=300)  # cache for 5 minutes
def get_market_snapshot(symbol: str) -> dict:
    ticker = yf.Ticker(symbol)
    fast_info = getattr(ticker, "fast_info", {}) or {}

    price = fast_info.get("lastPrice")
    previous_close = fast_info.get("previousClose")
    day_high = fast_info.get("dayHigh")
    day_low = fast_info.get("dayLow")
    market_cap = fast_info.get("marketCap")
    volume = fast_info.get("lastVolume")
    fifty_two_week_high = fast_info.get("yearHigh")
    fifty_two_week_low = fast_info.get("yearLow")

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
        "volume": volume,
        "fifty_two_week_high": fifty_two_week_high,
        "fifty_two_week_low": fifty_two_week_low,
    }


@st.cache_data(ttl=3600)  # cache for 1 hour
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

    # Add moving averages
    if "Close" in df.columns and len(df) > 20:
        df["MA20"] = df["Close"].rolling(window=20).mean()
    if "Close" in df.columns and len(df) > 50:
        df["MA50"] = df["Close"].rolling(window=50).mean()

    return df


@st.cache_data(ttl=3600)
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
    name = re.sub(r"[,\.]+$", "", name).strip()

    # Exact match first
    if name in TICKER_MAP:
        return TICKER_MAP[name]

    # Partial match
    for company, ticker in TICKER_MAP.items():
        if company in name or name in company:
            return ticker

    return None


import re

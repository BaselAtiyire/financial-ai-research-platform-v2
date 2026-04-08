def generate_recommendation(valuation: dict | None, sentiment: dict | None) -> str:
    if not valuation:
        return "INSUFFICIENT DATA"

    signal = valuation.get("signal", "")
    sentiment_label = "Neutral"
    if sentiment:
        sentiment_label = sentiment.get("sentiment_label", "Neutral")

    if signal == "Potentially Undervalued" and sentiment_label in ["Bullish", "Neutral"]:
        return "BUY"
    if signal == "Potentially Overvalued" and sentiment_label in ["Bearish", "Neutral"]:
        return "SELL"
    return "HOLD"

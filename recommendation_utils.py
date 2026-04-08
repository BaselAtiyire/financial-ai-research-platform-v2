def generate_recommendation(valuation: dict | None, sentiment: dict | None) -> str:
    """
    Generate a BUY / HOLD / SELL recommendation using valuation signal
    and sentiment label together.
    """
    if not valuation:
        return "INSUFFICIENT DATA"

    signal = valuation.get("signal", "")
    methods_used = valuation.get("methods_used", 0)
    sentiment_label = "Neutral"
    if sentiment:
        sentiment_label = sentiment.get("sentiment_label", "Neutral")

    # Need at least 1 valuation method to give a recommendation
    if methods_used == 0:
        return "INSUFFICIENT DATA"

    bullish_sentiment = sentiment_label in ["Bullish", "Mildly Bullish", "Neutral"]
    bearish_sentiment = sentiment_label in ["Bearish", "Mildly Bearish"]

    if signal in ["Potentially Undervalued", "Slight Upside"] and bullish_sentiment:
        return "BUY"
    if signal == "Potentially Undervalued" and bearish_sentiment:
        return "HOLD"
    if signal in ["Potentially Overvalued", "Slight Downside"] and bearish_sentiment:
        return "SELL"
    if signal == "Potentially Overvalued" and bullish_sentiment:
        return "HOLD"

    return "HOLD"

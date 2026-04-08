import re
from collections import Counter

POSITIVE_WORDS = {
    "growth", "strong", "improved", "record", "opportunity", "confidence",
    "accelerate", "expansion", "profitability", "beat", "outperform",
    "momentum", "efficient", "innovation", "resilient", "optimistic",
}

NEGATIVE_WORDS = {
    "risk", "decline", "weakness", "pressure", "uncertainty", "slowdown",
    "loss", "challenge", "volatile", "headwind", "miss", "underperform",
    "constraint", "inflation", "debt", "disruption", "cautious",
}

GUIDANCE_WORDS = {
    "guidance", "outlook", "forecast", "expect", "project", "anticipate",
    "target", "next quarter", "next year",
}

RISK_WORDS = {
    "risk", "uncertainty", "headwind", "competition", "inflation",
    "regulation", "supply chain", "macro", "slowdown", "currency",
}

STOP_WORDS = {
    "that", "this", "with", "from", "have", "were", "been", "their",
    "about", "into", "than", "will", "they", "year", "quarter",
}


def normalize_text(text: str) -> list[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def count_keyword_hits(tokens: list[str], keywords: set[str]) -> int:
    joined = " ".join(tokens)
    count = 0
    for kw in keywords:
        if " " in kw:
            count += joined.count(kw)
        else:
            count += tokens.count(kw)
    return count


def analyze_earnings_sentiment(text: str) -> dict:
    tokens = normalize_text(text)
    if not tokens:
        return {
            "sentiment_score": 0,
            "sentiment_label": "Neutral",
            "positive_hits": 0,
            "negative_hits": 0,
            "guidance_hits": 0,
            "risk_hits": 0,
            "token_count": 0,
            "top_terms": {},
        }

    positive_hits = count_keyword_hits(tokens, POSITIVE_WORDS)
    negative_hits = count_keyword_hits(tokens, NEGATIVE_WORDS)
    guidance_hits = count_keyword_hits(tokens, GUIDANCE_WORDS)
    risk_hits = count_keyword_hits(tokens, RISK_WORDS)

    sentiment_score = positive_hits - negative_hits

    if sentiment_score > 2:
        sentiment_label = "Bullish"
    elif sentiment_score < -2:
        sentiment_label = "Bearish"
    else:
        sentiment_label = "Neutral"

    filtered_terms = [
        t for t in tokens
        if len(t) > 3 and t not in STOP_WORDS
    ]
    top_terms = dict(Counter(filtered_terms).most_common(10))

    return {
        "sentiment_score": sentiment_score,
        "sentiment_label": sentiment_label,
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "guidance_hits": guidance_hits,
        "risk_hits": risk_hits,
        "token_count": len(tokens),
        "top_terms": top_terms,
    }

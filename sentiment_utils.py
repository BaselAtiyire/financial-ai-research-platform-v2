import re
from collections import Counter

POSITIVE_WORDS = {
    "growth", "strong", "improved", "record", "opportunity", "confidence",
    "accelerate", "expansion", "profitability", "beat", "outperform",
    "momentum", "efficient", "innovation", "resilient", "optimistic",
    "exceed", "surpass", "robust", "solid", "outstanding", "exceptional",
    "favorable", "upside", "breakthrough", "leadership", "gain", "increase",
    "strengthen", "advance", "positive", "achieve", "success", "milestone",
    "recovery", "rebound", "margin expansion", "market share",
}

NEGATIVE_WORDS = {
    "risk", "decline", "weakness", "pressure", "uncertainty", "slowdown",
    "loss", "challenge", "volatile", "headwind", "miss", "underperform",
    "constraint", "inflation", "debt", "disruption", "cautious",
    "concern", "difficult", "deteriorate", "decrease", "drop", "fall",
    "negative", "adverse", "below", "shortfall", "impairment", "write-off",
    "restructure", "layoff", "downgrade", "miss", "weak", "soften",
    "contraction", "compression", "unfavorable", "disappointing",
}

GUIDANCE_WORDS = {
    "guidance", "outlook", "forecast", "expect", "project", "anticipate",
    "target", "next quarter", "next year", "full year", "fiscal year",
    "going forward", "remainder", "pipeline", "backlog",
}

RISK_WORDS = {
    "risk", "uncertainty", "headwind", "competition", "inflation",
    "regulation", "supply chain", "macro", "slowdown", "currency",
    "geopolitical", "recession", "tariff", "litigation", "cybersecurity",
    "interest rate", "fx", "foreign exchange", "commodity", "disruption",
}

STOP_WORDS = {
    "that", "this", "with", "from", "have", "were", "been", "their",
    "about", "into", "than", "will", "they", "year", "quarter",
    "the", "and", "for", "are", "was", "our", "has", "not", "but",
    "also", "more", "said", "which", "would", "could", "should",
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


def compute_sentiment_score(positive: int, negative: int, token_count: int) -> float:
    """Normalize sentiment score to [-1, 1] range."""
    if token_count == 0:
        return 0.0
    raw = positive - negative
    # Normalize by document length
    normalized = raw / max(token_count / 100, 1)
    return round(max(-1.0, min(1.0, normalized)), 3)


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
            "normalized_score": 0.0,
        }

    positive_hits = count_keyword_hits(tokens, POSITIVE_WORDS)
    negative_hits = count_keyword_hits(tokens, NEGATIVE_WORDS)
    guidance_hits = count_keyword_hits(tokens, GUIDANCE_WORDS)
    risk_hits = count_keyword_hits(tokens, RISK_WORDS)

    raw_score = positive_hits - negative_hits
    normalized_score = compute_sentiment_score(positive_hits, negative_hits, len(tokens))

    if raw_score > 3:
        sentiment_label = "Bullish"
    elif raw_score > 1:
        sentiment_label = "Mildly Bullish"
    elif raw_score < -3:
        sentiment_label = "Bearish"
    elif raw_score < -1:
        sentiment_label = "Mildly Bearish"
    else:
        sentiment_label = "Neutral"

    filtered_terms = [
        t for t in tokens
        if len(t) > 4 and t not in STOP_WORDS
    ]
    top_terms = dict(Counter(filtered_terms).most_common(15))

    return {
        "sentiment_score": raw_score,
        "normalized_score": normalized_score,
        "sentiment_label": sentiment_label,
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "guidance_hits": guidance_hits,
        "risk_hits": risk_hits,
        "token_count": len(tokens),
        "top_terms": top_terms,
    }

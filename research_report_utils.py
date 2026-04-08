from datetime import date


def _fmt(value, suffix="", prefix="", decimals=2) -> str:
    """Safe number formatter."""
    if value is None:
        return "N/A"
    try:
        return f"{prefix}{float(value):,.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def generate_equity_research_report(
    company_name: str,
    ticker: str,
    financial_metrics: list[dict],
    market_snapshot: dict | None,
    sentiment_result: dict | None = None,
    valuation_result: dict | None = None,
) -> str:
    today = date.today().strftime("%B %d, %Y")
    report = []

    report.append(f"# 📊 Equity Research Report")
    report.append(f"## {company_name} ({ticker})")
    report.append(f"*Generated: {today} | AI-Assisted Research | For Educational Use Only*")
    report.append("---")

    # Market Snapshot
    if market_snapshot:
        price = market_snapshot.get("price")
        market_cap = market_snapshot.get("market_cap")
        change_pct = market_snapshot.get("change_pct")
        day_high = market_snapshot.get("day_high")
        day_low = market_snapshot.get("day_low")
        week_high = market_snapshot.get("fifty_two_week_high")
        week_low = market_snapshot.get("fifty_two_week_low")

        report.append("## 📈 Market Snapshot")
        report.append(f"| Metric | Value |")
        report.append(f"|--------|-------|")
        report.append(f"| Current Price | {_fmt(price, prefix='$')} |")
        report.append(f"| Daily Change | {_fmt(change_pct, suffix='%')} |")
        report.append(f"| Day Range | {_fmt(day_low, prefix='$')} – {_fmt(day_high, prefix='$')} |")
        report.append(f"| 52-Week Range | {_fmt(week_low, prefix='$')} – {_fmt(week_high, prefix='$')} |")
        report.append(f"| Market Cap | {_fmt(market_cap, prefix='$', decimals=0)} |")

    # Key Financial Metrics
    if financial_metrics:
        report.append("## 💰 Key Financial Metrics")
        report.append("| Metric | Value | Period | Source |")
        report.append("|--------|-------|--------|--------|")
        for m in financial_metrics[:12]:
            metric = m.get("metric", "Unknown")
            value = m.get("value", "N/A")
            period = m.get("period", "N/A")
            doc = m.get("document_name", "Unknown")
            report.append(f"| {metric} | {value} | {period} | {doc} |")

    # Valuation Summary
    if valuation_result:
        report.append("## 🔢 AI Valuation Analysis")
        signal = valuation_result.get("signal", "N/A")
        gap = valuation_result.get("valuation_gap_pct")
        methods = valuation_result.get("methods_used", 0)

        signal_emoji = {"Potentially Undervalued": "🟢", "Potentially Overvalued": "🔴",
                        "Fairly Valued": "🟡", "Slight Upside": "🟢", "Slight Downside": "🟠"}.get(signal, "⚪")

        report.append(f"**Valuation Signal: {signal_emoji} {signal}** (based on {methods} method(s))")
        report.append("")
        report.append("| Method | Estimated Value |")
        report.append("|--------|----------------|")
        report.append(f"| P/E Fair Value (20x) | {_fmt(valuation_result.get('pe_fair_value'), prefix='$', decimals=0)} |")
        report.append(f"| Revenue Multiple (3x) | {_fmt(valuation_result.get('revenue_multiple_value'), prefix='$', decimals=0)} |")
        report.append(f"| EV/EBITDA (12x) | {_fmt(valuation_result.get('ev_ebitda_value'), prefix='$', decimals=0)} |")
        report.append(f"| DCF Value | {_fmt(valuation_result.get('dcf_value'), prefix='$', decimals=0)} |")
        report.append(f"| **Avg. Fair Value** | **{_fmt(valuation_result.get('estimated_fair_value'), prefix='$', decimals=0)}** |")
        report.append(f"| Current Market Cap | {_fmt(valuation_result.get('market_cap'), prefix='$', decimals=0)} |")
        report.append(f"| Valuation Gap | {_fmt(gap, suffix='%')} |")

    # Sentiment
    if sentiment_result:
        report.append("## 🎙️ Earnings Call Sentiment")
        label = sentiment_result.get("sentiment_label", "N/A")
        score = sentiment_result.get("sentiment_score", 0)
        label_emoji = {"Bullish": "🟢", "Mildly Bullish": "🟢", "Neutral": "🟡",
                       "Mildly Bearish": "🟠", "Bearish": "🔴"}.get(label, "⚪")

        report.append(f"**Overall Tone: {label_emoji} {label}** (raw score: {score})")
        report.append("")
        report.append(f"- ✅ Positive signals: **{sentiment_result.get('positive_hits', 0)}**")
        report.append(f"- ❌ Negative signals: **{sentiment_result.get('negative_hits', 0)}**")
        report.append(f"- 🎯 Guidance mentions: **{sentiment_result.get('guidance_hits', 0)}**")
        report.append(f"- ⚠️ Risk mentions: **{sentiment_result.get('risk_hits', 0)}**")

    # Disclaimer
    report.append("---")
    report.append("## ⚠️ Disclaimer")
    report.append(
        "*This report is AI-generated using uploaded financial documents and publicly available "
        "market data. It is intended for educational and research purposes only and does NOT "
        "constitute investment advice. Always consult a qualified financial advisor before "
        "making investment decisions.*"
    )

    return "\n\n".join(report)

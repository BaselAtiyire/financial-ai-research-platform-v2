from research_report_utils import generate_equity_research_report
from valuation_utils import build_valuation_summary


def generate_investment_thesis(
    company_name: str,
    ticker: str,
    metrics: list[dict],
    market_snapshot: dict | None,
    sentiment_result: dict | None,
    valuation_result: dict | None,
) -> str:
    lines = []
    lines.append(f"## 📋 Investment Thesis: {company_name} ({ticker})")

    # Financial Performance
    lines.append("### Financial Performance")
    revenue_metric = next(
        (m for m in metrics if "revenue" in str(m.get("metric", "")).lower()), None
    )
    income_metric = next(
        (m for m in metrics if any(k in str(m.get("metric", "")).lower()
         for k in ["net income", "net profit", "earnings"])), None
    )
    ebitda_metric = next(
        (m for m in metrics if "ebitda" in str(m.get("metric", "")).lower()), None
    )

    if revenue_metric:
        lines.append(f"- **Revenue:** {revenue_metric.get('value')} ({revenue_metric.get('period', 'N/A')})")
    if income_metric:
        lines.append(f"- **Net Income:** {income_metric.get('value')} ({income_metric.get('period', 'N/A')})")
    if ebitda_metric:
        lines.append(f"- **EBITDA:** {ebitda_metric.get('value')} ({ebitda_metric.get('period', 'N/A')})")
    if not any([revenue_metric, income_metric, ebitda_metric]):
        lines.append("- Insufficient financial data from uploaded documents.")

    # Market Position
    if market_snapshot:
        price = market_snapshot.get("price")
        market_cap = market_snapshot.get("market_cap")
        change_pct = market_snapshot.get("change_pct")
        w52_high = market_snapshot.get("fifty_two_week_high")
        w52_low = market_snapshot.get("fifty_two_week_low")
        lines.append("### Market Position")
        lines.append(f"- **Current Price:** {'${:.2f}'.format(price) if price else 'N/A'}")
        lines.append(f"- **Market Cap:** {'${:,.0f}'.format(market_cap) if market_cap else 'N/A'}")
        lines.append(f"- **Daily Move:** {'{:.2f}%'.format(change_pct) if change_pct is not None else 'N/A'}")
        if w52_high and w52_low:
            lines.append(f"- **52-Week Range:** ${w52_low:.2f} – ${w52_high:.2f}")

    # Sentiment
    if sentiment_result:
        label = sentiment_result.get("sentiment_label", "Unknown")
        score = sentiment_result.get("sentiment_score", 0)
        lines.append("### Management Sentiment")
        lines.append(f"- **Tone:** {label} (score: {score})")
        lines.append(
            f"- Positive: {sentiment_result.get('positive_hits', 0)} | "
            f"Negative: {sentiment_result.get('negative_hits', 0)} | "
            f"Risk: {sentiment_result.get('risk_hits', 0)}"
        )

    # Valuation
    if valuation_result:
        signal = valuation_result.get("signal", "N/A")
        efv = valuation_result.get("estimated_fair_value")
        gap = valuation_result.get("valuation_gap_pct")
        methods = valuation_result.get("methods_used", 0)
        lines.append("### Valuation")
        lines.append(f"- **Signal:** {signal} ({methods} method(s) used)")
        lines.append(f"- **Fair Value Estimate:** {'${:,.0f}'.format(efv) if efv else 'N/A'}")
        lines.append(f"- **Valuation Gap:** {'{:.2f}%'.format(gap) if gap is not None else 'N/A'}")

    # Conclusion
    final_signal = _derive_conclusion(valuation_result, sentiment_result)
    lines.append("### AI Analyst Conclusion")
    lines.append(f"- **Overall View:** {final_signal}")
    lines.append("- *Based on uploaded documents, market data, sentiment, and valuation heuristics.*")

    return "\n\n".join(lines)


def _derive_conclusion(valuation_result, sentiment_result) -> str:
    if not valuation_result:
        return "Hold / Needs More Research"
    signal = valuation_result.get("signal", "")
    sentiment_label = (sentiment_result or {}).get("sentiment_label", "Neutral")

    if signal in ["Potentially Undervalued", "Slight Upside"] and sentiment_label in ["Bullish", "Mildly Bullish", "Neutral"]:
        return "🟢 Constructive / Potential Buy Candidate"
    if signal in ["Potentially Overvalued", "Slight Downside"] and sentiment_label in ["Bearish", "Mildly Bearish", "Neutral"]:
        return "🔴 Cautious / Potential Sell or Avoid"
    return "🟡 Neutral / Hold"


def run_financial_analyst_agent(
    company_name: str,
    ticker: str,
    metrics: list[dict],
    market_snapshot: dict | None,
    sentiment_result: dict | None,
) -> dict:
    valuation_result = None
    if metrics and market_snapshot:
        valuation_result = build_valuation_summary(metrics, market_snapshot)

    research_report = generate_equity_research_report(
        company_name=company_name,
        ticker=ticker,
        financial_metrics=metrics,
        market_snapshot=market_snapshot,
        sentiment_result=sentiment_result,
        valuation_result=valuation_result,
    )

    investment_thesis = generate_investment_thesis(
        company_name=company_name,
        ticker=ticker,
        metrics=metrics,
        market_snapshot=market_snapshot,
        sentiment_result=sentiment_result,
        valuation_result=valuation_result,
    )

    return {
        "valuation_result": valuation_result,
        "research_report": research_report,
        "investment_thesis": investment_thesis,
    }

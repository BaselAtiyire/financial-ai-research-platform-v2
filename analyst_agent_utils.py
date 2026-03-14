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
    thesis_lines = []

    thesis_lines.append(f"## Investment Thesis: {company_name} ({ticker})")

    if metrics:
        revenue_metric = next(
            (m for m in metrics if "revenue" in str(m.get("metric", "")).lower()),
            None,
        )
        income_metric = next(
            (
                m for m in metrics
                if "income" in str(m.get("metric", "")).lower()
                or "profit" in str(m.get("metric", "")).lower()
            ),
            None,
        )

        thesis_lines.append("### Financial Performance")
        if revenue_metric:
            thesis_lines.append(
                f"- Revenue signal: {revenue_metric.get('value')} during {revenue_metric.get('period')}."
            )
        if income_metric:
            thesis_lines.append(
                f"- Profitability signal: {income_metric.get('value')} during {income_metric.get('period')}."
            )

    if market_snapshot:
        thesis_lines.append("### Market Position")
        thesis_lines.append(f"- Current price: {market_snapshot.get('price')}")
        thesis_lines.append(f"- Market cap: {market_snapshot.get('market_cap')}")
        thesis_lines.append(f"- Daily move: {market_snapshot.get('change_pct')}%")

    if sentiment_result:
        thesis_lines.append("### Management / Earnings Sentiment")
        thesis_lines.append(
            f"- Sentiment appears **{sentiment_result.get('sentiment_label', 'Unknown')}** "
            f"with score {sentiment_result.get('sentiment_score', 0)}."
        )
        thesis_lines.append(
            f"- Positive mentions: {sentiment_result.get('positive_hits', 0)}, "
            f"Negative mentions: {sentiment_result.get('negative_hits', 0)}, "
            f"Risk mentions: {sentiment_result.get('risk_hits', 0)}."
        )

    if valuation_result:
        thesis_lines.append("### Valuation View")
        thesis_lines.append(
            f"- Estimated fair value: {valuation_result.get('estimated_fair_value')}"
        )
        thesis_lines.append(
            f"- Valuation gap: {valuation_result.get('valuation_gap_pct')}%"
        )
        thesis_lines.append(
            f"- Signal: **{valuation_result.get('signal', 'Insufficient Data')}**"
        )

    final_signal = "Hold / Needs More Research"
    if valuation_result:
        signal = valuation_result.get("signal")
        sentiment_label = (sentiment_result or {}).get("sentiment_label", "Neutral")

        if signal == "Potentially Undervalued" and sentiment_label in ["Bullish", "Neutral"]:
            final_signal = "Constructive / Potential Buy Candidate"
        elif signal == "Potentially Overvalued" and sentiment_label in ["Bearish", "Neutral"]:
            final_signal = "Cautious / Potential Sell or Avoid"
        else:
            final_signal = "Neutral / Hold"

    thesis_lines.append("### AI Analyst Conclusion")
    thesis_lines.append(f"- Overall view: **{final_signal}**.")
    thesis_lines.append(
        "- This thesis is AI-generated from uploaded documents, market data, sentiment signals, and valuation heuristics."
    )

    return "\n\n".join(thesis_lines)


def run_financial_analyst_agent(
    company_name: str,
    ticker: str,
    metrics: list[dict],
    market_snapshot: dict | None,
    sentiment_result: dict | None,
):
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
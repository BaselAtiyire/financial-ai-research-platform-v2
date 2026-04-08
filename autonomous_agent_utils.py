from analyst_agent_utils import run_financial_analyst_agent
from recommendation_utils import generate_recommendation
from risk_utils import detect_company_risk
from datetime import date


def run_autonomous_financial_agent(
    company_name: str,
    ticker: str,
    metrics: list[dict],
    market_snapshot: dict | None,
    sentiment_result: dict | None,
) -> dict:
    agent_output = run_financial_analyst_agent(
        company_name=company_name,
        ticker=ticker,
        metrics=metrics,
        market_snapshot=market_snapshot,
        sentiment_result=sentiment_result,
    )

    valuation_result = agent_output.get("valuation_result")
    research_report = agent_output.get("research_report")
    investment_thesis = agent_output.get("investment_thesis")

    risks = detect_company_risk(metrics, sentiment_result)
    recommendation = generate_recommendation(valuation_result, sentiment_result)
    today = date.today().strftime("%B %d, %Y")

    rec_emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(recommendation, "⚪")

    lines = []
    lines.append(f"# 🤖 Autonomous Financial Analyst Brief")
    lines.append(f"## {company_name} ({ticker})")
    lines.append(f"*Generated: {today}*")
    lines.append("---")

    lines.append(f"## Recommendation: {rec_emoji} {recommendation}")

    if valuation_result:
        efv = valuation_result.get("estimated_fair_value")
        gap = valuation_result.get("valuation_gap_pct")
        signal = valuation_result.get("signal", "N/A")
        methods = valuation_result.get("methods_used", 0)
        lines.append("## Valuation Summary")
        lines.append(f"- **Signal:** {signal} ({methods} method(s))")
        lines.append(f"- **Fair Value:** {'${:,.0f}'.format(efv) if efv else 'N/A'}")
        lines.append(f"- **Valuation Gap:** {'{:.2f}%'.format(gap) if gap is not None else 'N/A'}")

    if market_snapshot:
        price = market_snapshot.get("price")
        market_cap = market_snapshot.get("market_cap")
        lines.append("## Market Data")
        lines.append(f"- **Price:** {'${:.2f}'.format(price) if price else 'N/A'}")
        lines.append(f"- **Market Cap:** {'${:,.0f}'.format(market_cap) if market_cap else 'N/A'}")

    if sentiment_result:
        label = sentiment_result.get("sentiment_label", "N/A")
        lines.append("## Sentiment")
        lines.append(f"- **Label:** {label}")
        lines.append(f"- **Score:** {sentiment_result.get('sentiment_score', 0)}")
        lines.append(f"- **Risk Mentions:** {sentiment_result.get('risk_hits', 0)}")

    lines.append("## Risk Assessment")
    if risks:
        for risk in risks:
            lines.append(f"- {risk}")
    else:
        lines.append("- ✅ No major risk flags detected.")

    if investment_thesis:
        lines.append(investment_thesis)

    lines.append("---")
    lines.append(
        "> ⚠️ *This brief is AI-generated for educational/research purposes only. "
        "Not investment advice.*"
    )

    return {
        "recommendation": recommendation,
        "risks": risks,
        "valuation_result": valuation_result,
        "research_report": research_report,
        "investment_thesis": investment_thesis,
        "final_brief": "\n\n".join(lines),
    }

from analyst_agent_utils import run_financial_analyst_agent
from recommendation_utils import generate_recommendation
from risk_utils import detect_company_risk


def run_autonomous_financial_agent(
    company_name: str,
    ticker: str,
    metrics: list[dict],
    market_snapshot: dict | None,
    sentiment_result: dict | None,
):
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

    final_brief_lines = []
    final_brief_lines.append(f"# Autonomous Financial Analyst Brief: {company_name} ({ticker})")
    final_brief_lines.append(f"## Recommendation\n{recommendation}")

    if valuation_result:
        final_brief_lines.append("## Valuation Summary")
        final_brief_lines.append(f"- Signal: {valuation_result.get('signal')}")
        final_brief_lines.append(f"- Fair Value: {valuation_result.get('estimated_fair_value')}")
        final_brief_lines.append(f"- Valuation Gap %: {valuation_result.get('valuation_gap_pct')}")

    if sentiment_result:
        final_brief_lines.append("## Sentiment Summary")
        final_brief_lines.append(f"- Sentiment Label: {sentiment_result.get('sentiment_label')}")
        final_brief_lines.append(f"- Sentiment Score: {sentiment_result.get('sentiment_score')}")
        final_brief_lines.append(f"- Risk Mentions: {sentiment_result.get('risk_hits')}")

    if risks:
        final_brief_lines.append("## Risk Signals")
        for risk in risks:
            final_brief_lines.append(f"- {risk}")
    else:
        final_brief_lines.append("## Risk Signals\n- No major risk flags detected.")

    if investment_thesis:
        final_brief_lines.append(investment_thesis)

    return {
        "recommendation": recommendation,
        "risks": risks,
        "valuation_result": valuation_result,
        "research_report": research_report,
        "investment_thesis": investment_thesis,
        "final_brief": "\n\n".join(final_brief_lines),
    }
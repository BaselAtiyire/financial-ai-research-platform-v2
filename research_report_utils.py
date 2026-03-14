def generate_equity_research_report(
    company_name,
    ticker,
    financial_metrics,
    market_snapshot,
    sentiment_result=None,
    valuation_result=None,
):
    report = []

    report.append(f"# Equity Research Report: {company_name} ({ticker})")

    report.append("## Company Overview")
    report.append(
        f"{company_name} is analyzed based on uploaded financial documents "
        "and current market conditions."
    )

    if market_snapshot:
        report.append("## Market Snapshot")
        report.append(f"Current Price: {market_snapshot.get('price')}")
        report.append(f"Market Cap: {market_snapshot.get('market_cap')}")
        report.append(f"Daily Change: {market_snapshot.get('change_pct')}%")

    if financial_metrics:
        report.append("## Key Financial Metrics")

        for m in financial_metrics[:10]:
            metric = m.get("metric")
            value = m.get("value")
            period = m.get("period")
            document_name = m.get("document_name", "Unknown document")

            report.append(f"- {metric}: {value} ({period}) | Source: {document_name}")

    if sentiment_result:
        report.append("## Earnings Call Sentiment")
        report.append(f"Sentiment Label: {sentiment_result.get('sentiment_label')}")
        report.append(f"Positive Signals: {sentiment_result.get('positive_hits')}")
        report.append(f"Negative Signals: {sentiment_result.get('negative_hits')}")
        report.append(f"Risk Mentions: {sentiment_result.get('risk_hits')}")
        report.append(f"Guidance Mentions: {sentiment_result.get('guidance_hits')}")

    if valuation_result:
        report.append("## AI Valuation Summary")
        report.append(f"Revenue Used: {valuation_result.get('revenue')}")
        report.append(f"Net Income Used: {valuation_result.get('net_income')}")
        report.append(f"P/E Fair Value: {valuation_result.get('pe_fair_value')}")
        report.append(f"Revenue Multiple Value: {valuation_result.get('revenue_multiple_value')}")
        report.append(f"Estimated Fair Value: {valuation_result.get('estimated_fair_value')}")
        report.append(f"Valuation Gap %: {valuation_result.get('valuation_gap_pct')}")
        report.append(f"Valuation Signal: {valuation_result.get('signal')}")

    report.append("## AI Investment Insight")
    report.append(
        "Based on the financial metrics, sentiment signals, valuation output, "
        "and market performance, the company shows potential for further evaluation by investors."
    )

    report.append(
        "This report is AI-generated and should be used for research purposes only."
    )

    return "\n\n".join(report)
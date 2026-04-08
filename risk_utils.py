def detect_company_risk(metrics: list[dict], sentiment: dict | None) -> list[str]:
    risks = []

    for m in metrics:
        metric_name = str(m.get("metric", "")).lower()
        numeric_value = m.get("numeric_value")
        if numeric_value is None:
            continue
        try:
            numeric_value = float(numeric_value)
        except (TypeError, ValueError):
            continue

        if "net income" in metric_name and numeric_value < 0:
            risks.append("Negative net income detected")
        if "cash flow" in metric_name and numeric_value < 0:
            risks.append("Negative cash flow detected")
        if "revenue" in metric_name and numeric_value < 0:
            risks.append("Negative revenue detected")
        if "debt" in metric_name and numeric_value > 0:
            risks.append("Significant debt levels present")

    if sentiment:
        if sentiment.get("sentiment_label") == "Bearish":
            risks.append("Negative management sentiment")
        if sentiment.get("risk_hits", 0) > 5:
            risks.append(f"High risk mention count: {sentiment.get('risk_hits')}")

    return list(set(risks))

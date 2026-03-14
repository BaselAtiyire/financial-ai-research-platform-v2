def detect_company_risk(metrics, sentiment):
    risks = []

    for m in metrics:
        metric_name = str(m.get("metric", "")).lower()
        numeric_value = m.get("numeric_value")

        if numeric_value is None:
            continue

        if "net income" in metric_name and numeric_value < 0:
            risks.append("Negative net income detected")

        if "cash flow" in metric_name and numeric_value < 0:
            risks.append("Negative cash flow detected")

    if sentiment and sentiment.get("sentiment_label") == "Bearish":
        risks.append("Negative management sentiment")

    return list(set(risks))
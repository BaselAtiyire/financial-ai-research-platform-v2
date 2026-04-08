def detect_company_risk(metrics: list[dict], sentiment: dict | None) -> list[str]:
    """Detect financial risk signals from metrics and sentiment."""
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
            risks.append(f"⚠️ Negative net income: {numeric_value:,.0f}")
        if "cash flow" in metric_name and numeric_value < 0:
            risks.append(f"⚠️ Negative cash flow: {numeric_value:,.0f}")
        if "revenue" in metric_name and numeric_value < 0:
            risks.append("⚠️ Negative revenue detected")
        if "gross margin" in metric_name and numeric_value < 0.10:
            risks.append(f"⚠️ Very low gross margin: {numeric_value:.1%}")
        if "debt" in metric_name and "equity" not in metric_name and numeric_value > 0:
            risks.append("📌 Significant debt levels present")
        if "ebitda" in metric_name and numeric_value < 0:
            risks.append("⚠️ Negative EBITDA detected")

    if sentiment:
        label = sentiment.get("sentiment_label", "")
        if label in ["Bearish", "Mildly Bearish"]:
            risks.append(f"📉 {label} management sentiment detected")
        risk_hits = sentiment.get("risk_hits", 0)
        if risk_hits > 5:
            risks.append(f"⚠️ High risk keyword count: {risk_hits} mentions")
        if sentiment.get("negative_hits", 0) > sentiment.get("positive_hits", 0) * 2:
            risks.append("⚠️ Negative language significantly outweighs positive")

    return list(dict.fromkeys(risks))  # deduplicate preserving order

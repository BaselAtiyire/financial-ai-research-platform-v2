from typing import Optional


def safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_key_metric(metrics: list[dict], target_names: list[str]) -> Optional[float]:
    for item in metrics:
        metric_name = str(item.get("metric", "")).lower()
        numeric_value = item.get("numeric_value")

        if numeric_value is None:
            continue

        for target in target_names:
            if target in metric_name:
                return safe_float(numeric_value)

    return None


def estimate_pe_fair_value(
    net_income: Optional[float],
    market_cap: Optional[float],
    pe_multiple: float = 20.0,
) -> Optional[float]:
    if net_income is None or market_cap is None or net_income <= 0:
        return None

    return net_income * pe_multiple


def estimate_revenue_multiple_value(
    revenue: Optional[float],
    revenue_multiple: float = 3.0,
) -> Optional[float]:
    if revenue is None or revenue <= 0:
        return None

    return revenue * revenue_multiple


def estimate_intrinsic_signal(
    market_cap: Optional[float],
    pe_value: Optional[float],
    revenue_value: Optional[float],
) -> dict:
    estimates = [v for v in [pe_value, revenue_value] if v is not None]

    if market_cap is None or not estimates:
        return {
            "estimated_fair_value": None,
            "valuation_gap_pct": None,
            "signal": "Insufficient Data",
        }

    estimated_fair_value = sum(estimates) / len(estimates)
    valuation_gap_pct = ((estimated_fair_value - market_cap) / market_cap) * 100

    if valuation_gap_pct > 20:
        signal = "Potentially Undervalued"
    elif valuation_gap_pct < -20:
        signal = "Potentially Overvalued"
    else:
        signal = "Fairly Valued"

    return {
        "estimated_fair_value": estimated_fair_value,
        "valuation_gap_pct": valuation_gap_pct,
        "signal": signal,
    }


def build_valuation_summary(metrics: list[dict], market_snapshot: dict | None) -> dict:
    market_cap = None
    if market_snapshot:
        market_cap = safe_float(market_snapshot.get("market_cap"))

    revenue = extract_key_metric(metrics, ["revenue", "sales"])
    net_income = extract_key_metric(metrics, ["net income", "profit", "earnings"])

    pe_value = estimate_pe_fair_value(
        net_income=net_income,
        market_cap=market_cap,
        pe_multiple=20.0,
    )

    revenue_value = estimate_revenue_multiple_value(
        revenue=revenue,
        revenue_multiple=3.0,
    )

    signal_result = estimate_intrinsic_signal(
        market_cap=market_cap,
        pe_value=pe_value,
        revenue_value=revenue_value,
    )

    return {
        "revenue": revenue,
        "net_income": net_income,
        "market_cap": market_cap,
        "pe_fair_value": pe_value,
        "revenue_multiple_value": revenue_value,
        "estimated_fair_value": signal_result["estimated_fair_value"],
        "valuation_gap_pct": signal_result["valuation_gap_pct"],
        "signal": signal_result["signal"],
    }
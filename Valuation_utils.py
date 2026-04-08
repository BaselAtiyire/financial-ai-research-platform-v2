from typing import Optional


def safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_key_metric(metrics: list[dict], target_names: list[str]) -> Optional[float]:
    """Find the best numeric match for a metric by name keywords."""
    best = None
    best_conf = -1.0
    for item in metrics:
        metric_name = str(item.get("metric", "")).lower()
        numeric_value = safe_float(item.get("numeric_value"))
        if numeric_value is None:
            continue
        confidence = safe_float(item.get("confidence") or 1.0) or 1.0
        for target in target_names:
            if target in metric_name and confidence > best_conf:
                best = numeric_value
                best_conf = confidence
    return best


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


def estimate_ev_ebitda_value(
    ebitda: Optional[float],
    ev_multiple: float = 12.0,
) -> Optional[float]:
    """EV/EBITDA valuation — typically 10-15x for mature companies."""
    if ebitda is None or ebitda <= 0:
        return None
    return ebitda * ev_multiple


def estimate_dcf_value(
    free_cash_flow: Optional[float],
    growth_rate: float = 0.08,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.03,
    years: int = 5,
) -> Optional[float]:
    """Simplified DCF: project FCF for N years then apply terminal value."""
    if free_cash_flow is None or free_cash_flow <= 0:
        return None

    pv_cash_flows = 0.0
    fcf = free_cash_flow
    for year in range(1, years + 1):
        fcf *= (1 + growth_rate)
        pv_cash_flows += fcf / ((1 + discount_rate) ** year)

    # Terminal value
    terminal_value = (fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** years)

    return pv_cash_flows + pv_terminal


def estimate_intrinsic_signal(
    market_cap: Optional[float],
    estimates: list[Optional[float]],
) -> dict:
    valid_estimates = [v for v in estimates if v is not None and v > 0]
    if market_cap is None or not valid_estimates:
        return {
            "estimated_fair_value": None,
            "valuation_gap_pct": None,
            "signal": "Insufficient Data",
        }

    estimated_fair_value = sum(valid_estimates) / len(valid_estimates)
    valuation_gap_pct = ((estimated_fair_value - market_cap) / market_cap) * 100

    if valuation_gap_pct > 25:
        signal = "Potentially Undervalued"
    elif valuation_gap_pct < -25:
        signal = "Potentially Overvalued"
    elif valuation_gap_pct > 10:
        signal = "Slight Upside"
    elif valuation_gap_pct < -10:
        signal = "Slight Downside"
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

    revenue = extract_key_metric(metrics, ["revenue", "sales", "total revenue", "net revenue"])
    net_income = extract_key_metric(metrics, ["net income", "net profit", "earnings", "net earnings"])
    ebitda = extract_key_metric(metrics, ["ebitda", "adjusted ebitda"])
    free_cash_flow = extract_key_metric(metrics, ["free cash flow", "fcf", "free cash"])

    pe_value = estimate_pe_fair_value(net_income, market_cap, pe_multiple=20.0)
    revenue_value = estimate_revenue_multiple_value(revenue, revenue_multiple=3.0)
    ev_ebitda_value = estimate_ev_ebitda_value(ebitda, ev_multiple=12.0)
    dcf_value = estimate_dcf_value(free_cash_flow)

    signal_result = estimate_intrinsic_signal(
        market_cap=market_cap,
        estimates=[pe_value, revenue_value, ev_ebitda_value, dcf_value],
    )

    methods_used = sum(1 for v in [pe_value, revenue_value, ev_ebitda_value, dcf_value] if v is not None)

    return {
        "revenue": revenue,
        "net_income": net_income,
        "ebitda": ebitda,
        "free_cash_flow": free_cash_flow,
        "market_cap": market_cap,
        "pe_fair_value": pe_value,
        "revenue_multiple_value": revenue_value,
        "ev_ebitda_value": ev_ebitda_value,
        "dcf_value": dcf_value,
        "estimated_fair_value": signal_result["estimated_fair_value"],
        "valuation_gap_pct": signal_result["valuation_gap_pct"],
        "signal": signal_result["signal"],
        "methods_used": methods_used,
    }

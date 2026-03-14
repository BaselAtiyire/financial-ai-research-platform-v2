import pandas as pd


def detect_metric_anomalies(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """
    Flags anomalies when the relative change between consecutive periods
    exceeds the threshold.
    Example: threshold=0.3 means 30% change.
    """
    if df.empty:
        return pd.DataFrame()

    working = df.copy()

    required_cols = {"metric", "numeric_value", "period_index"}
    if not required_cols.issubset(set(working.columns)):
        return pd.DataFrame()

    if "document_name" not in working.columns:
        working["document_name"] = "Unknown"

    working = working.dropna(subset=["metric", "numeric_value", "period_index"])
    working = working.sort_values(["document_name", "metric", "period_index"])

    anomaly_rows = []

    grouped = working.groupby(["document_name", "metric"], dropna=True)

    for (document_name, metric), group in grouped:
        group = group.sort_values("period_index").copy()

        if len(group) < 2:
            continue

        group["previous_value"] = group["numeric_value"].shift(1)
        group["absolute_change"] = group["numeric_value"] - group["previous_value"]

        def pct_change(current, previous):
            if previous in (None, 0) or pd.isna(previous):
                return None
            return (current - previous) / abs(previous)

        group["pct_change"] = group.apply(
            lambda row: pct_change(row["numeric_value"], row["previous_value"]),
            axis=1,
        )

        flagged = group[group["pct_change"].abs() > threshold].copy()

        for _, row in flagged.iterrows():
            change_pct = row["pct_change"] * 100 if row["pct_change"] is not None else None

            if change_pct is None:
                continue

            anomaly_type = "Spike" if change_pct > 0 else "Drop"

            anomaly_rows.append(
                {
                    "document_name": document_name,
                    "metric": metric,
                    "period": row["period"],
                    "current_value": row["numeric_value"],
                    "previous_value": row["previous_value"],
                    "change_pct": change_pct,
                    "anomaly_type": anomaly_type,
                }
            )

    if not anomaly_rows:
        return pd.DataFrame()

    return pd.DataFrame(anomaly_rows)
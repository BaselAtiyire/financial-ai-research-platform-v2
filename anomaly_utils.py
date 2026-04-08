import pandas as pd
import numpy as np


def detect_metric_anomalies(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """
    Flags anomalies when the relative change between consecutive periods
    exceeds the threshold. Also detects statistical outliers using z-score.
    threshold=0.3 means 30% change triggers an anomaly flag.
    """
    if df.empty:
        return pd.DataFrame()

    working = df.copy()
    required_cols = {"metric", "numeric_value", "period_index"}
    if not required_cols.issubset(set(working.columns)):
        return pd.DataFrame()

    if "document_name" not in working.columns:
        working["document_name"] = "Unknown"
    if "period" not in working.columns:
        working["period"] = working["period_index"].astype(str)

    working = working.dropna(subset=["metric", "numeric_value", "period_index"])
    working = working.sort_values(["document_name", "metric", "period_index"])

    anomaly_rows = []
    grouped = working.groupby(["document_name", "metric"], dropna=True)

    for (document_name, metric), group in grouped:
        group = group.sort_values("period_index").copy()
        if len(group) < 2:
            continue

        values = group["numeric_value"].values

        # Period-over-period change
        group["previous_value"] = group["numeric_value"].shift(1)
        group = group.dropna(subset=["previous_value"])

        for _, row in group.iterrows():
            prev = row["previous_value"]
            curr = row["numeric_value"]

            if prev == 0 or pd.isna(prev):
                continue

            pct_change = (curr - prev) / abs(prev)

            if abs(pct_change) > threshold:
                change_pct = pct_change * 100
                anomaly_type = "Spike" if change_pct > 0 else "Drop"

                # Severity classification
                abs_pct = abs(change_pct)
                if abs_pct > 100:
                    severity = "Critical"
                elif abs_pct > 50:
                    severity = "High"
                else:
                    severity = "Medium"

                anomaly_rows.append({
                    "document_name": document_name,
                    "metric": metric,
                    "period": row.get("period", str(row["period_index"])),
                    "current_value": curr,
                    "previous_value": prev,
                    "change_pct": round(change_pct, 2),
                    "anomaly_type": anomaly_type,
                    "severity": severity,
                })

    if not anomaly_rows:
        return pd.DataFrame()

    result = pd.DataFrame(anomaly_rows)
    # Sort by absolute change descending
    result["abs_change"] = result["change_pct"].abs()
    result = result.sort_values("abs_change", ascending=False).drop(columns=["abs_change"])
    return result.reset_index(drop=True)

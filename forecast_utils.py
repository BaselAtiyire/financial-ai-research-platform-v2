import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def prepare_metric_history(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    metric_df = df[df["metric"].str.lower() == metric_name.lower()].copy()
    metric_df = metric_df.dropna(subset=["numeric_value", "period_index"])
    metric_df = metric_df.sort_values("period_index").reset_index(drop=True)
    return metric_df


def forecast_next_value(metric_df: pd.DataFrame) -> dict | None:
    if metric_df.empty or len(metric_df) < 2:
        return None

    X = metric_df[["period_index"]].values.astype(float)
    y = metric_df["numeric_value"].values.astype(float)

    model = LinearRegression()
    model.fit(X, y)

    next_index = float(metric_df["period_index"].max()) + 1
    predicted_value = float(model.predict([[next_index]])[0])

    # Calculate R² for confidence indication
    r2 = float(model.score(X, y))

    # Simple confidence interval: ±1 std of residuals
    residuals = y - model.predict(X)
    std_residual = float(np.std(residuals))

    # Trend direction
    slope = float(model.coef_[0])
    if slope > 0:
        trend = "Upward"
    elif slope < 0:
        trend = "Downward"
    else:
        trend = "Flat"

    return {
        "next_period_index": int(next_index),
        "forecast_value": predicted_value,
        "lower_bound": predicted_value - std_residual,
        "upper_bound": predicted_value + std_residual,
        "r2_score": r2,
        "trend": trend,
        "slope": slope,
    }

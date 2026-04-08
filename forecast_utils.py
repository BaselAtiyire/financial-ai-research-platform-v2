import pandas as pd
from sklearn.linear_model import LinearRegression


def prepare_metric_history(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    metric_df = df[df["metric"].str.lower() == metric_name.lower()].copy()
    metric_df = metric_df.dropna(subset=["numeric_value", "period_index"])
    metric_df = metric_df.sort_values("period_index")
    return metric_df


def forecast_next_value(metric_df: pd.DataFrame) -> dict | None:
    if metric_df.empty or len(metric_df) < 2:
        return None

    X = metric_df[["period_index"]].values
    y = metric_df["numeric_value"].values

    model = LinearRegression()
    model.fit(X, y)

    next_index = int(metric_df["period_index"].max()) + 1
    predicted_value = model.predict([[next_index]])[0]

    return {
        "next_period_index": next_index,
        "forecast_value": float(predicted_value),
    }

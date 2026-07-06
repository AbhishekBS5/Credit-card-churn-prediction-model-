"""
Builds the aggregated CSV feeding the Power BI dashboard:
predicted churn rate by customer segment, credit utilization band,
and transaction frequency quartile.

Run this after churn_prediction.py has produced outputs/churn_predictions.csv.
"""

import os
import pandas as pd

INPUT_PATH = "outputs/churn_predictions.csv"
OUTPUT_PATH = "outputs/powerbi_dashboard_data.csv"


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Could not find {INPUT_PATH}. Run churn_prediction.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    # Utilization band
    df["utilization_band"] = pd.cut(
        df["Avg_Utilization_Ratio"],
        bins=[-0.01, 0.25, 0.5, 0.75, 1.0],
        labels=["0-25%", "25-50%", "50-75%", "75-100%"],
    )

    # Transaction frequency quartile
    df["transaction_quartile"] = pd.qcut(
        df["Total_Trans_Ct"], q=4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
    )

    summary = (
        df.groupby(["utilization_band", "transaction_quartile"], observed=True)
        .agg(
            accounts=("predicted_churn", "count"),
            predicted_churn_rate=("predicted_churn", "mean"),
            avg_churn_probability=("churn_probability", "mean"),
        )
        .reset_index()
    )
    summary["predicted_churn_rate"] = (summary["predicted_churn_rate"] * 100).round(1)
    summary["avg_churn_probability"] = (summary["avg_churn_probability"] * 100).round(1)

    summary.to_csv(OUTPUT_PATH, index=False)
    print(f"Power BI dashboard data saved to {OUTPUT_PATH}")
    print(summary.head(10))


if __name__ == "__main__":
    main()

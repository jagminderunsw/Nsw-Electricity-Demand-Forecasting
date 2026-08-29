"""Create a compact, public-safe prediction dataset for the portfolio app.

The exporter can read either the merged output produced by the model-comparison
notebook or the three individual prediction files produced by the baseline,
XGBoost and LSTM notebooks. It validates that all models are evaluated against
the same actual demand values before writing any output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MERGED_COLUMNS = {
    "DATETIME",
    "Actual_Demand",
    "Baseline_Prediction",
    "XGBoost_Prediction",
    "LSTM_Prediction",
    "SelectedBaselineName",
}

PUBLIC_COLUMNS = [
    "datetime",
    "actual_mw",
    "baseline_mw",
    "xgboost_mw",
    "lstm_mw",
    "selected_baseline",
]


def _read_parquet(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    frame = pd.read_parquet(path)
    missing = required_columns.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path.name} is missing columns: {missing_text}")

    frame = frame.copy()
    frame["DATETIME"] = pd.to_datetime(frame["DATETIME"], errors="raise")
    if frame["DATETIME"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate DATETIME values")

    return frame


def _validate_actuals(
    left: pd.Series,
    right: pd.Series,
    comparison_name: str,
) -> None:
    if not np.allclose(
        left.to_numpy(dtype="float64"),
        right.to_numpy(dtype="float64"),
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError(
            f"Actual demand does not match between {comparison_name} predictions"
        )


def merge_prediction_files(
    baseline_path: Path,
    xgboost_path: Path,
    lstm_path: Path,
) -> pd.DataFrame:
    baseline = _read_parquet(
        baseline_path,
        {
            "DATETIME",
            "TOTALDEMAND",
            "SelectedBaseline",
            "SelectedBaselineName",
        },
    )
    xgboost = _read_parquet(
        xgboost_path,
        {"DATETIME", "TOTALDEMAND", "XGBoost_Prediction"},
    )
    lstm = _read_parquet(
        lstm_path,
        {"DATETIME", "TOTALDEMAND", "LSTM_Prediction"},
    )

    baseline = baseline[
        ["DATETIME", "TOTALDEMAND", "SelectedBaseline", "SelectedBaselineName"]
    ].rename(
        columns={
            "TOTALDEMAND": "Actual_Baseline",
            "SelectedBaseline": "Baseline_Prediction",
        }
    )
    xgboost = xgboost[
        ["DATETIME", "TOTALDEMAND", "XGBoost_Prediction"]
    ].rename(columns={"TOTALDEMAND": "Actual_XGBoost"})
    lstm = lstm[["DATETIME", "TOTALDEMAND", "LSTM_Prediction"]].rename(
        columns={"TOTALDEMAND": "Actual_LSTM"}
    )

    comparison = (
        baseline.merge(xgboost, on="DATETIME", how="inner", validate="one_to_one")
        .merge(lstm, on="DATETIME", how="inner", validate="one_to_one")
        .sort_values("DATETIME")
        .reset_index(drop=True)
    )
    if comparison.empty:
        raise ValueError("The prediction files have no common timestamps")

    _validate_actuals(
        comparison["Actual_Baseline"],
        comparison["Actual_XGBoost"],
        "baseline and XGBoost",
    )
    _validate_actuals(
        comparison["Actual_Baseline"],
        comparison["Actual_LSTM"],
        "baseline and LSTM",
    )

    comparison["Actual_Demand"] = comparison["Actual_Baseline"]
    return comparison[
        [
            "DATETIME",
            "Actual_Demand",
            "Baseline_Prediction",
            "XGBoost_Prediction",
            "LSTM_Prediction",
            "SelectedBaselineName",
        ]
    ]


def read_merged_comparison(path: Path) -> pd.DataFrame:
    comparison = _read_parquet(path, MERGED_COLUMNS)
    return comparison[sorted(MERGED_COLUMNS)].sort_values("DATETIME").reset_index(
        drop=True
    )


def calculate_metrics(comparison: pd.DataFrame) -> pd.DataFrame:
    actual = comparison["Actual_Demand"].to_numpy(dtype="float64")
    selected_baseline = str(comparison["SelectedBaselineName"].iloc[0])
    model_columns = {
        f"Baseline - {selected_baseline}": "Baseline_Prediction",
        "XGBoost": "XGBoost_Prediction",
        "LSTM": "LSTM_Prediction",
    }

    rows: list[dict[str, float | str | int]] = []
    for model_name, prediction_column in model_columns.items():
        predicted = comparison[prediction_column].to_numpy(dtype="float64")
        valid = np.isfinite(actual) & np.isfinite(predicted)
        if not valid.all():
            raise ValueError(f"{model_name} contains non-finite evaluation values")

        residuals = predicted - actual
        absolute_errors = np.abs(residuals)
        squared_errors = residuals**2
        total_variation = np.sum((actual - actual.mean()) ** 2)
        if total_variation == 0:
            raise ValueError("R2 is undefined because actual demand is constant")

        rows.append(
            {
                "Model": model_name,
                "Observations": int(len(actual)),
                "MAE": np.mean(absolute_errors),
                "RMSE": np.sqrt(np.mean(squared_errors)),
                "MAPE": np.mean(absolute_errors / actual) * 100,
                "R2": 1 - (np.sum(squared_errors) / total_variation),
            }
        )

    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def build_demo_predictions(comparison: pd.DataFrame, days: int) -> pd.DataFrame:
    if days < 1:
        raise ValueError("days must be at least 1")

    required = MERGED_COLUMNS.difference(comparison.columns)
    if required:
        missing_text = ", ".join(sorted(required))
        raise ValueError(f"Merged comparison is missing columns: {missing_text}")

    comparison = comparison.sort_values("DATETIME").reset_index(drop=True)
    expected_rows = days * 48
    if len(comparison) < expected_rows:
        raise ValueError(
            f"Need at least {expected_rows:,} observations for {days} days; "
            f"found {len(comparison):,}"
        )

    demo = comparison.tail(expected_rows).rename(
        columns={
            "DATETIME": "datetime",
            "Actual_Demand": "actual_mw",
            "Baseline_Prediction": "baseline_mw",
            "XGBoost_Prediction": "xgboost_mw",
            "LSTM_Prediction": "lstm_mw",
            "SelectedBaselineName": "selected_baseline",
        }
    )
    demo = demo[PUBLIC_COLUMNS].reset_index(drop=True)

    numeric_columns = [
        "actual_mw",
        "baseline_mw",
        "xgboost_mw",
        "lstm_mw",
    ]
    if not np.isfinite(demo[numeric_columns].to_numpy(dtype="float64")).all():
        raise ValueError("Demo predictions contain missing or non-finite values")

    # Float32 is precise enough for MW display and keeps the deployment file small.
    demo[numeric_columns] = demo[numeric_columns].astype("float32")

    if not demo["datetime"].is_monotonic_increasing:
        raise ValueError("Demo timestamps are not ordered chronologically")

    return demo


def export_demo(
    comparison: pd.DataFrame,
    output_path: Path,
    metrics_path: Path,
    days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = calculate_metrics(comparison)
    demo = build_demo_predictions(comparison, days)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    demo.to_parquet(output_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    return demo, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        help="Merged common_test_predictions.parquet from notebook 07",
    )
    parser.add_argument("--baseline", type=Path, help="Baseline prediction Parquet")
    parser.add_argument("--xgboost", type=Path, help="XGBoost prediction Parquet")
    parser.add_argument("--lstm", type=Path, help="LSTM prediction Parquet")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/demo_predictions.parquet"),
        help="Compact deployment dataset to write",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/model_comparison.csv"),
        help="Full common-test metrics table to write",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of final half-hourly days to export (default: 90)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    individual_paths = [args.baseline, args.xgboost, args.lstm]

    if args.comparison:
        if any(individual_paths):
            raise SystemExit(
                "Use --comparison alone, or provide all of --baseline, "
                "--xgboost and --lstm"
            )
        comparison = read_merged_comparison(args.comparison)
    else:
        if not all(individual_paths):
            raise SystemExit(
                "Provide --comparison, or all of --baseline, --xgboost and --lstm"
            )
        comparison = merge_prediction_files(
            args.baseline,
            args.xgboost,
            args.lstm,
        )

    demo, metrics = export_demo(
        comparison,
        output_path=args.output,
        metrics_path=args.metrics_output,
        days=args.days,
    )

    print(f"Common evaluation observations: {len(comparison):,}")
    print(f"Demo observations: {len(demo):,}")
    print(f"Demo period: {demo['datetime'].min()} to {demo['datetime'].max()}")
    print(f"Demo output: {args.output.resolve()}")
    print(f"Metrics output: {args.metrics_output.resolve()}")
    print("\nValidated model metrics:")
    print(metrics.round({"MAE": 2, "RMSE": 2, "MAPE": 2, "R2": 4}).to_string(index=False))


if __name__ == "__main__":
    main()

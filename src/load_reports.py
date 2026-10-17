
from pathlib import Path
import json
import math
import os
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "customer_reviews.db"


CSV_FILES = {
    "sentiment_distribution": "sentiment_distribution.csv",
    "category_sentiment_summary": "category_sentiment_summary.csv",
    "complaint_theme_heatmap_data": "complaint_theme_heatmap_data.csv",
}

INDEXED_FILES = {
    "classification_report": (
        "classification_report.csv",
        "class_name",
    ),
    "confusion_matrix": (
        "confusion_matrix.csv",
        "actual_label",
    ),
}


def require_columns(df, required, source):
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"{source}: missing required columns: {sorted(missing)}"
        )


def require_finite_numeric(df, columns, source):
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or not values.map(
            lambda value: math.isfinite(float(value))
        ).all():
            raise ValueError(
                f"{source}: '{column}' contains missing or non-numeric values"
            )


def read_and_validate_reports():
    tables = {}

    # Load standard report CSVs.
    for table_name, filename in CSV_FILES.items():
        path = REPORTS / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required report not found: {path}")

        tables[table_name] = pd.read_csv(path)

        if tables[table_name].empty:
            raise ValueError(f"{filename}: report must not be empty")

    # Restore the row labels saved as the first CSV column.
    for table_name, (filename, index_name) in INDEXED_FILES.items():
        path = REPORTS / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required report not found: {path}")

        df = pd.read_csv(path, index_col=0)
        df.index.name = index_name
        tables[table_name] = df.reset_index()

    # Read and validate model metrics.
    metrics_path = REPORTS / "model_metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"Required metrics file not found: {metrics_path}"
        )

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    required_metrics = {
        "model",
        "test_samples",
        "accuracy",
        "macro_f1",
        "weighted_f1",
    }
    missing_metrics = required_metrics - set(metrics)
    if missing_metrics:
        raise ValueError(
            f"model_metrics.json: missing keys {sorted(missing_metrics)}"
        )

    if not isinstance(metrics["model"], str) or not metrics["model"].strip():
        raise ValueError("Model name must be a non-empty string")

    if (
        isinstance(metrics["test_samples"], bool)
        or not isinstance(metrics["test_samples"], int)
        or metrics["test_samples"] <= 0
    ):
        raise ValueError("test_samples must be a positive integer")

    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        value = metrics[metric]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{metric} must be a finite number between 0 and 1")

    # Validate report schemas.
    sentiment = tables["sentiment_distribution"]
    require_columns(
        sentiment,
        ["Sentiment", "Review count", "Percentage"],
        "sentiment_distribution.csv",
    )
    require_finite_numeric(
        sentiment, ["Review count", "Percentage"],
        "sentiment_distribution.csv",
    )

    categories = tables["category_sentiment_summary"]
    require_columns(
        categories,
        ["Category", "total_reviews", "negative_reviews",
         "negative_share_pct"],
        "category_sentiment_summary.csv",
    )
    require_finite_numeric(
        categories,
        ["total_reviews", "negative_reviews", "negative_share_pct"],
        "category_sentiment_summary.csv",
    )

    themes = tables["complaint_theme_heatmap_data"]
    require_columns(themes, ["Category"], "complaint_theme_heatmap_data.csv")
    theme_columns = [c for c in themes.columns if c != "Category"]
    if not theme_columns:
        raise ValueError("Complaint theme report has no theme columns")
    require_finite_numeric(
        themes, theme_columns, "complaint_theme_heatmap_data.csv"
    )

    classification = tables["classification_report"]
    require_columns(
        classification,
        ["class_name", "precision", "recall", "f1-score", "support"],
        "classification_report.csv",
    )
    require_finite_numeric(
        classification,
        ["precision", "recall", "f1-score", "support"],
        "classification_report.csv",
    )

    confusion = tables["confusion_matrix"]
    require_columns(
        confusion,
        ["actual_label"],
        "confusion_matrix.csv",
    )
    confusion_columns = [
        c for c in confusion.columns if c != "actual_label"
    ]
    if not confusion_columns:
        raise ValueError("Confusion matrix has no prediction columns")
    require_finite_numeric(
        confusion, confusion_columns, "confusion_matrix.csv"
    )

    # Check consistency and sensible ranges.
    if sentiment["Sentiment"].duplicated().any():
        raise ValueError("Sentiment report contains duplicate sentiment labels")
    if set(sentiment["Sentiment"]) != {"Positive", "Negative", "Neutral"}:
        raise ValueError("Sentiment report must contain Positive, Negative, Neutral")

    if (sentiment["Review count"] < 0).any():
        raise ValueError("Review counts cannot be negative")
    if (sentiment["Review count"] % 1 != 0).any():
        raise ValueError("Review counts must be whole numbers")
    if (sentiment["Percentage"] < 0).any() or (
        sentiment["Percentage"] > 100
    ).any():
        raise ValueError("Sentiment percentages must be between 0 and 100")
    if abs(sentiment["Percentage"].sum() - 100) > 0.1:
        raise ValueError("Sentiment percentages do not sum to approximately 100")

    for column in ("total_reviews", "negative_reviews"):
        if (categories[column] < 0).any() or (
            categories[column] % 1 != 0
        ).any():
            raise ValueError(f"{column} must contain non-negative whole numbers")

    if (categories["total_reviews"] <= 0).any():
        raise ValueError("Category totals must be greater than zero")
    if (categories["negative_reviews"] > categories["total_reviews"]).any():
        raise ValueError("Negative reviews cannot exceed total reviews")
    if (
        (categories["negative_share_pct"] < 0)
        | (categories["negative_share_pct"] > 100)
    ).any():
        raise ValueError("Category negative shares must be between 0 and 100")

    calculated_shares = (
        100 * categories["negative_reviews"] / categories["total_reviews"]
    )
    if (
        calculated_shares - categories["negative_share_pct"]
    ).abs().max() >= 0.01:
        raise ValueError("Category negative shares do not match their counts")

    if sentiment["Review count"].sum() != categories["total_reviews"].sum():
        raise ValueError("Sentiment and category review totals do not match")

    if (
        (themes[theme_columns] < 0).any().any()
        or (themes[theme_columns] > 100).any().any()
    ):
        raise ValueError("Theme percentages must be between 0 and 100")

    if (classification[["precision", "recall", "f1-score"]] < 0).any().any():
        raise ValueError("Classification metrics cannot be negative")
    if (
        classification[["precision", "recall", "f1-score"]] > 1
    ).any().any():
        raise ValueError("Classification metrics cannot exceed 1")

    if (confusion[confusion_columns] < 0).any().any():
        raise ValueError("Confusion matrix counts cannot be negative")
    if (
        confusion[confusion_columns] % 1 != 0
    ).any().any():
        raise ValueError("Confusion matrix counts must be whole numbers")

    matrix_total = int(confusion[confusion_columns].to_numpy().sum())
    if matrix_total != metrics["test_samples"]:
        raise ValueError(
            f"Confusion matrix total ({matrix_total}) does not match "
            f"test_samples ({metrics['test_samples']})"
        )

    tables["model_metrics"] = pd.DataFrame([metrics])
    return tables


def load_reports():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Validate every source before changing the existing database.
    tables = read_and_validate_reports()
    # Build a separate database first. Close the SQLite connection
    # before replacing the existing database on Windows.
    temp_path = DATA_DIR / ".customer_reviews.tmp.db"

    try:
        if temp_path.exists():
            temp_path.unlink()

        connection = sqlite3.connect(temp_path)

        try:
            for table_name, df in tables.items():
                df.to_sql(
                    table_name,
                    connection,
                    if_exists="replace",
                    index=False,
                )

            connection.commit()

            result = connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]

            if result != "ok":
                raise RuntimeError(
                    f"SQLite integrity check failed: {result}"
                )

            loaded_tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }

            if loaded_tables != set(tables):
                raise RuntimeError("Database table verification failed")

        finally:
            connection.close()

        os.replace(temp_path, DB_PATH)

        print(f"Validated reports and refreshed database: {DB_PATH}")
        print("Tables loaded:")
        for table_name, df in tables.items():
            print(f"  {table_name}: {len(df)} rows")

    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except PermissionError:
                print(
                    f"Warning: Could not remove temporary file: {temp_path}. "
                    "Close any process using it and remove it later."
                )


if __name__ == "__main__":
    load_reports()

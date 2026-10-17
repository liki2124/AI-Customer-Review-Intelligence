from pathlib import Path
import json
import sqlite3

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "customer_reviews.db"


def load_reports():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "sentiment_distribution": "sentiment_distribution.csv",
        "category_sentiment_summary": "category_sentiment_summary.csv",
        "complaint_theme_heatmap_data": "complaint_theme_heatmap_data.csv",
    }

    tables = {}

    for table_name, filename in files.items():
        path = REPORTS / filename
        if not path.exists():
            raise FileNotFoundError(f"Required report not found: {path}")

        tables[table_name] = pd.read_csv(path)

    # These CSV files store their row labels in the first, unnamed column.
    for table_name, filename, index_name in [
        ("classification_report", "classification_report.csv", "class_name"),
        ("confusion_matrix", "confusion_matrix.csv", "actual_label"),
    ]:
        path = REPORTS / filename
        if not path.exists():
            raise FileNotFoundError(f"Required report not found: {path}")

        df = pd.read_csv(path, index_col=0)
        df.index.name = index_name
        tables[table_name] = df.reset_index()

    metrics_path = REPORTS / "model_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Required metrics file not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    # Write all generated tables in one database transaction.
    with sqlite3.connect(DB_PATH) as connection:
        for table_name, df in tables.items():
            df.to_sql(table_name, connection, if_exists="replace", index=False)

        pd.DataFrame([metrics]).to_sql(
            "model_metrics", connection, if_exists="replace", index=False
        )

    print(f"SQLite database created: {DB_PATH}")
    print("Tables loaded:")
    for table_name, df in tables.items():
        print(f"  {table_name}: {len(df)} rows")
    print(f"  model_metrics: {len(metrics)} row")


if __name__ == "__main__":
    load_reports()

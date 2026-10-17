import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from load_reports import DB_PATH, load_reports


@pytest.fixture(scope="module", autouse=True)
def loaded_database():
    load_reports()
    return DB_PATH


def test_sentiment_counts_total_60886():
    df = pd.read_csv(ROOT / "reports" / "sentiment_distribution.csv")

    assert df["Review count"].sum() == 60886
    assert set(df["Sentiment"]) == {"Positive", "Negative", "Neutral"}


def test_category_counts_total_60886():
    df = pd.read_csv(ROOT / "reports" / "category_sentiment_summary.csv")

    assert df["total_reviews"].sum() == 60886
    assert (df["negative_reviews"] <= df["total_reviews"]).all()


def test_negative_share_calculations():
    df = pd.read_csv(ROOT / "reports" / "category_sentiment_summary.csv")

    calculated = 100 * df["negative_reviews"] / df["total_reviews"]

    assert (calculated - df["negative_share_pct"]).abs().max() < 0.01


def test_model_metrics_match_expected_results():
    with (ROOT / "reports" / "model_metrics.json").open(
        encoding="utf-8"
    ) as file:
        metrics = json.load(file)

    assert metrics["test_samples"] == 12178
    assert metrics["accuracy"] == pytest.approx(0.8432419116)
    assert metrics["macro_f1"] == pytest.approx(0.6779108464)
    assert metrics["weighted_f1"] == pytest.approx(0.8581089453)


def test_confusion_matrix_totals_match_test_samples():
    df = pd.read_csv(ROOT / "reports" / "confusion_matrix.csv", index_col=0)

    assert int(df.to_numpy().sum()) == 12178


def test_sqlite_database_contains_expected_tables(loaded_database):
    with sqlite3.connect(loaded_database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert {
        "sentiment_distribution",
        "category_sentiment_summary",
        "complaint_theme_heatmap_data",
        "classification_report",
        "confusion_matrix",
        "model_metrics",
    }.issubset(tables)


def test_sqlite_sentiment_total(loaded_database):
    with sqlite3.connect(loaded_database) as connection:
        total = connection.execute(
            'SELECT SUM("Review count") FROM sentiment_distribution'
        ).fetchone()[0]

    assert total == 60886


def test_sqlite_category_negative_share(loaded_database):
    with sqlite3.connect(loaded_database) as connection:
        rows = connection.execute(
            """
            SELECT Category, negative_share_pct
            FROM category_sentiment_summary
            ORDER BY negative_share_pct DESC
            """
        ).fetchall()

    assert len(rows) == 5
    assert rows[0][0] == "mobile accessories"
    assert rows[0][1] == pytest.approx(16.447235, abs=0.001)

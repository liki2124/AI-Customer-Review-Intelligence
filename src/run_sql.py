import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "customer_reviews.db"
SQL_DIR = ROOT / "sql"


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}. Run python src/load_reports.py first."
        )

    sql_files = sorted(SQL_DIR.glob("*.sql"))

    if not sql_files:
        raise FileNotFoundError(f"No SQL files found in {SQL_DIR}")

    with sqlite3.connect(DB_PATH) as connection:
        for sql_file in sql_files:
            query = sql_file.read_text(encoding="utf-8-sig")

            print(f"\n===== {sql_file.name} =====")

            cursor = connection.execute(query)
            headers = [column[0] for column in cursor.description or []]
            rows = cursor.fetchall()

            if not headers:
                print("No result columns returned.")
                continue

            print(" | ".join(headers))
            print("-" * 90)

            for row in rows:
                print(" | ".join(str(value) for value in row))

            print(f"\nRows returned: {len(rows)}")


if __name__ == "__main__":
    main()

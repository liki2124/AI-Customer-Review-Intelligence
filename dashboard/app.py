from pathlib import Path
import sqlite3
import sys
from src.load_reports import load_reports
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DB_PATH = ROOT / "data" / "customer_reviews.db"

st.set_page_config(
    page_title="AI Customer Review Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AI Customer Review Intelligence")
st.caption(
    "Customer sentiment, product-category risks, complaint themes "
    "and machine-learning performance."
)
if not DB_PATH.exists():
    try:
        with st.spinner("Preparing the reporting database..."):
            load_reports()
    except Exception as exc:
        st.error("Could not prepare the reporting database.")
        st.exception(exc)
        st.stop()


@st.cache_data
def load_table(table_name: str) -> pd.DataFrame:
    allowed_tables = {
        "sentiment_distribution",
        "category_sentiment_summary",
        "complaint_theme_heatmap_data",
        "classification_report",
        "confusion_matrix",
        "model_metrics",
    }
    if table_name not in allowed_tables:
        raise ValueError("Unknown reporting table")

    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(
            f'SELECT * FROM "{table_name}"',
            connection,
        )


sentiment = load_table("sentiment_distribution")
categories = load_table("category_sentiment_summary")
themes = load_table("complaint_theme_heatmap_data")
classification = load_table("classification_report")
metrics = load_table("model_metrics")
confusion = load_table("confusion_matrix")

required_sentiment = {"Sentiment", "Review count", "Percentage"}
required_categories = {
    "Category",
    "total_reviews",
    "negative_reviews",
    "negative_share_pct",
}

if not required_sentiment.issubset(sentiment.columns):
    st.error("The sentiment report is missing required columns.")
    st.stop()

if not required_categories.issubset(categories.columns):
    st.error("The category report is missing required columns.")
    st.stop()

total_reviews = int(sentiment["Review count"].sum())
negative_reviews = int(
    sentiment.loc[
        sentiment["Sentiment"].eq("Negative"), "Review count"
    ].sum()
)
positive_reviews = int(
    sentiment.loc[
        sentiment["Sentiment"].eq("Positive"), "Review count"
    ].sum()
)

st.sidebar.header("Dashboard filters")
category_options = sorted(categories["Category"].astype(str).unique())
selected_categories = st.sidebar.multiselect(
    "Product categories",
    options=category_options,
    default=category_options,
)

filtered_categories = categories[
    categories["Category"].isin(selected_categories)
].copy()

st.sidebar.caption(f"{len(selected_categories)} of {len(category_options)} categories selected")

tab_overview, tab_categories, tab_themes, tab_model = st.tabs(
    [
        "Executive overview",
        "Category intelligence",
        "Complaint themes",
        "Model performance",
    ]
)

with tab_overview:
    st.subheader("Review portfolio")

    negative_share = (
        100 * negative_reviews / total_reviews if total_reviews else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total reviews", f"{total_reviews:,}")
    col2.metric("Positive reviews", f"{positive_reviews:,}")
    col3.metric("Negative reviews", f"{negative_reviews:,}")
    col4.metric("Negative share", f"{negative_share:.2f}%")

    left, right = st.columns(2)

    with left:
        st.markdown("#### Sentiment distribution")
        sentiment_chart = px.pie(
            sentiment,
            names="Sentiment",
            values="Review count",
            hole=0.48,
            title="Share of classified reviews",
        )
        sentiment_chart.update_traces(
            textposition="inside",
            textinfo="label+percent",
        )
        st.plotly_chart(sentiment_chart, width="stretch")

    with right:
        st.markdown("#### Review counts")
        sentiment_bar = px.bar(
            sentiment,
            x="Sentiment",
            y="Review count",
            color="Sentiment",
            text="Review count",
            title="Reviews by sentiment",
        )
        sentiment_bar.update_layout(showlegend=False)
        st.plotly_chart(sentiment_bar, width="stretch")

    st.download_button(
        "Download sentiment report (CSV)",
        data=sentiment.to_csv(index=False).encode("utf-8"),
        file_name="sentiment_distribution.csv",
        mime="text/csv",
        key="download_sentiment",
    )

    st.info(
        "Business takeaway: mobile accessories has the highest negative-review "
        "rate among the five categories. Neutral sentiment is the model's "
        "weakest classification class, so neutral-related conclusions need care."
    )

with tab_categories:
    st.subheader("Negative reviews by product category")

    if filtered_categories.empty:
        st.warning("Select at least one category in the sidebar.")
    else:
        category_chart = px.bar(
            filtered_categories.sort_values("negative_share_pct"),
            x="negative_share_pct",
            y="Category",
            orientation="h",
            text="negative_share_pct",
            title="Negative reviews as a percentage of category reviews",
            labels={
                "negative_share_pct": "Negative reviews (%)",
                "Category": "Product category",
            },
        )
        category_chart.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside",
        )
        st.plotly_chart(category_chart, width="stretch")

        volume_chart = px.bar(
            filtered_categories.sort_values("negative_reviews"),
            x="Category",
            y="negative_reviews",
            text="negative_reviews",
            title="Negative-review volume",
            labels={
                "negative_reviews": "Negative reviews",
                "Category": "Product category",
            },
        )
        st.plotly_chart(volume_chart, width="stretch")

        st.dataframe(
            filtered_categories.sort_values(
                "negative_share_pct", ascending=False
            ),
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download filtered category report (CSV)",
            data=filtered_categories.to_csv(index=False).encode("utf-8"),
            file_name="category_sentiment_summary.csv",
            mime="text/csv",
            key="download_categories",
        )

with tab_themes:
    st.subheader("Complaint-theme analysis")
    st.caption(
        "Compare the reported complaint-theme percentages across product "
        "categories. These are aggregated report values, not individual reviews."
    )

    theme_columns = [
        column for column in themes.columns
        if column != "Category"
    ]

    filtered_themes = themes[
        themes["Category"].isin(selected_categories)
    ].copy()

    if filtered_themes.empty or not theme_columns:
        st.warning("Select at least one category to view complaint themes.")
    else:
        theme_values = filtered_themes[theme_columns]

        theme_chart = px.imshow(
            theme_values,
            x=theme_columns,
            y=filtered_themes["Category"],
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="YlOrRd",
            title="Complaint-theme percentages by product category",
            labels={
                "x": "Complaint theme",
                "y": "Product category",
                "color": "Percentage",
            },
        )
        theme_chart.update_layout(xaxis={"side": "bottom"})
        st.plotly_chart(theme_chart, width="stretch")

        st.markdown("#### Report data")
        st.dataframe(
            filtered_themes,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download complaint themes (CSV)",
            data=filtered_themes.to_csv(index=False).encode("utf-8"),
            file_name="complaint_theme_analysis.csv",
            mime="text/csv",
            key="download_themes",
        )

with tab_model:
    st.subheader("Model performance")

    if not metrics.empty:
        model = metrics.iloc[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Test accuracy", f"{float(model['accuracy']):.2%}")
        col2.metric("Macro F1", f"{float(model['macro_f1']):.3f}")
        col3.metric("Weighted F1", f"{float(model['weighted_f1']):.3f}")

        st.caption(
            f"Model: {model['model']} · "
            f"Test samples: {int(model['test_samples']):,}"
        )

    if {"class_name", "precision", "recall", "f1-score"}.issubset(
        classification.columns
    ):
        st.markdown("#### Performance by sentiment class")
        class_chart = px.bar(
            classification,
            x="class_name",
            y=["precision", "recall", "f1-score"],
            barmode="group",
            range_y=[0, 1],
            title="Precision, recall and F1-score",
            labels={
                "class_name": "Sentiment class",
                "value": "Score",
                "variable": "Metric",
            },
        )
        st.plotly_chart(class_chart, width="stretch")
        st.dataframe(classification, width="stretch", hide_index=True)

        st.download_button(
            "Download classification report (CSV)",
            data=classification.to_csv(index=False).encode("utf-8"),
            file_name="classification_report.csv",
            mime="text/csv",
            key="download_classification",
        )

    st.markdown("#### Confusion matrix")
    if not confusion.empty:
        st.dataframe(confusion, width="stretch", hide_index=True)

        st.download_button(
            "Download confusion matrix (CSV)",
            data=confusion.to_csv(index=False).encode("utf-8"),
            file_name="confusion_matrix.csv",
            mime="text/csv",
            key="download_confusion",
        )

    st.warning(
        "The neutral class has lower precision, recall and F1 than the "
        "positive and negative classes. Avoid treating the overall accuracy "
        "as a complete measure of model quality."
    )

st.divider()
st.caption(
    "AI Customer Review Intelligence · Built with Python, SQL, SQLite, "
    "Streamlit and Plotly."
)

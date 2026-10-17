-- 02: Categories ranked by negative-review share
SELECT
    Category,
    total_reviews,
    negative_reviews,
    ROUND(negative_share_pct, 2) AS negative_share_pct
FROM category_sentiment_summary
ORDER BY negative_share_pct DESC;

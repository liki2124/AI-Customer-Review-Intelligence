-- 03: Categories with the highest number of negative reviews
SELECT
    Category,
    total_reviews,
    negative_reviews,
    ROUND(
        100.0 * negative_reviews / NULLIF(total_reviews, 0),
        2
    ) AS calculated_negative_share_pct
FROM category_sentiment_summary
ORDER BY negative_reviews DESC;

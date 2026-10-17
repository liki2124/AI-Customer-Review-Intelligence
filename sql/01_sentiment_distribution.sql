-- 01: Overall sentiment distribution
SELECT
    Sentiment,
    "Review count" AS review_count,
    ROUND(Percentage, 2) AS percentage
FROM sentiment_distribution
ORDER BY "Review count" DESC;

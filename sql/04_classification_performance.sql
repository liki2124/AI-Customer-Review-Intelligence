-- 04: Model performance summary
SELECT
    class_name,
    precision,
    recall,
    "f1-score" AS f1_score,
    support
FROM classification_report
WHERE class_name IN ('Negative', 'Neutral', 'Positive')
ORDER BY
    CASE class_name
        WHEN 'Negative' THEN 1
        WHEN 'Neutral' THEN 2
        WHEN 'Positive' THEN 3
        ELSE 4
    END;

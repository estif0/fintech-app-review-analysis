-- ============================================================
-- Database Verification Queries
-- Bank Reviews Analysis - Task 3
-- ============================================================

-- Query 1: Total Review Count
-- Expected: 827 reviews
SELECT COUNT(*) AS total_reviews FROM reviews;

-- Query 2: Reviews per Bank
-- Expected: CBE (~227), BOA (~290), Dashen (~310)
SELECT 
    b.bank_name,
    COUNT(r.review_id) AS review_count
FROM banks b
LEFT JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name
ORDER BY review_count DESC;

-- Query 3: Average Rating per Bank
-- Expected: Overall ~3.43
SELECT 
    b.bank_name,
    ROUND(AVG(r.rating), 2) AS avg_rating,
    COUNT(r.review_id) AS review_count
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY b.bank_name
ORDER BY avg_rating DESC;

-- Query 4: Rating Distribution
-- Expected: Mostly 1-star and 5-star reviews
SELECT 
    rating,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM reviews
GROUP BY rating
ORDER BY rating;

-- Query 5: Sentiment Distribution
-- Expected: ~510 Positive, ~304 Negative, ~13 Neutral
SELECT 
    sentiment_label,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM reviews
WHERE sentiment_label IS NOT NULL
GROUP BY sentiment_label
ORDER BY count DESC;

-- Query 6: Sentiment Distribution by Bank
SELECT 
    b.bank_name,
    r.sentiment_label,
    COUNT(*) AS count
FROM banks b
JOIN reviews r ON b.bank_id = r.bank_id
WHERE r.sentiment_label IS NOT NULL
GROUP BY b.bank_name, r.sentiment_label
ORDER BY b.bank_name, r.sentiment_label;

-- Query 7: Reviews with Themes
-- Expected: ~701 reviews with themes (84.8%)
SELECT 
    COUNT(*) AS reviews_with_themes,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM reviews), 2) AS percentage
FROM reviews
WHERE identified_themes IS NOT NULL AND identified_themes != '';

-- Query 8: Date Range of Reviews
-- Expected: 2022-07-16 to 2025-11-26
SELECT 
    MIN(review_date) AS earliest_review,
    MAX(review_date) AS latest_review,
    (MAX(review_date) - MIN(review_date)) AS date_span_days
FROM reviews;

-- Query 9: Reviews by Year
SELECT 
    EXTRACT(YEAR FROM review_date) AS year,
    COUNT(*) AS review_count
FROM reviews
GROUP BY year
ORDER BY year;

-- Query 10: Sample Reviews - High Rated (5 stars)
SELECT 
    b.bank_name,
    r.rating,
    r.sentiment_label,
    LEFT(r.review_text, 100) AS review_snippet,
    r.review_date
FROM reviews r
JOIN banks b ON r.bank_id = b.bank_id
WHERE r.rating = 5
ORDER BY r.review_date DESC
LIMIT 5;

-- Query 11: Sample Reviews - Low Rated (1 star)
SELECT 
    b.bank_name,
    r.rating,
    r.sentiment_label,
    LEFT(r.review_text, 100) AS review_snippet,
    r.review_date
FROM reviews r
JOIN banks b ON r.bank_id = b.bank_id
WHERE r.rating = 1
ORDER BY r.review_date DESC
LIMIT 5;

-- Query 12: Reviews with Rating Adjustments
-- Expected: ~221 reviews (26.7%)
SELECT 
    COUNT(*) AS rating_adjusted_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM reviews), 2) AS percentage
FROM reviews
WHERE rating_adjusted = TRUE;

-- Query 13: Data Quality Check - Missing Values
SELECT 
    'review_text' AS field,
    COUNT(*) - COUNT(review_text) AS missing_count
FROM reviews
UNION ALL
SELECT 
    'rating',
    COUNT(*) - COUNT(rating)
FROM reviews
UNION ALL
SELECT 
    'sentiment_score',
    COUNT(*) - COUNT(sentiment_score)
FROM reviews
UNION ALL
SELECT 
    'sentiment_label',
    COUNT(*) - COUNT(sentiment_label)
FROM reviews;

-- Query 14: Sentiment Score Statistics
SELECT 
    ROUND(AVG(sentiment_score), 4) AS avg_sentiment_score,
    ROUND(MIN(sentiment_score), 4) AS min_sentiment_score,
    ROUND(MAX(sentiment_score), 4) AS max_sentiment_score,
    ROUND(STDDEV(sentiment_score), 4) AS stddev_sentiment_score
FROM reviews
WHERE sentiment_score IS NOT NULL;

-- Query 15: Database Size and Table Statistics
SELECT 
    pg_size_pretty(pg_database_size('bank_reviews')) AS database_size,
    (SELECT COUNT(*) FROM banks) AS bank_count,
    (SELECT COUNT(*) FROM reviews) AS review_count;

-- Query 16: Table Row Counts and Disk Usage
SELECT 
    tablename,
    n_live_tup AS row_count,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

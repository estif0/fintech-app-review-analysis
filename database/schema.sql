-- Database Schema for Bank Reviews Analysis
-- Created: 2025-12-01
-- Database: bank_reviews

-- Drop tables if they exist (for clean re-creation)
DROP TABLE IF EXISTS reviews CASCADE;
DROP TABLE IF EXISTS banks CASCADE;

-- =======================
-- Table: banks
-- =======================
-- Stores information about the three Ethiopian banks
CREATE TABLE banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) NOT NULL UNIQUE,
    app_name VARCHAR(200) NOT NULL,
    app_id VARCHAR(200) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert bank data
INSERT INTO banks (bank_name, app_name, app_id) VALUES
    ('CBE', 'Commercial Bank of Ethiopia Mobile', 'com.combanketh.mobilebanking'),
    ('BOA', 'Bank of Abyssinia', 'com.boa.boaMobileBanking'),
    ('Dashen', 'Dashen Bank', 'com.cr2.amolelight');

-- =======================
-- Table: reviews
-- =======================
-- Stores review data with sentiment and thematic analysis results
CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    bank_id INTEGER NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    review_text TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    review_date DATE NOT NULL,
    sentiment_label VARCHAR(20),
    sentiment_score DECIMAL(5,4),
    pos_score DECIMAL(5,4),
    neu_score DECIMAL(5,4),
    neg_score DECIMAL(5,4),
    rating_adjusted BOOLEAN DEFAULT FALSE,
    identified_themes TEXT,
    preprocessed_text TEXT,
    source VARCHAR(50) DEFAULT 'Google Play',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =======================
-- Indexes for Performance
-- =======================
-- Index on bank_id for faster joins
CREATE INDEX idx_reviews_bank_id ON reviews(bank_id);

-- Index on rating for filtering by rating
CREATE INDEX idx_reviews_rating ON reviews(rating);

-- Index on sentiment_label for sentiment analysis queries
CREATE INDEX idx_reviews_sentiment ON reviews(sentiment_label);

-- Index on review_date for time-series analysis
CREATE INDEX idx_reviews_date ON reviews(review_date);

-- Full-text search index on review_text for keyword searches
CREATE INDEX idx_reviews_text_search ON reviews USING gin(to_tsvector('english', review_text));

-- =======================
-- Comments/Documentation
-- =======================
COMMENT ON TABLE banks IS 'Stores information about Ethiopian banks and their mobile banking apps';
COMMENT ON TABLE reviews IS 'Stores scraped and analyzed Google Play Store reviews with sentiment and thematic data';

COMMENT ON COLUMN reviews.bank_id IS 'Foreign key reference to banks table';
COMMENT ON COLUMN reviews.review_text IS 'Original review text from Google Play Store';
COMMENT ON COLUMN reviews.rating IS 'User rating (1-5 stars)';
COMMENT ON COLUMN reviews.review_date IS 'Date when the review was posted';
COMMENT ON COLUMN reviews.sentiment_label IS 'Sentiment classification: Positive, Negative, or Neutral';
COMMENT ON COLUMN reviews.sentiment_score IS 'VADER compound sentiment score (-1 to 1)';
COMMENT ON COLUMN reviews.pos_score IS 'VADER positive sentiment score';
COMMENT ON COLUMN reviews.neu_score IS 'VADER neutral sentiment score';
COMMENT ON COLUMN reviews.neg_score IS 'VADER negative sentiment score';
COMMENT ON COLUMN reviews.rating_adjusted IS 'Flag indicating if sentiment was adjusted based on rating';
COMMENT ON COLUMN reviews.identified_themes IS 'Comma-separated list of themes identified in the review';
COMMENT ON COLUMN reviews.preprocessed_text IS 'Cleaned and preprocessed review text for NLP analysis';
COMMENT ON COLUMN reviews.source IS 'Data source (Google Play, App Store, etc.)';

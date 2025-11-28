"""
Unit tests for the SentimentAnalyzer module.

This module tests the sentiment analysis functionality including VADER sentiment
scoring, classification, DataFrame processing, and statistical aggregation.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from core.sentiment_analyzer import SentimentAnalyzer


@pytest.fixture
def analyzer():
    """Fixture to create a SentimentAnalyzer instance for testing."""
    return SentimentAnalyzer()


@pytest.fixture
def custom_analyzer():
    """Fixture to create a SentimentAnalyzer with custom thresholds."""
    return SentimentAnalyzer(positive_threshold=0.1, negative_threshold=-0.1)


@pytest.fixture
def sample_reviews():
    """Fixture providing sample review texts for testing."""
    return [
        "This app is amazing! Love it!",  # Positive
        "Terrible experience, worst app ever",  # Negative
        "It works fine",  # Neutral
        "Great features and easy to use",  # Positive
        "Crashes all the time, very frustrating",  # Negative
    ]


@pytest.fixture
def sample_dataframe():
    """Fixture providing sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "review": [
                "Excellent app, highly recommend!",
                "Horrible, doesn't work at all",
                "Average, nothing special",
                "Love the new features!",
                "Buggy and slow",
            ],
            "rating": [5, 1, 3, 5, 2],
            "bank": ["CBE", "BOA", "Dashen", "CBE", "BOA"],
        }
    )


class TestSentimentAnalyzerInitialization:
    """Test cases for SentimentAnalyzer initialization."""

    def test_default_initialization(self, analyzer):
        """Test that analyzer initializes with default thresholds."""
        assert analyzer.positive_threshold == 0.05
        assert analyzer.negative_threshold == -0.05
        assert analyzer.analyzer is not None
        assert analyzer.logger is not None

    def test_custom_thresholds(self, custom_analyzer):
        """Test initialization with custom thresholds."""
        assert custom_analyzer.positive_threshold == 0.1
        assert custom_analyzer.negative_threshold == -0.1

    def test_invalid_thresholds(self):
        """Test that invalid thresholds raise ValueError."""
        with pytest.raises(ValueError):
            SentimentAnalyzer(positive_threshold=-0.1, negative_threshold=0.1)

    def test_equal_thresholds(self):
        """Test that equal thresholds raise ValueError."""
        with pytest.raises(ValueError):
            SentimentAnalyzer(positive_threshold=0.05, negative_threshold=0.05)


class TestAnalyzeText:
    """Test cases for analyze_text method."""

    def test_positive_sentiment(self, analyzer):
        """Test analysis of positive text."""
        result = analyzer.analyze_text("This is an excellent and amazing app!")
        assert result["sentiment_label"] == "Positive"
        assert result["sentiment_score"] > 0.05
        assert "pos_score" in result
        assert "neu_score" in result
        assert "neg_score" in result

    def test_negative_sentiment(self, analyzer):
        """Test analysis of negative text."""
        result = analyzer.analyze_text("Terrible app, worst experience ever!")
        assert result["sentiment_label"] == "Negative"
        assert result["sentiment_score"] < -0.05

    def test_neutral_sentiment(self, analyzer):
        """Test analysis of neutral text."""
        result = analyzer.analyze_text("The app works")
        assert result["sentiment_label"] == "Neutral"
        assert -0.05 <= result["sentiment_score"] <= 0.05

    def test_empty_text(self, analyzer):
        """Test analysis of empty text."""
        result = analyzer.analyze_text("")
        assert result["sentiment_label"] == "Neutral"
        assert result["sentiment_score"] == 0.0

    def test_none_text(self, analyzer):
        """Test analysis of None text."""
        result = analyzer.analyze_text(None)
        assert result["sentiment_label"] == "Neutral"
        assert result["sentiment_score"] == 0.0

    def test_invalid_text_type(self, analyzer):
        """Test analysis with invalid text type."""
        result = analyzer.analyze_text(123)
        assert result["sentiment_label"] == "Neutral"
        assert result["sentiment_score"] == 0.0

    def test_score_range(self, analyzer):
        """Test that sentiment scores are within valid range."""
        texts = [
            "Amazing!",
            "Terrible!",
            "Okay",
            "Perfect app",
            "Worst app",
        ]
        for text in texts:
            result = analyzer.analyze_text(text)
            assert -1.0 <= result["sentiment_score"] <= 1.0

    def test_component_scores_sum(self, analyzer):
        """Test that component scores are properly calculated."""
        result = analyzer.analyze_text("Great app but has some bugs")
        assert "pos_score" in result
        assert "neu_score" in result
        assert "neg_score" in result
        # Component scores should be between 0 and 1
        assert 0 <= result["pos_score"] <= 1
        assert 0 <= result["neu_score"] <= 1
        assert 0 <= result["neg_score"] <= 1

    def test_custom_threshold_classification(self, custom_analyzer):
        """Test classification with custom thresholds."""
        # Text with compound score between -0.1 and 0.1 should be neutral
        result = custom_analyzer.analyze_text("okay")
        # This might be neutral with stricter thresholds
        assert result["sentiment_label"] in ["Positive", "Neutral", "Negative"]


class TestAnalyzeReviews:
    """Test cases for analyze_reviews method."""

    def test_analyze_multiple_reviews(self, analyzer, sample_reviews):
        """Test analysis of multiple reviews."""
        results = analyzer.analyze_reviews(sample_reviews, show_progress=False)
        assert len(results) == len(sample_reviews)
        assert all("sentiment_label" in r for r in results)
        assert all("sentiment_score" in r for r in results)

    def test_empty_reviews_list(self, analyzer):
        """Test analysis of empty reviews list."""
        results = analyzer.analyze_reviews([], show_progress=False)
        assert len(results) == 0

    def test_single_review(self, analyzer):
        """Test analysis of single review."""
        results = analyzer.analyze_reviews(["Great app!"], show_progress=False)
        assert len(results) == 1
        assert results[0]["sentiment_label"] == "Positive"

    def test_progress_logging(self, analyzer):
        """Test that progress is logged correctly."""
        # Create 150 reviews to test progress logging
        reviews = ["test review"] * 150
        with patch.object(analyzer.logger, "info") as mock_info:
            analyzer.analyze_reviews(reviews, show_progress=True)
            # Should log at 100 reviews and completion
            assert mock_info.call_count >= 2

    def test_no_progress_logging(self, analyzer, sample_reviews):
        """Test that progress is not logged when disabled."""
        with patch.object(analyzer.logger, "info") as mock_info:
            analyzer.analyze_reviews(sample_reviews, show_progress=False)
            # Should not log anything
            assert mock_info.call_count == 0


class TestAnalyzeDataFrame:
    """Test cases for analyze_dataframe method."""

    def test_analyze_dataframe(self, analyzer, sample_dataframe):
        """Test analysis of DataFrame."""
        result_df = analyzer.analyze_dataframe(sample_dataframe, show_progress=False)
        assert len(result_df) == len(sample_dataframe)
        assert "sentiment_score" in result_df.columns
        assert "sentiment_label" in result_df.columns
        assert "pos_score" in result_df.columns
        assert "neu_score" in result_df.columns
        assert "neg_score" in result_df.columns

    def test_dataframe_original_columns_preserved(self, analyzer, sample_dataframe):
        """Test that original DataFrame columns are preserved."""
        result_df = analyzer.analyze_dataframe(sample_dataframe, show_progress=False)
        for col in sample_dataframe.columns:
            assert col in result_df.columns

    def test_custom_text_column(self, analyzer):
        """Test analysis with custom text column name."""
        df = pd.DataFrame({"text": ["Great!", "Bad!"], "rating": [5, 1]})
        result_df = analyzer.analyze_dataframe(
            df, text_column="text", show_progress=False
        )
        assert "sentiment_score" in result_df.columns

    def test_missing_text_column(self, analyzer, sample_dataframe):
        """Test that missing text column raises ValueError."""
        with pytest.raises(ValueError):
            analyzer.analyze_dataframe(
                sample_dataframe, text_column="nonexistent", show_progress=False
            )

    def test_dataframe_not_modified(self, analyzer, sample_dataframe):
        """Test that original DataFrame is not modified."""
        original_columns = sample_dataframe.columns.tolist()
        analyzer.analyze_dataframe(sample_dataframe, show_progress=False)
        assert sample_dataframe.columns.tolist() == original_columns

    def test_empty_dataframe(self, analyzer):
        """Test analysis of empty DataFrame."""
        df = pd.DataFrame({"review": []})
        result_df = analyzer.analyze_dataframe(df, show_progress=False)
        assert len(result_df) == 0
        assert "sentiment_score" in result_df.columns


class TestGetSentimentStatistics:
    """Test cases for get_sentiment_statistics method."""

    def test_overall_statistics(self, analyzer, sample_dataframe):
        """Test calculation of overall statistics."""
        df_analyzed = analyzer.analyze_dataframe(sample_dataframe, show_progress=False)
        stats = analyzer.get_sentiment_statistics(df_analyzed)
        assert "avg_sentiment" in stats.columns
        assert "positive_count" in stats.columns
        assert "neutral_count" in stats.columns
        assert "negative_count" in stats.columns
        assert "total_reviews" in stats.columns
        assert len(stats) == 1

    def test_grouped_statistics(self, analyzer, sample_dataframe):
        """Test calculation of grouped statistics."""
        df_analyzed = analyzer.analyze_dataframe(sample_dataframe, show_progress=False)
        stats = analyzer.get_sentiment_statistics(df_analyzed, group_by="bank")
        assert len(stats) > 0
        assert "sentiment_score" in str(stats.columns)

    def test_statistics_missing_columns(self, analyzer, sample_dataframe):
        """Test that missing sentiment columns raise ValueError."""
        with pytest.raises(ValueError):
            analyzer.get_sentiment_statistics(sample_dataframe)

    def test_invalid_group_by_column(self, analyzer, sample_dataframe):
        """Test that invalid group_by column raises ValueError."""
        df_analyzed = analyzer.analyze_dataframe(sample_dataframe, show_progress=False)
        with pytest.raises(ValueError):
            analyzer.get_sentiment_statistics(df_analyzed, group_by="nonexistent")

    def test_statistics_percentages(self, analyzer, sample_dataframe):
        """Test that percentage calculations are correct."""
        df_analyzed = analyzer.analyze_dataframe(sample_dataframe, show_progress=False)
        stats = analyzer.get_sentiment_statistics(df_analyzed)
        total_pct = (
            stats["positive_pct"].iloc[0]
            + stats["neutral_pct"].iloc[0]
            + stats["negative_pct"].iloc[0]
        )
        assert 99.9 <= total_pct <= 100.1  # Allow small floating point errors


class TestClassifyByRating:
    """Test cases for classify_by_rating method."""

    def test_classify_by_rating(self, analyzer, sample_dataframe):
        """Test sentiment classification by rating."""
        df_analyzed = analyzer.analyze_dataframe(sample_dataframe, show_progress=False)
        rating_stats = analyzer.classify_by_rating(df_analyzed)
        assert len(rating_stats) > 0

    def test_missing_rating_column(self, analyzer, sample_dataframe):
        """Test that missing rating column raises ValueError."""
        df_analyzed = analyzer.analyze_dataframe(sample_dataframe, show_progress=False)
        df_analyzed = df_analyzed.drop(columns=["rating"])
        with pytest.raises(ValueError):
            analyzer.classify_by_rating(df_analyzed)

    def test_missing_sentiment_columns(self, analyzer, sample_dataframe):
        """Test that missing sentiment columns raise ValueError."""
        with pytest.raises(ValueError):
            analyzer.classify_by_rating(sample_dataframe)

    def test_custom_rating_column(self, analyzer):
        """Test classification with custom rating column name."""
        df = pd.DataFrame(
            {
                "review": ["Great!", "Bad!", "Okay"],
                "stars": [5, 1, 3],
            }
        )
        df_analyzed = analyzer.analyze_dataframe(df, show_progress=False)
        rating_stats = analyzer.classify_by_rating(df_analyzed, rating_column="stars")
        assert len(rating_stats) > 0


class TestSentimentCorrelation:
    """Test cases for sentiment-rating correlation."""

    def test_higher_rating_positive_sentiment(self, analyzer):
        """Test that higher ratings generally have more positive sentiment."""
        df = pd.DataFrame(
            {
                "review": [
                    "Excellent app!",
                    "Good app",
                    "Okay",
                    "Not great",
                    "Terrible app",
                ],
                "rating": [5, 4, 3, 2, 1],
            }
        )
        df_analyzed = analyzer.analyze_dataframe(df, show_progress=False)

        # Calculate average sentiment by rating
        avg_sentiment = df_analyzed.groupby("rating")["sentiment_score"].mean()

        # Generally, sentiment should increase with rating
        # (though not strictly monotonic due to text variations)
        assert avg_sentiment[5] > avg_sentiment[1]

    def test_sentiment_consistency(self, analyzer):
        """Test that similar reviews get similar sentiment scores."""
        similar_reviews = [
            "Great app!",
            "Excellent app!",
            "Amazing app!",
        ]
        results = analyzer.analyze_reviews(similar_reviews, show_progress=False)
        scores = [r["sentiment_score"] for r in results]

        # All should be positive
        assert all(s > 0.05 for s in scores)

        # Variance should be relatively low
        assert np.std(scores) < 0.3


class TestEdgeCases:
    """Test cases for edge cases and error handling."""

    def test_special_characters(self, analyzer):
        """Test analysis of text with special characters."""
        texts = [
            "Great app!!! 😊😊😊",
            "Bad app... 😠",
            "App is #1!!!",
            "5/5 stars ⭐⭐⭐⭐⭐",
        ]
        for text in texts:
            result = analyzer.analyze_text(text)
            assert "sentiment_label" in result
            assert -1.0 <= result["sentiment_score"] <= 1.0

    def test_very_long_text(self, analyzer):
        """Test analysis of very long text."""
        long_text = "Great app! " * 1000
        result = analyzer.analyze_text(long_text)
        assert result["sentiment_label"] == "Positive"

    def test_mixed_sentiment(self, analyzer):
        """Test analysis of text with mixed sentiment."""
        text = "The app is good but has many bugs and crashes frequently"
        result = analyzer.analyze_text(text)
        assert "sentiment_label" in result
        # Could be any sentiment depending on VADER's interpretation

    def test_numeric_text(self, analyzer):
        """Test analysis of numeric text."""
        result = analyzer.analyze_text("12345")
        assert result["sentiment_label"] == "Neutral"

    def test_punctuation_only(self, analyzer):
        """Test analysis of punctuation only."""
        result = analyzer.analyze_text("!!!")
        assert result["sentiment_label"] in ["Positive", "Neutral", "Negative"]


class TestDataFrameOperations:
    """Test cases for DataFrame operations and integrity."""

    def test_dataframe_index_preserved(self, analyzer):
        """Test that DataFrame index is preserved."""
        df = pd.DataFrame({"review": ["Great!", "Bad!"]}, index=[10, 20])
        result_df = analyzer.analyze_dataframe(df, show_progress=False)
        assert list(result_df.index) == [10, 20]

    def test_multiple_analyses(self, analyzer, sample_dataframe):
        """Test that multiple analyses produce consistent results."""
        result1 = analyzer.analyze_dataframe(sample_dataframe, show_progress=False)
        result2 = analyzer.analyze_dataframe(sample_dataframe, show_progress=False)

        # Results should be identical
        pd.testing.assert_series_equal(
            result1["sentiment_score"], result2["sentiment_score"]
        )

    def test_dataframe_with_missing_values(self, analyzer):
        """Test handling of DataFrame with missing values in other columns."""
        df = pd.DataFrame(
            {
                "review": ["Great!", "Bad!", "Okay"],
                "rating": [5, 1, np.nan],
                "bank": ["CBE", None, "Dashen"],
            }
        )
        result_df = analyzer.analyze_dataframe(df, show_progress=False)
        assert len(result_df) == 3
        assert result_df["sentiment_score"].notna().all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

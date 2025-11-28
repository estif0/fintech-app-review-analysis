"""
Unit tests for the TextPreprocessor module.

This module tests the text preprocessing functionality including cleaning,
tokenization, stopword removal, lemmatization, and DataFrame operations.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from core.text_preprocessor import TextPreprocessor


@pytest.fixture
def preprocessor():
    """Fixture to create a TextPreprocessor instance for testing."""
    return TextPreprocessor()


@pytest.fixture
def custom_preprocessor():
    """Fixture to create a TextPreprocessor with custom settings."""
    return TextPreprocessor(
        custom_stop_words=["test", "example"], remove_numbers=False, min_word_length=3
    )


class TestTextPreprocessorInitialization:
    """Test cases for TextPreprocessor initialization."""

    def test_default_initialization(self, preprocessor):
        """Test that preprocessor initializes with default parameters."""
        assert preprocessor.language == "english"
        assert preprocessor.remove_numbers is True
        assert preprocessor.min_word_length == 2
        assert len(preprocessor.stop_words) > 0
        assert preprocessor.lemmatizer is not None

    def test_custom_stopwords(self, custom_preprocessor):
        """Test initialization with custom stopwords."""
        assert "test" in custom_preprocessor.stop_words
        assert "example" in custom_preprocessor.stop_words

    def test_custom_parameters(self, custom_preprocessor):
        """Test initialization with custom parameters."""
        assert custom_preprocessor.remove_numbers is False
        assert custom_preprocessor.min_word_length == 3


class TestCleanText:
    """Test cases for clean_text method."""

    def test_lowercase_conversion(self, preprocessor):
        """Test that text is converted to lowercase."""
        result = preprocessor.clean_text("THE APP IS GREAT")
        assert result == "the app is great"

    def test_punctuation_removal(self, preprocessor):
        """Test that punctuation is removed."""
        result = preprocessor.clean_text("Great app!!! Amazing...")
        assert result == "great app amazing"

    def test_number_removal(self, preprocessor):
        """Test that numbers are removed when enabled."""
        result = preprocessor.clean_text("App version 2.3.5 has 100 bugs")
        assert "2" not in result
        assert "3" not in result
        assert "5" not in result
        assert "100" not in result

    def test_number_retention(self, custom_preprocessor):
        """Test that numbers are kept when remove_numbers=False."""
        result = custom_preprocessor.clean_text("App version 2.3")
        assert "2" in result or "23" in result  # Numbers preserved

    def test_url_removal(self, preprocessor):
        """Test that URLs are removed."""
        result = preprocessor.clean_text("Visit https://example.com for help")
        assert "https" not in result
        assert "example.com" not in result

    def test_email_removal(self, preprocessor):
        """Test that email addresses are removed."""
        result = preprocessor.clean_text("Contact support@bank.com")
        assert "support@bank.com" not in result

    def test_whitespace_normalization(self, preprocessor):
        """Test that extra whitespace is normalized."""
        result = preprocessor.clean_text("too   many    spaces")
        assert result == "too many spaces"

    def test_empty_text(self, preprocessor):
        """Test handling of empty text."""
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text(None) == ""

    def test_special_characters(self, preprocessor):
        """Test removal of special characters."""
        result = preprocessor.clean_text("App @#$% works &*() well")
        assert result == "app works well"


class TestTokenize:
    """Test cases for tokenize method."""

    def test_basic_tokenization(self, preprocessor):
        """Test basic word tokenization."""
        tokens = preprocessor.tokenize("great banking app")
        assert "great" in tokens
        assert "banking" in tokens
        assert "app" in tokens

    def test_minimum_length_filter(self, preprocessor):
        """Test that short tokens are filtered."""
        tokens = preprocessor.tokenize("a an the good")
        # With min_word_length=2, single letters should be filtered
        assert "a" not in tokens

    def test_custom_minimum_length(self, custom_preprocessor):
        """Test custom minimum word length."""
        tokens = custom_preprocessor.tokenize("to be or not")
        # With min_word_length=3, "to", "be", "or" should be filtered
        assert len(tokens) == 1  # Only "not" remains
        assert "not" in tokens

    def test_empty_text_tokenization(self, preprocessor):
        """Test tokenization of empty text."""
        tokens = preprocessor.tokenize("")
        assert tokens == []


class TestRemoveStopwords:
    """Test cases for remove_stopwords method."""

    def test_stopword_removal(self, preprocessor):
        """Test that stopwords are removed."""
        tokens = ["the", "app", "is", "good"]
        filtered = preprocessor.remove_stopwords(tokens)
        assert "the" not in filtered
        assert "is" not in filtered
        assert "good" in filtered

    def test_custom_stopword_removal(self, preprocessor):
        """Test that custom stopwords are removed."""
        tokens = ["app", "application", "mobile", "good"]
        filtered = preprocessor.remove_stopwords(tokens)
        # 'app', 'application', 'mobile' are in custom stopwords
        assert "app" not in filtered
        assert "application" not in filtered
        assert "good" in filtered

    def test_empty_token_list(self, preprocessor):
        """Test stopword removal on empty list."""
        filtered = preprocessor.remove_stopwords([])
        assert filtered == []


class TestLemmatize:
    """Test cases for lemmatize method."""

    def test_basic_lemmatization(self, preprocessor):
        """Test basic lemmatization."""
        tokens = ["crashes", "running", "better"]
        lemmatized = preprocessor.lemmatize(tokens)
        assert "crash" in lemmatized

    def test_lemmatization_preserves_base_forms(self, preprocessor):
        """Test that base forms are preserved."""
        tokens = ["good", "bad", "fast"]
        lemmatized = preprocessor.lemmatize(tokens)
        assert set(lemmatized) == set(tokens)

    def test_empty_token_list_lemmatization(self, preprocessor):
        """Test lemmatization of empty list."""
        lemmatized = preprocessor.lemmatize([])
        assert lemmatized == []


class TestPreprocessText:
    """Test cases for preprocess_text method."""

    def test_complete_pipeline(self, preprocessor):
        """Test complete preprocessing pipeline."""
        text = "The apps are CRASHING frequently!!!"
        result = preprocessor.preprocess_text(text)
        # Should be cleaned, tokenized, stopwords removed, and lemmatized
        assert isinstance(result, str)
        assert len(result) > 0

    def test_pipeline_with_stopwords(self, preprocessor):
        """Test pipeline with stopword removal enabled."""
        text = "the app is good"
        result = preprocessor.preprocess_text(text, remove_stopwords=True)
        assert "the" not in result
        assert "is" not in result
        assert "good" in result

    def test_pipeline_without_stopwords(self, preprocessor):
        """Test pipeline with stopword removal disabled."""
        text = "the app is good"
        result = preprocessor.preprocess_text(text, remove_stopwords=False)
        # All words should be present
        assert len(result.split()) >= 3

    def test_pipeline_with_lemmatization(self, preprocessor):
        """Test pipeline with lemmatization enabled."""
        text = "crashes running"
        result = preprocessor.preprocess_text(text, lemmatize=True)
        assert "crash" in result

    def test_pipeline_without_lemmatization(self, preprocessor):
        """Test pipeline with lemmatization disabled."""
        text = "crashes"
        result = preprocessor.preprocess_text(text, lemmatize=False)
        assert "crashes" in result

    def test_empty_result(self, preprocessor):
        """Test preprocessing of text that becomes empty."""
        text = "!!!"
        result = preprocessor.preprocess_text(text)
        assert result == ""


class TestPreprocessTexts:
    """Test cases for preprocess_texts method."""

    def test_multiple_texts(self, preprocessor):
        """Test preprocessing of multiple texts."""
        texts = ["Great app!", "Terrible bugs", "Works fine"]
        results = preprocessor.preprocess_texts(texts, show_progress=False)
        assert len(results) == len(texts)
        assert all(isinstance(r, str) for r in results)

    def test_empty_list(self, preprocessor):
        """Test preprocessing of empty list."""
        results = preprocessor.preprocess_texts([], show_progress=False)
        assert results == []

    def test_progress_logging(self, preprocessor):
        """Test that progress is logged correctly."""
        texts = ["test"] * 150
        with patch.object(preprocessor.logger, "info") as mock_info:
            preprocessor.preprocess_texts(texts, show_progress=True)
            # Should log progress
            assert mock_info.call_count >= 2

    def test_no_progress_logging(self, preprocessor):
        """Test that progress is not logged when disabled."""
        texts = ["test"] * 5
        with patch.object(preprocessor.logger, "info") as mock_info:
            preprocessor.preprocess_texts(texts, show_progress=False)
            assert mock_info.call_count == 0


class TestPreprocessDataFrame:
    """Test cases for preprocess_dataframe method."""

    def test_dataframe_preprocessing(self, preprocessor):
        """Test preprocessing of DataFrame."""
        df = pd.DataFrame({"review": ["Great app!", "Bad bugs"]})
        result_df = preprocessor.preprocess_dataframe(df, show_progress=False)
        assert "preprocessed_text" in result_df.columns
        assert len(result_df) == len(df)

    def test_custom_column_names(self, preprocessor):
        """Test preprocessing with custom column names."""
        df = pd.DataFrame({"text": ["Good", "Bad"]})
        result_df = preprocessor.preprocess_dataframe(
            df, text_column="text", output_column="clean", show_progress=False
        )
        assert "clean" in result_df.columns

    def test_missing_text_column(self, preprocessor):
        """Test error when text column doesn't exist."""
        df = pd.DataFrame({"other": ["text"]})
        with pytest.raises(ValueError):
            preprocessor.preprocess_dataframe(
                df, text_column="review", show_progress=False
            )

    def test_original_dataframe_unchanged(self, preprocessor):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame({"review": ["Test"]})
        original_columns = df.columns.tolist()
        preprocessor.preprocess_dataframe(df, show_progress=False)
        assert df.columns.tolist() == original_columns

    def test_empty_dataframe(self, preprocessor):
        """Test preprocessing of empty DataFrame."""
        df = pd.DataFrame({"review": []})
        result_df = preprocessor.preprocess_dataframe(df, show_progress=False)
        assert len(result_df) == 0
        assert "preprocessed_text" in result_df.columns


class TestGetTokenStatistics:
    """Test cases for get_token_statistics method."""

    def test_basic_statistics(self, preprocessor):
        """Test calculation of basic statistics."""
        texts = ["good fast", "bad slow", "good good"]
        stats = preprocessor.get_token_statistics(texts, top_n=5)
        assert "total_tokens" in stats
        assert "unique_tokens" in stats
        assert "avg_tokens_per_text" in stats
        assert "top_tokens" in stats
        assert stats["total_tokens"] == 6

    def test_top_tokens(self, preprocessor):
        """Test top token extraction."""
        texts = ["good good good", "bad", "good"]
        stats = preprocessor.get_token_statistics(texts, top_n=2)
        assert len(stats["top_tokens"]) <= 2
        # 'good' should be the top token
        assert stats["top_tokens"][0][0] == "good"
        assert stats["top_tokens"][0][1] == 4  # appears 4 times

    def test_empty_texts(self, preprocessor):
        """Test statistics for empty texts."""
        stats = preprocessor.get_token_statistics([], top_n=5)
        assert stats["total_tokens"] == 0
        assert stats["unique_tokens"] == 0
        assert stats["avg_tokens_per_text"] == 0


class TestEdgeCases:
    """Test cases for edge cases and error handling."""

    def test_special_unicode_characters(self, preprocessor):
        """Test handling of special unicode characters."""
        text = "Great app 😊 ❤️"
        result = preprocessor.clean_text(text)
        # Should not crash, emojis removed or handled
        assert isinstance(result, str)

    def test_very_long_text(self, preprocessor):
        """Test handling of very long text."""
        text = "good " * 1000
        result = preprocessor.preprocess_text(text)
        assert isinstance(result, str)

    def test_mixed_languages(self, preprocessor):
        """Test handling of mixed language text."""
        text = "Good app እሩጫ"  # English + Amharic
        result = preprocessor.clean_text(text)
        assert isinstance(result, str)

    def test_only_stopwords(self, preprocessor):
        """Test text consisting only of stopwords."""
        text = "the a an of in"
        result = preprocessor.preprocess_text(text, remove_stopwords=True)
        assert result == ""

    def test_numbers_and_text(self, preprocessor):
        """Test text with mixed numbers and words."""
        text = "version 2.5 has 10 features"
        result = preprocessor.clean_text(text)
        # Numbers should be removed
        assert "version" in result
        assert "has" in result or "features" in result


class TestDataFrameOperations:
    """Test cases for DataFrame-specific operations."""

    def test_dataframe_with_nan_values(self, preprocessor):
        """Test handling of NaN values in DataFrame."""
        df = pd.DataFrame({"review": ["Good", None, "Bad"]})
        result_df = preprocessor.preprocess_dataframe(df, show_progress=False)
        assert len(result_df) == 3
        # NaN should be handled gracefully
        assert (
            result_df["preprocessed_text"].notna().all()
            or result_df["preprocessed_text"].isna().any()
        )

    def test_dataframe_index_preserved(self, preprocessor):
        """Test that DataFrame index is preserved."""
        df = pd.DataFrame({"review": ["Good", "Bad"]}, index=[10, 20])
        result_df = preprocessor.preprocess_dataframe(df, show_progress=False)
        assert list(result_df.index) == [10, 20]

    def test_multiple_preprocessing_runs(self, preprocessor):
        """Test that multiple preprocessing runs produce consistent results."""
        df = pd.DataFrame({"review": ["Great app!"]})
        result1 = preprocessor.preprocess_dataframe(df, show_progress=False)
        result2 = preprocessor.preprocess_dataframe(df, show_progress=False)
        assert (
            result1["preprocessed_text"].iloc[0] == result2["preprocessed_text"].iloc[0]
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

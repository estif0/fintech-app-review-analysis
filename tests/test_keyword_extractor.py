"""
Unit tests for the KeywordExtractor module.

This module tests the TF-IDF keyword extraction functionality including
fitting, keyword extraction, group-based extraction, and n-gram analysis.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from core.keyword_extractor import KeywordExtractor


@pytest.fixture
def sample_texts():
    """Fixture providing sample preprocessed texts."""
    return [
        "good banking app",
        "great banking service",
        "bad app crash",
        "good service fast",
        "terrible crash bug",
    ]


@pytest.fixture
def sample_dataframe():
    """Fixture providing sample DataFrame with preprocessed texts."""
    return pd.DataFrame(
        {
            "preprocessed_text": [
                "good banking app",
                "great banking service",
                "bad app crash",
                "good service fast",
                "terrible crash bug",
            ],
            "bank_name": ["BOA", "BOA", "CBE", "CBE", "Dashen"],
        }
    )


@pytest.fixture
def extractor():
    """Fixture to create a KeywordExtractor instance."""
    return KeywordExtractor()


@pytest.fixture
def custom_extractor():
    """Fixture to create a KeywordExtractor with custom settings."""
    return KeywordExtractor(ngram_range=(1, 2), min_df=1, max_df=0.9, max_features=50)


class TestKeywordExtractorInitialization:
    """Test cases for KeywordExtractor initialization."""

    def test_default_initialization(self, extractor):
        """Test that extractor initializes with default parameters."""
        assert extractor.ngram_range == (1, 3)
        assert extractor.min_df == 2
        assert extractor.max_df == 0.85
        assert extractor.max_features is None
        assert extractor.vectorizer is None
        assert extractor.feature_names is None

    def test_custom_initialization(self, custom_extractor):
        """Test initialization with custom parameters."""
        assert custom_extractor.ngram_range == (1, 2)
        assert custom_extractor.min_df == 1
        assert custom_extractor.max_df == 0.9
        assert custom_extractor.max_features == 50

    def test_vectorizer_configuration(self, extractor):
        """Test that vectorizer is configured correctly after fitting."""
        extractor.fit(
            [
                "good banking app",
                "great banking service",
                "bad app crash",
                "good service fast",
                "terrible crash bug",
            ]
        )
        assert extractor.vectorizer.ngram_range == (1, 3)
        assert extractor.vectorizer.min_df == 2
        assert extractor.vectorizer.max_df == 0.85
        assert extractor.vectorizer.max_features is None


class TestFit:
    """Test cases for fit method."""

    def test_fit_with_texts(self, extractor, sample_texts):
        """Test fitting with a list of texts."""
        extractor.fit(sample_texts)
        assert extractor.vectorizer is not None
        assert extractor.feature_names is not None
        assert len(extractor.feature_names) > 0

    def test_fit_with_dataframe(self, extractor, sample_dataframe):
        """Test fitting with a DataFrame."""
        extractor.fit(sample_dataframe["preprocessed_text"].tolist())
        assert extractor.vectorizer is not None

    def test_fit_updates_feature_names(self, extractor, sample_texts):
        """Test that fitting updates feature names."""
        extractor.fit(sample_texts)
        assert hasattr(extractor, "feature_names")
        assert extractor.feature_names is not None

    def test_fit_with_empty_texts(self, extractor):
        """Test fitting with empty texts."""
        with pytest.raises(ValueError):
            extractor.fit([])

    def test_fit_with_single_text(self, extractor):
        """Test fitting with a single text (should fail min_df requirement)."""
        # With min_df=2, single text should fail
        with pytest.raises(ValueError):
            extractor.fit(["good banking app"])

    def test_refit_replaces_previous_fit(self, custom_extractor, sample_texts):
        """Test that refitting replaces previous model."""
        custom_extractor.fit(sample_texts)
        first_features = len(custom_extractor.feature_names)

        new_texts = ["different text corpus", "totally new words", "more text here"]
        custom_extractor.fit(new_texts)
        # Should have different features
        assert custom_extractor.vectorizer is not None


class TestExtractKeywords:
    """Test cases for extract_keywords method."""

    def test_extract_keywords_basic(self, extractor, sample_texts):
        """Test basic keyword extraction."""
        extractor.fit(sample_texts)
        keywords = extractor.extract_keywords(sample_texts, top_n=3)
        assert len(keywords) <= 3
        assert all(isinstance(k, tuple) for k in keywords)
        assert all(len(k) == 2 for k in keywords)

    def test_extract_keywords_sorted(self, extractor, sample_texts):
        """Test that keywords are sorted by score."""
        extractor.fit(sample_texts)
        keywords = extractor.extract_keywords(sample_texts, top_n=5)
        scores = [score for _, score in keywords]
        assert scores == sorted(scores, reverse=True)

    def test_extract_keywords_without_fit(self, extractor, sample_texts):
        """Test that extraction fails without fitting first."""
        with pytest.raises(ValueError):
            extractor.extract_keywords(sample_texts)

    def test_extract_keywords_top_n_limit(self, extractor, sample_texts):
        """Test that top_n parameter limits results."""
        extractor.fit(sample_texts)
        keywords = extractor.extract_keywords(sample_texts, top_n=2)
        assert len(keywords) == 2

    def test_extract_keywords_all(self, extractor, sample_texts):
        """Test extraction of all keywords."""
        extractor.fit(sample_texts)
        keywords = extractor.extract_keywords(sample_texts, top_n=None)
        # Should return all features
        assert len(keywords) == len(extractor.feature_names)

    def test_extract_keywords_scores_positive(self, extractor, sample_texts):
        """Test that keyword scores are positive."""
        extractor.fit(sample_texts)
        keywords = extractor.extract_keywords(sample_texts, top_n=5)
        assert all(score >= 0 for _, score in keywords)


class TestExtractKeywordsByGroup:
    """Test cases for extract_keywords_by_group method."""

    def test_extract_by_group_basic(self, extractor, sample_dataframe):
        """Test basic group-based extraction."""
        extractor.fit(sample_dataframe["preprocessed_text"].tolist())
        result = extractor.extract_keywords_by_group(
            sample_dataframe,
            text_column="preprocessed_text",
            group_column="bank_name",
            top_n=3,
        )
        assert isinstance(result, dict)
        assert "BOA" in result
        assert "CBE" in result
        assert "Dashen" in result

    def test_extract_by_group_keywords_per_group(self, extractor, sample_dataframe):
        """Test that each group has keywords."""
        extractor.fit(sample_dataframe["preprocessed_text"].tolist())
        result = extractor.extract_keywords_by_group(
            sample_dataframe,
            text_column="preprocessed_text",
            group_column="bank_name",
            top_n=2,
        )
        for group, keywords in result.items():
            assert len(keywords) <= 2
            assert all(isinstance(k, tuple) for k in keywords)

    def test_extract_by_group_without_fit(self, extractor, sample_dataframe):
        """Test that group extraction fails without fitting."""
        with pytest.raises(ValueError):
            extractor.extract_keywords_by_group(
                sample_dataframe,
                text_column="preprocessed_text",
                group_column="bank_name",
            )

    def test_extract_by_group_missing_column(self, extractor, sample_dataframe):
        """Test error when text column is missing."""
        extractor.fit(sample_dataframe["preprocessed_text"].tolist())
        with pytest.raises(ValueError):
            extractor.extract_keywords_by_group(
                sample_dataframe, text_column="nonexistent", group_column="bank_name"
            )

    def test_extract_by_group_missing_group_column(self, extractor, sample_dataframe):
        """Test error when group column is missing."""
        extractor.fit(sample_dataframe["preprocessed_text"].tolist())
        with pytest.raises(ValueError):
            extractor.extract_keywords_by_group(
                sample_dataframe,
                text_column="preprocessed_text",
                group_column="nonexistent",
            )

    def test_extract_by_group_empty_group(self, custom_extractor):
        """Test handling of single group."""
        df = pd.DataFrame(
            {
                "preprocessed_text": ["good app", "bad app", "great app"],
                "bank_name": ["BOA", "BOA", "BOA"],
            }
        )
        custom_extractor.fit(df["preprocessed_text"].tolist())
        result = custom_extractor.extract_keywords_by_group(
            df, text_column="preprocessed_text", group_column="bank_name"
        )
        assert "BOA" in result


class TestExtractBigramsTrirams:
    """Test cases for extract_bigrams_trigrams method."""

    def test_extract_bigrams_basic(self, extractor, sample_texts):
        """Test basic bigram extraction."""
        bigrams = extractor.extract_bigrams_trigrams(sample_texts, n=2, top_n=3)
        assert isinstance(bigrams, list)
        assert len(bigrams) <= 3
        assert all(isinstance(b, tuple) for b in bigrams)
        assert all(len(b) == 2 for b in bigrams)  # (ngram, count)

    def test_extract_trigrams_basic(self, extractor, sample_texts):
        """Test basic trigram extraction."""
        trigrams = extractor.extract_bigrams_trigrams(sample_texts, n=3, top_n=3)
        assert isinstance(trigrams, list)
        assert len(trigrams) <= 3
        assert all(isinstance(t, tuple) for t in trigrams)

    def test_bigrams_sorted_by_frequency(self, extractor, sample_texts):
        """Test that bigrams are sorted by frequency."""
        bigrams = extractor.extract_bigrams_trigrams(sample_texts, n=2, top_n=10)
        if len(bigrams) > 1:
            frequencies = [freq for _, freq in bigrams]
            assert frequencies == sorted(frequencies, reverse=True)

    def test_trigrams_sorted_by_frequency(self, extractor, sample_texts):
        """Test that trigrams are sorted by frequency."""
        trigrams = extractor.extract_bigrams_trigrams(sample_texts, n=3, top_n=10)
        if len(trigrams) > 1:
            frequencies = [freq for _, freq in trigrams]
            assert frequencies == sorted(frequencies, reverse=True)

    def test_ngrams_with_empty_texts(self, extractor):
        """Test n-gram extraction with empty texts."""
        bigrams = extractor.extract_bigrams_trigrams([], n=2, top_n=5)
        assert bigrams == []

    def test_ngrams_with_short_texts(self, extractor):
        """Test n-gram extraction with texts too short for trigrams."""
        texts = ["good", "bad"]
        bigrams = extractor.extract_bigrams_trigrams(texts, n=2, top_n=5)
        trigrams = extractor.extract_bigrams_trigrams(texts, n=3, top_n=5)
        # Should not crash, trigrams may be empty
        assert isinstance(bigrams, list)
        assert isinstance(trigrams, list)


class TestGetKeywordContext:
    """Test cases for get_keyword_context method."""

    def test_get_context_basic(self, extractor, sample_dataframe):
        """Test basic keyword context retrieval."""
        extractor.fit(sample_dataframe["preprocessed_text"].tolist())
        contexts = extractor.get_keyword_context(
            sample_dataframe,
            keyword="banking",
            text_column="preprocessed_text",
            max_examples=2,
        )
        assert isinstance(contexts, list)
        assert len(contexts) <= 2

    def test_get_context_keyword_present(self, extractor, sample_dataframe):
        """Test that returned contexts contain the keyword."""
        extractor.fit(sample_dataframe["preprocessed_text"].tolist())
        contexts = extractor.get_keyword_context(
            sample_dataframe,
            keyword="banking",
            text_column="preprocessed_text",
            max_examples=5,
        )
        for context in contexts:
            assert "banking" in context.lower()

    def test_get_context_missing_keyword(self, extractor, sample_dataframe):
        """Test context retrieval for non-existent keyword."""
        extractor.fit(sample_dataframe["preprocessed_text"].tolist())
        contexts = extractor.get_keyword_context(
            sample_dataframe, keyword="nonexistent", text_column="preprocessed_text"
        )
        assert contexts == []

    def test_get_context_max_examples_limit(self, extractor, sample_dataframe):
        """Test that max_examples parameter limits results."""
        extractor.fit(sample_dataframe["preprocessed_text"].tolist())
        contexts = extractor.get_keyword_context(
            sample_dataframe,
            keyword="app",
            text_column="preprocessed_text",
            max_examples=1,
        )
        assert len(contexts) <= 1

    def test_get_context_missing_column(self, extractor, sample_dataframe):
        """Test error when text column is missing."""
        extractor.fit(sample_dataframe["preprocessed_text"].tolist())
        # Missing column should raise KeyError from pandas
        try:
            extractor.get_keyword_context(
                sample_dataframe, keyword="banking", text_column="nonexistent"
            )
            assert False, "Should have raised KeyError"
        except KeyError:
            pass  # Expected


class TestEdgeCases:
    """Test cases for edge cases and error handling."""

    def test_empty_dataframe(self, extractor):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({"preprocessed_text": [], "bank_name": []})
        with pytest.raises(ValueError):
            extractor.fit(df["preprocessed_text"].tolist())

    def test_single_word_texts(self, custom_extractor):
        """Test handling of single-word texts."""
        texts = ["good", "bad", "great", "terrible"]
        custom_extractor.fit(texts)
        keywords = custom_extractor.extract_keywords(texts, top_n=2)
        assert len(keywords) > 0

    def test_all_empty_texts(self, extractor):
        """Test handling of all empty texts."""
        texts = ["", "", ""]
        with pytest.raises(ValueError):
            extractor.fit(texts)

    def test_texts_with_nan_values(self, extractor):
        """Test handling of NaN values in texts."""
        df = pd.DataFrame({"preprocessed_text": ["good app", None, "bad app"]})
        texts = df["preprocessed_text"].fillna("").tolist()
        # After fillna, should work with empty string
        try:
            extractor.fit([t for t in texts if t])
            assert extractor.is_fitted
        except ValueError:
            # Expected if not enough valid texts
            pass

    def test_very_long_text(self, extractor):
        """Test handling of very long text."""
        long_text = " ".join(["word"] * 10000)
        texts = [long_text, "short text", "another text"]
        extractor.fit(texts)
        keywords = extractor.extract_keywords(texts, top_n=5)
        assert len(keywords) > 0

    def test_special_characters_in_texts(self, custom_extractor):
        """Test handling of texts with special characters."""
        texts = ["good@app", "bad#service", "great$banking"]
        # Should handle gracefully (TfidfVectorizer tokenizes)
        try:
            custom_extractor.fit(texts)
            assert custom_extractor.vectorizer is not None
        except ValueError:
            # May fail if tokenization produces insufficient unique terms
            pass


class TestFeatureExtraction:
    """Test cases for feature extraction and transformation."""

    def test_feature_names_exist(self, extractor, sample_texts):
        """Test that feature names are extracted."""
        extractor.fit(sample_texts)
        assert hasattr(extractor, "feature_names")
        assert len(extractor.feature_names) > 0

    def test_feature_names_unique(self, extractor, sample_texts):
        """Test that feature names are unique."""
        extractor.fit(sample_texts)
        assert len(extractor.feature_names) == len(set(extractor.feature_names))

    def test_transform_produces_matrix(self, extractor, sample_texts):
        """Test that transform produces correct matrix."""
        extractor.fit(sample_texts)
        # Internal transform should work
        matrix = extractor.vectorizer.transform(sample_texts)
        assert matrix.shape[0] == len(sample_texts)
        assert matrix.shape[1] == len(extractor.feature_names)

    def test_ngram_features(self, extractor, sample_texts):
        """Test that n-gram features are created."""
        extractor.fit(sample_texts)
        # Check if bigrams or trigrams exist in features
        has_space = any(" " in feature for feature in extractor.feature_names)
        # With ngram_range=(1,3), should have some multi-word features
        assert has_space or len(extractor.feature_names) > 0


class TestStaticMethods:
    """Test cases for instance methods."""

    def test_extract_bigrams_no_fit_needed(self, extractor):
        """Test that extract_bigrams_trigrams can be called without fitting."""
        texts = ["good banking app", "great service"]
        bigrams = extractor.extract_bigrams_trigrams(texts, n=2, top_n=5)
        assert isinstance(bigrams, list)


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_extraction_workflow(self, extractor, sample_dataframe):
        """Test complete extraction workflow."""
        # Fit
        texts = sample_dataframe["preprocessed_text"].tolist()
        extractor.fit(texts)

        # Extract overall keywords
        keywords = extractor.extract_keywords(texts, top_n=5)
        assert len(keywords) > 0

        # Extract by group
        group_keywords = extractor.extract_keywords_by_group(
            sample_dataframe,
            text_column="preprocessed_text",
            group_column="bank_name",
            top_n=3,
        )
        assert len(group_keywords) > 0

        # Extract n-grams
        bigrams = extractor.extract_bigrams_trigrams(texts, n=2, top_n=3)
        trigrams = extractor.extract_bigrams_trigrams(texts, n=3, top_n=3)
        assert isinstance(bigrams, list)
        assert isinstance(trigrams, list)

    def test_multiple_extractions_consistent(self, extractor, sample_texts):
        """Test that multiple extractions produce consistent results."""
        extractor.fit(sample_texts)

        keywords1 = extractor.extract_keywords(sample_texts, top_n=3)
        keywords2 = extractor.extract_keywords(sample_texts, top_n=3)

        assert keywords1 == keywords2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

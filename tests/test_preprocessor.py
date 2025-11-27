"""
Unit tests for the ReviewPreprocessor class.

This module contains tests for data preprocessing functionality including
loading, cleaning, validation, and saving operations.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from core.preprocessor import ReviewPreprocessor, PreprocessorError
from core.config import Config


@pytest.fixture
def sample_raw_data():
    """Create sample raw review data for testing."""
    return pd.DataFrame(
        {
            "reviewId": ["rev1", "rev2", "rev3", "rev2", "rev4"],
            "userName": ["User1", "User2", "User3", "User2", "User4"],
            "content": ["Great app!", "Terrible", None, "Terrible", "Good app"],
            "score": [5, 1, 3, 1, 4],
            "thumbsUpCount": [10, 2, 5, 2, 8],
            "at": [
                "2024-01-15 10:30:00",
                "2024-01-16 14:20:00",
                "2024-01-17 09:15:00",
                "2024-01-16 14:20:00",
                "2024-01-18 16:45:00",
            ],
            "replyContent": ["Thank you!", None, None, None, "Thanks!"],
            "bank_name": ["CBE", "BOA", "Dashen", "BOA", "CBE"],
            "bank_code": ["cbe", "boa", "dashen", "boa", "cbe"],
            "app_id": ["com.cbe", "com.boa", "com.dashen", "com.boa", "com.cbe"],
            "scraped_at": ["2024-11-27"] * 5,
        }
    )


@pytest.fixture
def preprocessor():
    """Create a ReviewPreprocessor instance for testing."""
    return ReviewPreprocessor()


@pytest.fixture
def temp_csv_files(tmp_path, sample_raw_data):
    """Create temporary CSV files for testing."""
    csv_file1 = tmp_path / "cbe_reviews_raw.csv"
    csv_file2 = tmp_path / "boa_reviews_raw.csv"

    sample_raw_data.iloc[:2].to_csv(csv_file1, index=False)
    sample_raw_data.iloc[2:4].to_csv(csv_file2, index=False)

    return tmp_path


class TestPreprocessorInitialization:
    """Test ReviewPreprocessor initialization."""

    def test_init_with_default_config(self):
        """Test preprocessor initialization with default config."""
        preprocessor = ReviewPreprocessor()

        assert preprocessor.config is not None
        assert preprocessor.logger is not None
        assert preprocessor.df is None
        assert preprocessor.quality_report == {}

    def test_init_with_custom_config(self):
        """Test preprocessor initialization with custom config."""
        custom_config = Config()
        preprocessor = ReviewPreprocessor(config=custom_config)

        assert preprocessor.config == custom_config


class TestDataLoading:
    """Test data loading functionality."""

    def test_load_raw_data_success(self, preprocessor, temp_csv_files):
        """Test successful loading of raw data."""
        df = preprocessor.load_raw_data(directory=temp_csv_files)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 2 from first file + 2 from second file
        assert "reviewId" in df.columns

    def test_load_raw_data_no_files(self, preprocessor, tmp_path):
        """Test loading when no CSV files exist."""
        with pytest.raises(PreprocessorError, match="No CSV files found"):
            preprocessor.load_raw_data(directory=tmp_path)

    def test_load_raw_data_sets_internal_df(self, preprocessor, temp_csv_files):
        """Test that load_raw_data sets the internal df attribute."""
        df = preprocessor.load_raw_data(directory=temp_csv_files)

        assert preprocessor.df is not None
        assert len(preprocessor.df) == len(df)


class TestDuplicateRemoval:
    """Test duplicate removal functionality."""

    def test_remove_duplicates_with_duplicates(self, preprocessor, sample_raw_data):
        """Test removing duplicate reviews."""
        df_clean = preprocessor.remove_duplicates(sample_raw_data)

        # Should remove row index 3 (duplicate of row 1)
        assert len(df_clean) == 4
        assert len(sample_raw_data) == 5

    def test_remove_duplicates_no_duplicates(self, preprocessor):
        """Test duplicate removal when no duplicates exist."""
        df = pd.DataFrame(
            {
                "reviewId": ["rev1", "rev2", "rev3"],
                "content": ["Text1", "Text2", "Text3"],
                "at": ["2024-01-01", "2024-01-02", "2024-01-03"],
            }
        )

        df_clean = preprocessor.remove_duplicates(df)
        assert len(df_clean) == len(df)

    def test_remove_duplicates_missing_columns(self, preprocessor):
        """Test duplicate removal with missing subset columns."""
        df = pd.DataFrame({"data": [1, 2, 3]})

        df_clean = preprocessor.remove_duplicates(df)
        assert len(df_clean) == len(df)


class TestMissingValueHandling:
    """Test missing value handling functionality."""

    def test_handle_missing_critical_columns(self, preprocessor, sample_raw_data):
        """Test handling of missing critical columns."""
        df_clean = preprocessor.handle_missing_values(sample_raw_data)

        # Should remove row with missing 'content'
        assert len(df_clean) < len(sample_raw_data)
        assert df_clean["content"].notna().all()

    def test_handle_missing_non_critical_columns(self, preprocessor):
        """Test handling of missing non-critical columns."""
        df = pd.DataFrame(
            {
                "content": ["Text1", "Text2"],
                "score": [5, 4],
                "replyContent": [None, "Reply"],
                "thumbsUpCount": [None, 10],
            }
        )

        df_clean = preprocessor.handle_missing_values(df)

        assert len(df_clean) == 2  # No rows removed
        assert df_clean["replyContent"].iloc[0] == ""
        assert df_clean["thumbsUpCount"].iloc[0] == 0


class TestDateNormalization:
    """Test date normalization functionality."""

    def test_normalize_dates_success(self, preprocessor):
        """Test successful date normalization."""
        df = pd.DataFrame({"at": ["2024-01-15 10:30:00", "2024-02-20 14:45:00"]})

        df_normalized = preprocessor.normalize_dates(df)

        assert "date" in df_normalized.columns
        assert df_normalized["date"].iloc[0] == "2024-01-15"
        assert df_normalized["date"].iloc[1] == "2024-02-20"

    def test_normalize_dates_invalid_dates(self, preprocessor):
        """Test date normalization with invalid dates."""
        df = pd.DataFrame({"at": ["2024-01-15", "invalid-date", "2024-02-20"]})

        df_normalized = preprocessor.normalize_dates(df)

        assert "date" in df_normalized.columns
        # Invalid date should be converted to NaT
        assert pd.isna(df_normalized["at"].iloc[1])

    def test_normalize_dates_missing_column(self, preprocessor):
        """Test date normalization when 'at' column is missing."""
        df = pd.DataFrame({"data": [1, 2, 3]})

        df_result = preprocessor.normalize_dates(df)

        assert "date" not in df_result.columns


class TestLanguageFiltering:
    """Test language filtering functionality."""

    def test_filter_english_reviews_success(self, preprocessor):
        """Test filtering to keep only English reviews."""
        df = pd.DataFrame(
            {
                "content": [
                    "This is a great mobile banking application with excellent features",
                    "በጣም ጥሩ መተግበሪያ ነው እና በጣም ጥሩ አገልግሎት ይሰጣል",
                    "Excellent service and very user friendly interface",
                    "تطبيق رائع جدا وسهل الاستخدام",
                    "The app works very well and I highly recommend it",
                ]
            }
        )

        df_english = preprocessor.filter_non_english_reviews(df)

        # Should keep only English reviews (3 out of 5)
        assert len(df_english) < len(df)
        assert len(df_english) >= 2  # At least 2 English reviews

    def test_filter_english_reviews_all_english(self, preprocessor):
        """Test filtering when all reviews are English."""
        df = pd.DataFrame(
            {
                "content": [
                    "Great mobile banking app with good features",
                    "Excellent customer service and fast response",
                    "Very good application for everyday banking",
                    "Amazing service and easy to use interface",
                ]
            }
        )

        df_english = preprocessor.filter_non_english_reviews(df)

        # Should keep most/all reviews (language detection may vary for short text)
        assert len(df_english) >= len(df) * 0.5  # At least 50% detected as English

    def test_filter_english_reviews_missing_column(self, preprocessor):
        """Test filtering when content column is missing."""
        df = pd.DataFrame({"data": [1, 2, 3]})

        df_result = preprocessor.filter_non_english_reviews(df)

        assert len(df_result) == len(df)

    def test_filter_english_reviews_with_empty_text(self, preprocessor):
        """Test filtering with empty or very short text."""
        df = pd.DataFrame(
            {
                "content": [
                    "This is good",
                    "",
                    "OK",
                    None,
                    "Excellent application",
                ]
            }
        )

        df_english = preprocessor.filter_non_english_reviews(df)

        # Should handle empty/None values gracefully
        assert len(df_english) <= len(df)
        assert len(df_english) >= 1  # At least some valid English text


class TestColumnStandardization:
    """Test column standardization functionality."""

    def test_standardize_columns_rename(self, preprocessor):
        """Test column renaming during standardization."""
        df = pd.DataFrame(
            {
                "content": ["Text1", "Text2"],
                "score": [5, 4],
                "bank_name": ["CBE", "BOA"],
            }
        )

        df_std = preprocessor.standardize_columns(df)

        assert "review" in df_std.columns
        assert "rating" in df_std.columns
        assert "bank" in df_std.columns
        assert "content" not in df_std.columns
        assert "score" not in df_std.columns

    def test_standardize_columns_add_source(self, preprocessor):
        """Test adding source column during standardization."""
        df = pd.DataFrame({"data": [1, 2]})

        df_std = preprocessor.standardize_columns(df)

        assert "source" in df_std.columns
        assert (df_std["source"] == "Google Play").all()


class TestReviewIDGeneration:
    """Test review ID generation functionality."""

    def test_add_review_ids_with_existing_id(self, preprocessor):
        """Test adding review IDs when reviewId exists."""
        df = pd.DataFrame({"reviewId": ["rev1", "rev2", "rev3"]})

        df_with_ids = preprocessor.add_review_ids(df)

        assert "review_id" in df_with_ids.columns
        assert df_with_ids["review_id"].iloc[0] == "rev1"

    def test_add_review_ids_without_existing_id(self, preprocessor):
        """Test creating review IDs when reviewId doesn't exist."""
        df = pd.DataFrame({"data": [1, 2, 3]})

        df_with_ids = preprocessor.add_review_ids(df)

        assert "review_id" in df_with_ids.columns
        assert df_with_ids["review_id"].iloc[0] == "review_000000"
        assert df_with_ids["review_id"].iloc[2] == "review_000002"


class TestDataQualityValidation:
    """Test data quality validation functionality."""

    def test_validate_data_quality(self, preprocessor):
        """Test data quality validation report generation."""
        df = pd.DataFrame(
            {
                "review": ["Text1", "Text2", "Text3"],
                "rating": [5, 4, 3],
                "bank": ["CBE", "BOA", "CBE"],
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            }
        )

        report = preprocessor.validate_data_quality(df)

        assert "total_reviews" in report
        assert report["total_reviews"] == 3
        assert "banks" in report
        assert "rating_distribution" in report
        assert "missing_percentage" in report

    def test_validate_data_quality_with_missing(self, preprocessor):
        """Test quality validation with missing data."""
        df = pd.DataFrame(
            {
                "review": ["Text1", None, "Text3"],
                "rating": [5, 4, None],
                "bank": ["CBE", "BOA", "CBE"],
            }
        )

        report = preprocessor.validate_data_quality(df)

        assert report["missing_percentage"] > 0
        assert "missing_data" in report


class TestDataCleaning:
    """Test complete data cleaning pipeline."""

    def test_clean_data_full_pipeline(self, preprocessor, sample_raw_data):
        """Test full data cleaning pipeline."""
        cleaned_df = preprocessor.clean_data(sample_raw_data)

        assert isinstance(cleaned_df, pd.DataFrame)
        assert len(cleaned_df) <= len(sample_raw_data)
        assert "review_id" in cleaned_df.columns
        assert "date" in cleaned_df.columns

    def test_clean_data_no_data(self, preprocessor):
        """Test cleaning when no data is available."""
        with pytest.raises(PreprocessorError, match="No data available"):
            preprocessor.clean_data()

    def test_clean_data_sets_quality_report(self, preprocessor, sample_raw_data):
        """Test that clean_data generates quality report."""
        preprocessor.clean_data(sample_raw_data)

        assert preprocessor.quality_report != {}
        assert "total_reviews" in preprocessor.quality_report


class TestDataSaving:
    """Test data saving functionality."""

    def test_save_cleaned_data(self, preprocessor, tmp_path):
        """Test saving cleaned data to CSV."""
        df = pd.DataFrame({"review": ["Text1", "Text2"], "rating": [5, 4]})
        preprocessor.df = df

        filepath = preprocessor.save_cleaned_data(
            directory=tmp_path, filename="test_clean.csv"
        )

        assert filepath.exists()
        saved_df = pd.read_csv(filepath)
        assert len(saved_df) == 2

    def test_save_cleaned_data_no_data(self, preprocessor):
        """Test saving when no data is available."""
        with pytest.raises(PreprocessorError, match="No data available"):
            preprocessor.save_cleaned_data()

    def test_save_quality_report(self, preprocessor, tmp_path):
        """Test saving quality report to text file."""
        preprocessor.quality_report = {
            "total_reviews": 100,
            "banks": ["CBE", "BOA"],
            "missing_percentage": 5.0,
        }

        filepath = preprocessor.save_quality_report(
            directory=tmp_path, filename="test_report.txt"
        )

        assert filepath.exists()
        with open(filepath, "r") as f:
            content = f.read()
            assert "Total Reviews: 100" in content

    def test_save_quality_report_no_data(self, preprocessor, tmp_path):
        """Test saving quality report when no report exists."""
        result = preprocessor.save_quality_report(directory=tmp_path)

        assert result is None


class TestIntegration:
    """Integration tests for complete preprocessing workflow."""

    def test_complete_workflow(self, preprocessor, temp_csv_files, tmp_path):
        """Test complete preprocessing workflow from load to save."""
        # Load data
        df = preprocessor.load_raw_data(directory=temp_csv_files)
        assert len(df) > 0

        # Clean data
        cleaned_df = preprocessor.clean_data(df)
        assert len(cleaned_df) > 0

        # Save cleaned data
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        filepath = preprocessor.save_cleaned_data(directory=output_dir)
        assert filepath.exists()

        # Save quality report
        report_path = preprocessor.save_quality_report(directory=output_dir)
        assert report_path.exists()

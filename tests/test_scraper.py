"""
Unit tests for the ReviewScraper module.

This module tests the review scraping functionality including initialization,
scraping operations, data conversion, and file operations.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd

from core.scraper import ReviewScraper, ReviewScraperError
from core.config import Config


@pytest.fixture
def scraper():
    """Fixture to create a ReviewScraper instance for testing."""
    return ReviewScraper()


@pytest.fixture
def sample_review_data():
    """Fixture providing sample review data for testing."""
    return [
        {
            "reviewId": "3d88a334-958c-4717-9f97-c5d46359e054",
            "userName": "User One",
            "userImage": "https://play-lh.googleusercontent.com/a/test-image-1=mo",
            "content": "Great app, very useful!",
            "score": 5,
            "thumbsUpCount": 10,
            "reviewCreatedVersion": "5.2.1",
            "at": datetime(2024, 1, 1),
            "replyContent": None,
            "repliedAt": None,
            "appVersion": "5.2.1",
            "bank_code": "CBE",
            "bank_name": "Commercial Bank of Ethiopia",
            "app_id": "com.combanketh.mobilebanking",
            "scraped_at": "2024-01-01T00:00:00",
        },
        {
            "reviewId": "99d376ea-4824-4af9-a093-27360acc3a5c",
            "userName": "User Two",
            "userImage": "https://play-lh.googleusercontent.com/a/test-image-2=mo",
            "content": "Needs improvement",
            "score": 2,
            "thumbsUpCount": 3,
            "reviewCreatedVersion": "5.2.0",
            "at": datetime(2024, 1, 2),
            "replyContent": "Thank you for feedback",
            "repliedAt": datetime(2024, 1, 3),
            "appVersion": "5.2.0",
            "bank_code": "CBE",
            "bank_name": "Commercial Bank of Ethiopia",
            "app_id": "com.combanketh.mobilebanking",
            "scraped_at": "2024-01-02T00:00:00",
        },
    ]


class TestReviewScraperInitialization:
    """Test suite for ReviewScraper initialization."""

    def test_scraper_initialization(self, scraper):
        """Test that scraper initializes correctly."""
        assert scraper is not None
        assert isinstance(scraper.config, Config)
        assert scraper.logger is not None
        assert isinstance(scraper.scraped_data, dict)
        assert len(scraper.scraped_data) == 0

    def test_scraper_with_custom_config(self):
        """Test scraper initialization with custom config."""
        custom_config = Config()
        scraper = ReviewScraper(config=custom_config)
        assert scraper.config is custom_config

    def test_logger_setup(self, scraper):
        """Test that logger is properly configured."""
        assert scraper.logger is not None
        assert scraper.logger.name == "core.scraper"
        assert len(scraper.logger.handlers) >= 1


class TestReviewScraperValidation:
    """Test suite for input validation."""

    def test_invalid_bank_code_raises_error(self, scraper):
        """Test that invalid bank code raises ReviewScraperError."""
        with pytest.raises(ReviewScraperError) as exc_info:
            scraper.scrape_bank_reviews("INVALID_BANK", count=10)

        assert "Unknown bank code" in str(exc_info.value)

    def test_valid_bank_codes(self, scraper):
        """Test that valid bank codes are accepted."""
        valid_codes = ["CBE", "BOA", "Dashen"]
        for code in valid_codes:
            # Just test validation, not actual scraping
            try:
                app_id = scraper.config.get_app_id(code)
                assert app_id is not None
            except ValueError:
                pytest.fail(f"Valid bank code {code} raised ValueError")


class TestReviewScraperDataConversion:
    """Test suite for data conversion methods."""

    def test_convert_empty_data_to_dataframe(self, scraper):
        """Test converting empty data returns empty DataFrame."""
        df = scraper.convert_to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_convert_list_to_dataframe(self, scraper, sample_review_data):
        """Test converting list of reviews to DataFrame."""
        df = scraper.convert_to_dataframe(sample_review_data)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "reviewId" in df.columns
        assert "content" in df.columns
        assert "score" in df.columns
        assert "bank_code" in df.columns

    def test_convert_scraped_data_to_dataframe(self, scraper, sample_review_data):
        """Test converting internal scraped data to DataFrame."""
        scraper.scraped_data["CBE"] = sample_review_data

        df = scraper.convert_to_dataframe()
        assert len(df) == 2
        assert all(df["bank_code"] == "CBE")


class TestReviewScraperFileOperations:
    """Test suite for file save operations."""

    def test_save_to_csv_with_no_data_raises_error(self, scraper):
        """Test that saving with no data raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            scraper.save_to_csv()

        assert "No data to save" in str(exc_info.value)

    def test_save_to_csv_with_data(self, scraper, sample_review_data, tmp_path):
        """Test saving review data to CSV file."""
        filepath = scraper.save_to_csv(
            reviews_data=sample_review_data,
            filename="test_reviews.csv",
            directory=tmp_path,
        )

        assert filepath.exists()
        assert filepath.name == "test_reviews.csv"
        assert filepath.parent == tmp_path

        # Verify file contents
        df = pd.read_csv(filepath)
        assert len(df) == 2
        assert "content" in df.columns

    def test_save_to_csv_auto_filename(self, scraper, sample_review_data, tmp_path):
        """Test that auto-generated filename includes timestamp."""
        filepath = scraper.save_to_csv(
            reviews_data=sample_review_data, directory=tmp_path
        )

        assert filepath.exists()
        assert "raw_reviews_" in filepath.name
        assert filepath.name.endswith(".csv")

    def test_save_to_csv_adds_extension(self, scraper, sample_review_data, tmp_path):
        """Test that .csv extension is added if missing."""
        filepath = scraper.save_to_csv(
            reviews_data=sample_review_data, filename="test_file", directory=tmp_path
        )

        assert filepath.name == "test_file.csv"

    def test_save_bank_reviews_separately(self, scraper, sample_review_data, tmp_path):
        """Test saving each bank's reviews to separate files."""
        scraper.scraped_data["CBE"] = sample_review_data
        scraper.scraped_data["BOA"] = sample_review_data[:1]

        filepaths = scraper.save_bank_reviews_separately(directory=tmp_path)

        assert len(filepaths) == 2
        assert "CBE" in filepaths
        assert "BOA" in filepaths
        assert filepaths["CBE"].exists()
        assert filepaths["BOA"].exists()
        assert "cbe_reviews_raw.csv" in filepaths["CBE"].name
        assert "boa_reviews_raw.csv" in filepaths["BOA"].name

    def test_save_bank_reviews_separately_empty_data(self, scraper):
        """Test saving with no scraped data returns empty dict."""
        filepaths = scraper.save_bank_reviews_separately()
        assert filepaths == {}


class TestReviewScraperSummary:
    """Test suite for scraping summary methods."""

    def test_get_scraping_summary_empty(self, scraper):
        """Test summary with no scraped data."""
        summary = scraper.get_scraping_summary()

        assert summary["total_reviews"] == 0
        assert summary["banks_scraped"] == 0
        assert isinstance(summary["reviews_per_bank"], dict)
        assert "scraping_date" in summary

    def test_get_scraping_summary_with_data(self, scraper, sample_review_data):
        """Test summary with scraped data."""
        scraper.scraped_data["CBE"] = sample_review_data
        scraper.scraped_data["BOA"] = sample_review_data[:1]

        summary = scraper.get_scraping_summary()

        assert summary["total_reviews"] == 3
        assert summary["banks_scraped"] == 2
        assert summary["reviews_per_bank"]["CBE"] == 2
        assert summary["reviews_per_bank"]["BOA"] == 1
        assert isinstance(summary["scraping_date"], str)


class TestReviewScraperMocked:
    """Test suite using mocked API calls."""

    @patch("core.scraper.reviews")
    def test_scrape_bank_reviews_success(
        self, mock_reviews, scraper, sample_review_data
    ):
        """Test successful scraping with mocked API."""
        # Mock the API response
        mock_reviews.return_value = (sample_review_data[:2], None)

        result = scraper.scrape_bank_reviews("CBE", count=2)

        assert len(result) >= 2
        # Verify API was called with correct parameters
        mock_reviews.assert_called_once()

    @patch("core.scraper.reviews")
    def test_scrape_bank_reviews_adds_metadata(self, mock_reviews, scraper):
        """Test that scraping adds bank metadata to reviews."""
        # Create mock data without bank info
        mock_data = [
            {
                "reviewId": "test1",
                "content": "Test review",
                "score": 5,
                "at": datetime.now(),
            }
        ]
        mock_reviews.return_value = (mock_data, None)

        result = scraper.scrape_bank_reviews("CBE", count=1)

        # Verify metadata was added
        assert result[0]["bank_code"] == "CBE"
        assert result[0]["bank_name"] == "Commercial Bank of Ethiopia"
        assert "app_id" in result[0]
        assert "scraped_at" in result[0]

    @patch("core.scraper.reviews")
    def test_scrape_all_banks_success(self, mock_reviews, scraper):
        """Test scraping all banks with mocked API."""
        # Mock API to return empty list (quick test)
        mock_reviews.return_value = ([], None)

        all_reviews = scraper.scrape_all_banks(count_per_bank=1)

        assert isinstance(all_reviews, dict)
        assert len(all_reviews) == 3  # All three banks
        assert "CBE" in all_reviews
        assert "BOA" in all_reviews
        assert "Dashen" in all_reviews

    @patch("core.scraper.reviews")
    def test_scrape_bank_reviews_not_found_error(self, mock_reviews, scraper):
        """Test handling of NotFoundError from API."""
        from google_play_scraper.exceptions import NotFoundError

        mock_reviews.side_effect = NotFoundError("App not found")

        with pytest.raises(ReviewScraperError) as exc_info:
            scraper.scrape_bank_reviews("CBE", count=1)

        assert "App not found" in str(exc_info.value)

    @patch("core.scraper.reviews")
    def test_scrape_bank_reviews_general_exception(self, mock_reviews, scraper):
        """Test handling of general exceptions during scraping."""
        mock_reviews.side_effect = Exception("Network error")

        with pytest.raises(ReviewScraperError) as exc_info:
            scraper.scrape_bank_reviews("CBE", count=1)

        assert "Error scraping reviews" in str(exc_info.value)


class TestReviewScraperErrorHandling:
    """Test suite for error handling."""

    def test_custom_exception_inheritance(self):
        """Test that ReviewScraperError is an Exception."""
        error = ReviewScraperError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

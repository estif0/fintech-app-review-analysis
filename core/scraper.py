"""
Google Play Store Review Scraper Module.

This module provides functionality for scraping user reviews from the Google Play Store
for Ethiopian banking applications. It implements robust error handling, rate limiting,
and progress tracking.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd
from google_play_scraper import Sort, reviews
from google_play_scraper.exceptions import NotFoundError
from tqdm import tqdm

# Handle imports for both package usage and direct script execution
try:
    from core.config import Config
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    # Add parent directory to path for direct script execution
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.config import Config


class ReviewScraperError(Exception):
    """Custom exception for ReviewScraper errors."""

    pass


class ReviewScraper:
    """
    A class for scraping app reviews from Google Play Store.

    This class handles the scraping of reviews for mobile banking apps,
    including rate limiting, error handling, data validation, and progress tracking.

    Attributes:
        config (Config): Configuration object with scraping parameters
        logger (logging.Logger): Logger instance for tracking operations
        scraped_data (Dict[str, List[Dict]]): Storage for scraped reviews per bank

    Example:
        >>> scraper = ReviewScraper()
        >>> reviews_data = scraper.scrape_bank_reviews('CBE', count=400)
        >>> scraper.save_to_csv(reviews_data, 'cbe_reviews.csv')
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        """
        Initialize the ReviewScraper with configuration parameters.

        Args:
            config (Optional[Config]): Configuration object. If None, uses default Config.

        Example:
            >>> scraper = ReviewScraper()
            >>> print(scraper.config.BANK_APP_IDS)
        """
        self.config = config or Config()
        self.logger = self._setup_logger()
        self.scraped_data: Dict[str, List[Dict[str, Any]]] = {}

        # Ensure data directories exist
        self.config.ensure_directories()

        self.logger.info("ReviewScraper initialized successfully")

    def _setup_logger(self) -> logging.Logger:
        """
        Set up logging configuration for the scraper.

        Returns:
            logging.Logger: Configured logger instance

        Note:
            Logs are written to both console and file specified in config.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.LOG_LEVEL))

        # Avoid adding handlers multiple times
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter("%(levelname)s - %(message)s")
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            # File handler
            file_handler = logging.FileHandler(self.config.LOG_FILE)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(self.config.LOG_FORMAT)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        return logger

    def scrape_bank_reviews(
        self,
        bank_code: str,
        count: Optional[int] = None,
        sort_by: Sort = Sort.NEWEST,
        sleep_time: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Scrape reviews for a specific bank's mobile app.

        This method retrieves reviews from Google Play Store with rate
        limiting and error handling to ensure reliable data collection.

        Args:
            bank_code (str): Bank code ('CBE', 'BOA', or 'Dashen')
            count (Optional[int]): Number of reviews to scrape.
                                   Defaults to config value if None.
            sort_by (Sort): Sorting order for reviews. Defaults to Sort.NEWEST.
            sleep_time (float): Seconds to sleep between API calls. Defaults to 1.0.

        Returns:
            List[Dict[str, Any]]: List of review dictionaries containing:
                - reviewId (str): Unique review identifier (UUID format)
                - userName (str): Name of the reviewer
                - userImage (str): URL to reviewer's profile image
                - content (str): The review text
                - score (int): Rating from 1-5
                - thumbsUpCount (int): Number of helpful votes
                - reviewCreatedVersion (str): App version when review was created
                - at (datetime): Review date and time
                - replyContent (str): Developer reply text (if any)
                - repliedAt (datetime): Reply date and time (if any)
                - appVersion (str): Current app version
                - bank_code (str): Bank code added by scraper
                - bank_name (str): Full bank name added by scraper
                - app_id (str): Google Play app ID added by scraper
                - scraped_at (str): ISO format timestamp when scraped

        Raises:
            ReviewScraperError: If scraping fails due to invalid bank code or network issues
            ValueError: If bank_code is not recognized

        Example:
            >>> scraper = ReviewScraper()
            >>> reviews = scraper.scrape_bank_reviews('CBE', count=500)
            >>> print(f"Scraped {len(reviews)} reviews")
        """
        # Validate bank code
        try:
            app_id = self.config.get_app_id(bank_code)
            bank_name = self.config.get_bank_name(bank_code)
        except ValueError as e:
            self.logger.error(f"Invalid bank code: {bank_code}")
            raise ReviewScraperError(str(e)) from e

        # Use config count if not specified
        if count is None:
            count = self.config.SCRAPING_PARAMS["count"]

        self.logger.info(
            f"Starting to scrape {count} reviews for {bank_name} ({bank_code})"
        )

        try:
            # Scrape reviews with progress tracking
            result, continuation_token = reviews(
                app_id,
                lang=self.config.SCRAPING_PARAMS["language"],
                country=self.config.SCRAPING_PARAMS["country"],
                sort=sort_by,
                count=count,  # type: ignore
            )

            # Add bank information to each review
            for review in result:
                review["bank_code"] = bank_code
                review["bank_name"] = bank_name
                review["app_id"] = app_id
                review["scraped_at"] = datetime.now().isoformat()

            self.logger.info(
                f"Successfully scraped {len(result)} reviews for {bank_name}"
            )

            # Rate limiting
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Store in instance variable
            self.scraped_data[bank_code] = result

            return result

        except NotFoundError as e:
            error_msg = f"App not found for {bank_name}: {app_id}"
            self.logger.error(error_msg)
            raise ReviewScraperError(error_msg) from e

        except Exception as e:
            error_msg = f"Error scraping reviews for {bank_name}: {str(e)}"
            self.logger.error(error_msg)
            raise ReviewScraperError(error_msg) from e

    def scrape_all_banks(
        self, count_per_bank: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scrape reviews for all configured banks.

        This method iterates through all banks defined in the configuration
        and scrapes reviews for each, with progress tracking.

        Args:
            count_per_bank (Optional[int]): Number of reviews per bank.
                                            Defaults to config value if None.

        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary mapping bank codes to
                                             their review lists

        Raises:
            ReviewScraperError: If scraping fails for any bank

        Example:
            >>> scraper = ReviewScraper()
            >>> all_reviews = scraper.scrape_all_banks(count_per_bank=400)
            >>> print(f"Total banks scraped: {len(all_reviews)}")
        """
        banks = self.config.get_all_banks()
        self.logger.info(f"Starting to scrape reviews for {len(banks)} banks")

        all_reviews: Dict[str, List[Dict[str, Any]]] = {}

        # Use tqdm for progress tracking
        for bank_code in tqdm(banks, desc="Scraping banks"):
            try:
                reviews_list = self.scrape_bank_reviews(
                    bank_code=bank_code, count=count_per_bank
                )
                all_reviews[bank_code] = reviews_list

                self.logger.info(f"✓ {bank_code}: {len(reviews_list)} reviews scraped")

            except ReviewScraperError as e:
                self.logger.error(f"✗ Failed to scrape {bank_code}: {str(e)}")
                # Continue with other banks even if one fails
                all_reviews[bank_code] = []

        # Summary
        total_reviews = sum(len(reviews) for reviews in all_reviews.values())
        self.logger.info(
            f"Scraping complete: {total_reviews} total reviews from "
            f"{len(all_reviews)} banks"
        )

        self.scraped_data = all_reviews
        return all_reviews

    def convert_to_dataframe(
        self, reviews_data: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """
        Convert scraped reviews to a pandas DataFrame.

        Args:
            reviews_data (Optional[List[Dict]]): Review data to convert.
                                                 If None, uses all scraped data.

        Returns:
            pd.DataFrame: DataFrame with structured review data

        Example:
            >>> scraper = ReviewScraper()
            >>> scraper.scrape_all_banks()
            >>> df = scraper.convert_to_dataframe()
            >>> print(df.shape)
        """
        if reviews_data is None:
            # Combine all scraped data
            reviews_data = []
            for bank_reviews in self.scraped_data.values():
                reviews_data.extend(bank_reviews)

        if not reviews_data:
            self.logger.warning("No review data to convert")
            return pd.DataFrame()

        df = pd.DataFrame(reviews_data)

        self.logger.info(f"Converted {len(df)} reviews to DataFrame")
        return df

    def save_to_csv(
        self,
        reviews_data: Optional[List[Dict[str, Any]]] = None,
        filename: Optional[str] = None,
        directory: Optional[Path] = None,
    ) -> Path:
        """
        Save scraped reviews to a CSV file.

        Args:
            reviews_data (Optional[List[Dict]]): Review data to save.
                                                 If None, uses all scraped data.
            filename (Optional[str]): Output filename. If None, generates timestamp-based name.
            directory (Optional[Path]): Output directory. Defaults to config.RAW_DATA_DIR.

        Returns:
            Path: Path to the saved CSV file

        Raises:
            ValueError: If no data is available to save

        Example:
            >>> scraper = ReviewScraper()
            >>> scraper.scrape_all_banks()
            >>> filepath = scraper.save_to_csv()
            >>> print(f"Saved to: {filepath}")
        """
        df = self.convert_to_dataframe(reviews_data)

        if df.empty:
            raise ValueError("No data to save")

        # Set default directory
        if directory is None:
            directory = self.config.RAW_DATA_DIR

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"raw_reviews_{timestamp}.csv"

        # Ensure .csv extension
        if not filename.endswith(".csv"):
            filename += ".csv"

        filepath = directory / filename

        # Save to CSV
        df.to_csv(filepath, index=False)

        self.logger.info(f"Saved {len(df)} reviews to {filepath}")
        return filepath

    def save_bank_reviews_separately(
        self, directory: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Save each bank's reviews to separate CSV files.

        Args:
            directory (Optional[Path]): Output directory. Defaults to config.RAW_DATA_DIR.

        Returns:
            Dict[str, Path]: Dictionary mapping bank codes to their file paths

        Example:
            >>> scraper = ReviewScraper()
            >>> scraper.scrape_all_banks()
            >>> filepaths = scraper.save_bank_reviews_separately()
            >>> print(filepaths)
        """
        if not self.scraped_data:
            self.logger.warning("No scraped data to save")
            return {}

        if directory is None:
            directory = self.config.RAW_DATA_DIR

        saved_files: Dict[str, Path] = {}

        for bank_code, reviews_list in self.scraped_data.items():
            if reviews_list:
                filename = f"{bank_code.lower()}_reviews_raw.csv"
                filepath = self.save_to_csv(
                    reviews_data=reviews_list, filename=filename, directory=directory
                )
                saved_files[bank_code] = filepath

        self.logger.info(f"Saved reviews for {len(saved_files)} banks separately")
        return saved_files

    def get_scraping_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the scraping operation.

        Returns:
            Dict[str, Any]: Summary statistics including counts per bank

        Example:
            >>> scraper = ReviewScraper()
            >>> scraper.scrape_all_banks()
            >>> summary = scraper.get_scraping_summary()
            >>> print(summary)
        """
        summary = {
            "total_reviews": 0,
            "banks_scraped": 0,
            "reviews_per_bank": {},
            "scraping_date": datetime.now().isoformat(),
        }

        for bank_code, reviews_list in self.scraped_data.items():
            count = len(reviews_list)
            summary["reviews_per_bank"][bank_code] = count
            summary["total_reviews"] += count
            if count > 0:
                summary["banks_scraped"] += 1

        return summary


if __name__ == "__main__":
    # Example usage and testing
    print("Testing ReviewScraper...\n")

    # Initialize scraper
    scraper = ReviewScraper()

    # Test scraping for one bank (small count for testing)
    print("Testing single bank scrape (CBE)...")
    try:
        cbe_reviews = scraper.scrape_bank_reviews("CBE", count=10)
        print(f"✓ Scraped {len(cbe_reviews)} reviews for CBE")

        # Save to CSV
        filepath = scraper.save_to_csv(cbe_reviews, "test_cbe_reviews.csv")
        print(f"✓ Saved to {filepath}")

    except ReviewScraperError as e:
        print(f"✗ Error: {e}")

    print("\n" + "=" * 50 + "\n")
    print("To scrape all banks, run:")
    print(">>> scraper = ReviewScraper()")
    print(">>> scraper.scrape_all_banks(count_per_bank=400)")
    print(">>> scraper.save_bank_reviews_separately()")

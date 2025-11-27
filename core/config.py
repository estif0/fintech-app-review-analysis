"""
Configuration module for the Fintech App Review Analysis project.

This module contains configuration settings for scraping Google Play Store reviews,
including bank app identifiers, scraping parameters, and file paths.
"""

import os
from typing import Dict, Any
from pathlib import Path


class Config:
    """
    Configuration class for managing project settings.

    This class centralizes all configuration parameters including bank app IDs,
    scraping settings, file paths, and logging configurations.

    Attributes:
        BANK_APP_IDS (Dict[str, str]): Mapping of bank names to Google Play app IDs
        SCRAPING_PARAMS (Dict[str, Any]): Parameters for scraping configuration
        BASE_DIR (Path): Base directory of the project
        DATA_DIR (Path): Directory for data storage
        RAW_DATA_DIR (Path): Directory for raw scraped data
        PROCESSED_DATA_DIR (Path): Directory for processed data
        CLEANED_DATA_DIR (Path): Directory for cleaned data

    Example:
        >>> config = Config()
        >>> print(config.BANK_APP_IDS['CBE'])
        'com.combanketh.mobilebanking'
    """

    # Bank Application IDs on Google Play Store
    BANK_APP_IDS: Dict[str, str] = {
        "CBE": "com.combanketh.mobilebanking",  # Commercial Bank of Ethiopia
        "BOA": "com.boa.boamobilebanking",  # Bank of Abyssinia
        "Dashen": "com.dashenbank.amanu",  # Dashen Bank
    }

    # Full bank names for display purposes
    BANK_NAMES: Dict[str, str] = {
        "CBE": "Commercial Bank of Ethiopia",
        "BOA": "Bank of Abyssinia",
        "Dashen": "Dashen Bank",
    }

    # Scraping Parameters
    SCRAPING_PARAMS: Dict[str, Any] = {
        "language": "en",  # Language code for reviews
        "country": "et",  # Country code (Ethiopia)
        "count": 500,  # Number of reviews to scrape per bank
        "sort_by": 1,  # Sort by: 1=newest, 2=rating, 3=helpfulness
        "filter_score_with": None,  # Filter by rating (None = all ratings)
    }

    # Project Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    CLEANED_DATA_DIR: Path = DATA_DIR / "cleaned"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    FIGURES_DIR: Path = REPORTS_DIR / "figures"

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "scraper.log")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Data Quality Thresholds
    MIN_REVIEWS_PER_BANK: int = 400
    MAX_MISSING_DATA_PERCENT: float = 5.0

    # Database Configuration (loaded from environment variables)
    DB_CONFIG: Dict[str, str] = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "database": os.getenv("DB_NAME", "bank_reviews"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
    }

    @classmethod
    def ensure_directories(cls) -> None:
        """
        Create all necessary directories if they don't exist.

        This method ensures that all required directories for data storage
        and reports are created before any operations begin.

        Example:
            >>> Config.ensure_directories()
            >>> assert Config.RAW_DATA_DIR.exists()
        """
        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.CLEANED_DATA_DIR,
            cls.REPORTS_DIR,
            cls.FIGURES_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_app_id(cls, bank_code: str) -> str:
        """
        Get the Google Play app ID for a specific bank.

        Args:
            bank_code (str): Bank code ('CBE', 'BOA', or 'Dashen')

        Returns:
            str: Google Play app ID for the specified bank

        Raises:
            ValueError: If bank_code is not recognized

        Example:
            >>> Config.get_app_id('CBE')
            'com.combanketh.mobilebanking'
        """
        if bank_code not in cls.BANK_APP_IDS:
            raise ValueError(
                f"Unknown bank code: {bank_code}. "
                f"Valid codes are: {', '.join(cls.BANK_APP_IDS.keys())}"
            )
        return cls.BANK_APP_IDS[bank_code]

    @classmethod
    def get_bank_name(cls, bank_code: str) -> str:
        """
        Get the full name of the bank from its code.

        Args:
            bank_code (str): Bank code ('CBE', 'BOA', or 'Dashen')

        Returns:
            str: Full bank name

        Raises:
            ValueError: If bank_code is not recognized

        Example:
            >>> Config.get_bank_name('CBE')
            'Commercial Bank of Ethiopia'
        """
        if bank_code not in cls.BANK_NAMES:
            raise ValueError(
                f"Unknown bank code: {bank_code}. "
                f"Valid codes are: {', '.join(cls.BANK_NAMES.keys())}"
            )
        return cls.BANK_NAMES[bank_code]

    @classmethod
    def get_all_banks(cls) -> list:
        """
        Get a list of all bank codes.

        Returns:
            list: List of all bank codes

        Example:
            >>> Config.get_all_banks()
            ['CBE', 'BOA', 'Dashen']
        """
        return list(cls.BANK_APP_IDS.keys())


# Create a singleton instance for easy import
config = Config()


if __name__ == "__main__":
    # Test configuration
    print("Bank App IDs:")
    for bank, app_id in Config.BANK_APP_IDS.items():
        print(f"  {bank}: {app_id}")

    print("\nBank Names:")
    for code in Config.get_all_banks():
        print(f"  {code}: {Config.get_bank_name(code)}")

    print("\nScraping Parameters:")
    for key, value in Config.SCRAPING_PARAMS.items():
        print(f"  {key}: {value}")

    print("\nProject Paths:")
    print(f"  Base Directory: {Config.BASE_DIR}")
    print(f"  Raw Data: {Config.RAW_DATA_DIR}")
    print(f"  Processed Data: {Config.PROCESSED_DATA_DIR}")

    # Ensure directories exist
    Config.ensure_directories()
    print("\nAll directories created successfully!")

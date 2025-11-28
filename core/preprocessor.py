"""
Data Preprocessing Module for Bank Review Analysis.

This module provides functionality for cleaning, normalizing, and validating
scraped review data. It handles missing values, duplicates, date formatting,
and data quality checks.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
from langdetect import detect, LangDetectException

# Handle imports for both package usage and direct script execution
try:
    from core.config import Config
except ModuleNotFoundError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.config import Config


class PreprocessorError(Exception):
    """Custom exception for Preprocessor errors."""

    pass


class ReviewPreprocessor:
    """
    A class for preprocessing bank review data.

    This class handles data cleaning, normalization, validation, and quality checks
    for scraped review data before analysis.

    Attributes:
        config (Config): Configuration object with paths and settings
        logger (logging.Logger): Logger instance for tracking operations
        df (Optional[pd.DataFrame]): Current DataFrame being processed
        quality_report (Dict[str, Any]): Data quality metrics and statistics

    Example:
        >>> preprocessor = ReviewPreprocessor()
        >>> df = preprocessor.load_raw_data()
        >>> cleaned_df = preprocessor.clean_data(df)
        >>> preprocessor.save_cleaned_data(cleaned_df)
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        """
        Initialize the ReviewPreprocessor with configuration.

        Args:
            config (Optional[Config]): Configuration object. If None, uses default Config.

        Example:
            >>> preprocessor = ReviewPreprocessor()
            >>> print(preprocessor.config.BANK_APP_IDS)
        """
        self.config = config or Config()
        self.logger = self._setup_logger()
        self.df: Optional[pd.DataFrame] = None
        self.quality_report: Dict[str, Any] = {}

        self.logger.info("ReviewPreprocessor initialized successfully")

    def _setup_logger(self) -> logging.Logger:
        """
        Set up logging configuration for the preprocessor.

        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.LOG_LEVEL))

        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter("%(levelname)s - %(message)s")
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

            # File handler
            log_file = self.config.BASE_DIR / "preprocessing.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(self.config.LOG_FORMAT)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        return logger

    def load_raw_data(
        self, directory: Optional[Path] = None, pattern: str = "*_reviews_raw.csv"
    ) -> pd.DataFrame:
        """
        Load raw scraped data from CSV files.

        Args:
            directory (Optional[Path]): Directory containing CSV files.
                                       Defaults to config.RAW_DATA_DIR
            pattern (str): Glob pattern to match CSV files. Defaults to "*_reviews_raw.csv"

        Returns:
            pd.DataFrame: Combined DataFrame with all reviews

        Raises:
            PreprocessorError: If no files are found or loading fails

        Example:
            >>> preprocessor = ReviewPreprocessor()
            >>> df = preprocessor.load_raw_data()
            >>> print(f"Loaded {len(df)} reviews")
        """
        directory = directory or self.config.RAW_DATA_DIR

        # Find all CSV files matching pattern
        csv_files = list(directory.glob(pattern))

        if not csv_files:
            raise PreprocessorError(
                f"No CSV files found in {directory} matching pattern '{pattern}'"
            )

        self.logger.info(f"Found {len(csv_files)} CSV files to load")

        # Load and combine all files
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dataframes.append(df)
                self.logger.info(f"Loaded {len(df)} reviews from {csv_file.name}")
            except Exception as e:
                self.logger.error(f"Failed to load {csv_file}: {e}")
                raise PreprocessorError(f"Error loading {csv_file}: {e}") from e

        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)

        self.logger.info(
            f"Combined total: {len(combined_df)} reviews from {len(csv_files)} files"
        )
        self.df = combined_df

        return combined_df

    def remove_duplicates(
        self, df: pd.DataFrame, subset: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Remove duplicate reviews from the dataset.

        Args:
            df (pd.DataFrame): Input DataFrame
            subset (Optional[List[str]]): Columns to consider for duplicates.
                                         Defaults to ['reviewId', 'content', 'at']

        Returns:
            pd.DataFrame: DataFrame with duplicates removed

        Example:
            >>> df_clean = preprocessor.remove_duplicates(df)
            >>> print(f"Removed {len(df) - len(df_clean)} duplicates")
        """
        if subset is None:
            subset = ["reviewId", "content", "at"]

        # Check which subset columns exist
        available_cols = [col for col in subset if col in df.columns]

        if not available_cols:
            self.logger.warning(
                "No duplicate check columns found, skipping duplicate removal"
            )
            return df

        initial_count = len(df)
        df_clean = df.drop_duplicates(subset=available_cols, keep="first")
        removed_count = initial_count - len(df_clean)

        if removed_count > 0:
            self.logger.info(
                f"Removed {removed_count} duplicate reviews ({removed_count/initial_count*100:.2f}%)"
            )
        else:
            self.logger.info("No duplicates found")

        return df_clean

    def handle_missing_values(
        self, df: pd.DataFrame, critical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Critical columns (content, score) will cause row removal if missing.
        Non-critical columns will be filled or left as-is.

        Args:
            df (pd.DataFrame): Input DataFrame
            critical_columns (Optional[List[str]]): Columns that must have values.
                                                    Defaults to ['content', 'score']

        Returns:
            pd.DataFrame: DataFrame with missing values handled

        Example:
            >>> df_clean = preprocessor.handle_missing_values(df)
        """
        if critical_columns is None:
            critical_columns = ["content", "score"]

        initial_count = len(df)

        # Remove rows with missing critical columns
        for col in critical_columns:
            if col in df.columns:
                before = len(df)
                df = df.dropna(subset=[col])
                removed = before - len(df)
                if removed > 0:
                    self.logger.info(f"Removed {removed} rows with missing '{col}'")

        # Handle non-critical columns
        if "replyContent" in df.columns:
            df["replyContent"] = df["replyContent"].fillna("")

        if "thumbsUpCount" in df.columns:
            df["thumbsUpCount"] = df["thumbsUpCount"].fillna(0)

        if "reviewCreatedVersion" in df.columns:
            df["reviewCreatedVersion"] = df["reviewCreatedVersion"].fillna("Unknown")

        if "appVersion" in df.columns:
            df["appVersion"] = df["appVersion"].fillna("Unknown")

        total_removed = initial_count - len(df)
        if total_removed > 0:
            self.logger.info(
                f"Total rows removed due to missing values: {total_removed} ({total_removed/initial_count*100:.2f}%)"
            )

        return df

    def normalize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize date columns to consistent format.

        Converts 'at' column to datetime and adds a normalized 'date' column in YYYY-MM-DD format.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with normalized dates

        Example:
            >>> df_normalized = preprocessor.normalize_dates(df)
            >>> print(df_normalized['date'].dtype)
        """
        if "at" not in df.columns:
            self.logger.warning("'at' column not found, skipping date normalization")
            return df

        try:
            # Convert 'at' to datetime
            df["at"] = pd.to_datetime(df["at"], errors="coerce")

            # Create normalized date column (YYYY-MM-DD)
            df["date"] = df["at"].dt.strftime("%Y-%m-%d")

            # Check for invalid dates
            invalid_dates = df["at"].isna().sum()
            if invalid_dates > 0:
                self.logger.warning(f"Found {invalid_dates} invalid dates")

            self.logger.info("Date normalization complete")

        except Exception as e:
            self.logger.error(f"Error normalizing dates: {e}")
            raise PreprocessorError(f"Date normalization failed: {e}") from e

        return df

    def filter_non_english_reviews(
        self, df: pd.DataFrame, review_column: str = "content"
    ) -> pd.DataFrame:
        """
        Filter out non-English reviews from the dataset.

        Uses language detection to identify and remove reviews not written in English.
        This is essential for sentiment analysis models trained on English text.

        Args:
            df (pd.DataFrame): Input DataFrame
            review_column (str): Column containing review text. Defaults to "content"

        Returns:
            pd.DataFrame: DataFrame with only English reviews

        Example:
            >>> df_english = preprocessor.filter_non_english_reviews(df)
            >>> print(f"Kept {len(df_english)} English reviews")
        """
        if review_column not in df.columns:
            self.logger.warning(
                f"'{review_column}' column not found, skipping language filtering"
            )
            return df

        initial_count = len(df)

        def detect_language(text: str) -> str:
            """
            Detect language of text, returning 'unknown' on failure.

            Args:
                text (str): Text to analyze

            Returns:
                str: Detected language code (e.g., 'en', 'am', 'ar') or 'unknown'
            """
            if pd.isna(text) or not isinstance(text, str) or len(text.strip()) < 3:
                return "unknown"
            try:
                return detect(str(text))
            except LangDetectException:
                return "unknown"

        # Detect language for each review
        self.logger.info("Detecting language for all reviews...")
        df["detected_language"] = df[review_column].apply(detect_language)

        # Count reviews by language
        language_counts = df["detected_language"].value_counts()
        self.logger.info(f"Language distribution: {language_counts.to_dict()}")

        # Filter to keep only English reviews
        df_english = df[df["detected_language"] == "en"].copy()

        # Drop the temporary language column
        df_english = df_english.drop(columns=["detected_language"])

        removed_count = initial_count - len(df_english)
        if removed_count > 0:
            self.logger.info(
                f"Removed {removed_count} non-English reviews ({removed_count/initial_count*100:.2f}%)"
            )
        else:
            self.logger.info("All reviews are in English")

        return df_english

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and add required columns.

        Renames columns to standard names: review, rating, date, bank, source.
        Adds 'source' column if not present.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with standardized columns

        Example:
            >>> df_std = preprocessor.standardize_columns(df)
        """
        # Column mapping
        rename_map = {"content": "review", "score": "rating", "bank_name": "bank"}

        # Rename columns that exist
        for old_name, new_name in rename_map.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})

        # Add source column if not present
        if "source" not in df.columns:
            df["source"] = "Google Play"

        # Ensure bank code column exists
        if "bank" not in df.columns and "bank_code" in df.columns:
            df["bank"] = df["bank_code"]

        self.logger.info("Column standardization complete")

        return df

    def add_review_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure each review has a unique identifier.

        If 'reviewId' column exists, uses it. Otherwise, creates unique IDs.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with review_id column

        Example:
            >>> df_with_ids = preprocessor.add_review_ids(df)
        """
        if "reviewId" in df.columns:
            # Use existing reviewId
            df["review_id"] = df["reviewId"]
            self.logger.info("Using existing reviewId as review_id")
        else:
            # Create unique IDs
            df["review_id"] = [f"review_{i:06d}" for i in range(len(df))]
            self.logger.info("Created new review_id column")

        return df

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and generate quality report.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            Dict[str, Any]: Data quality report with metrics

        Example:
            >>> quality = preprocessor.validate_data_quality(df)
            >>> print(f"Missing data: {quality['missing_percentage']:.2f}%")
        """
        report = {
            "total_reviews": len(df),
            "banks": df["bank"].unique().tolist() if "bank" in df.columns else [],
            "date_range": {},
            "rating_distribution": {},
            "missing_data": {},
            "missing_percentage": 0.0,
        }

        # Reviews per bank
        if "bank" in df.columns:
            report["reviews_per_bank"] = df["bank"].value_counts().to_dict()

        # Date range
        if "date" in df.columns:
            report["date_range"] = {"start": df["date"].min(), "end": df["date"].max()}

        # Rating distribution
        if "rating" in df.columns:
            report["rating_distribution"] = (
                df["rating"].value_counts().sort_index().to_dict()
            )
            report["average_rating"] = float(df["rating"].mean())

        # Missing data analysis
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df) * 100).round(2)
        report["missing_data"] = missing_percentages[missing_percentages > 0].to_dict()
        report["missing_percentage"] = float(missing_percentages.mean())

        # Quality check against thresholds
        report["quality_checks"] = {
            "meets_min_reviews": len(df)
            >= self.config.MIN_REVIEWS_PER_BANK * len(report["banks"]),
            "meets_missing_threshold": report["missing_percentage"]
            < self.config.MAX_MISSING_DATA_PERCENT,
        }

        self.quality_report = report
        self.logger.info(
            f"Data quality validation complete: {len(df)} reviews, {report['missing_percentage']:.2f}% missing"
        )

        return report

    def clean_data(
        self,
        df: Optional[pd.DataFrame] = None,
        remove_duplicates: bool = True,
        handle_missing: bool = True,
        normalize_dates: bool = True,
        filter_language: bool = True,
        standardize: bool = True,
    ) -> pd.DataFrame:
        """
        Apply all cleaning steps to the data.

        This is the main preprocessing pipeline that applies all cleaning operations.

        Args:
            df (Optional[pd.DataFrame]): Input DataFrame. If None, uses self.df
            remove_duplicates (bool): Whether to remove duplicates. Defaults to True.
            handle_missing (bool): Whether to handle missing values. Defaults to True.
            normalize_dates (bool): Whether to normalize dates. Defaults to True.
            filter_language (bool): Whether to filter non-English reviews. Defaults to True.
            standardize (bool): Whether to standardize columns. Defaults to True.

        Returns:
            pd.DataFrame: Cleaned DataFrame

        Raises:
            PreprocessorError: If no data is available

        Example:
            >>> preprocessor = ReviewPreprocessor()
            >>> df = preprocessor.load_raw_data()
            >>> cleaned_df = preprocessor.clean_data(df)
        """
        if df is None:
            if self.df is None:
                raise PreprocessorError(
                    "No data available. Load data first using load_raw_data()"
                )
            df = self.df.copy()
        else:
            df = df.copy()

        initial_count = len(df)
        self.logger.info(
            f"Starting data cleaning pipeline with {initial_count} reviews"
        )

        # Apply cleaning steps
        if remove_duplicates:
            df = self.remove_duplicates(df)

        if handle_missing:
            df = self.handle_missing_values(df)

        if normalize_dates:
            df = self.normalize_dates(df)

        if filter_language:
            df = self.filter_non_english_reviews(df)

        if standardize:
            df = self.standardize_columns(df)

        # Add review IDs
        df = self.add_review_ids(df)

        # Validate quality
        quality_report = self.validate_data_quality(df)

        final_count = len(df)
        removed_count = initial_count - final_count

        self.logger.info(
            f"Cleaning complete: {final_count} reviews ({removed_count} removed, {removed_count/initial_count*100:.2f}%)"
        )

        self.df = df
        return df

    def save_cleaned_data(
        self,
        df: Optional[pd.DataFrame] = None,
        filename: str = "cleaned_reviews.csv",
        directory: Optional[Path] = None,
    ) -> Path:
        """
        Save cleaned data to CSV file with standardized columns.

        Only saves the essential columns: review, rating, date, bank, source

        Args:
            df (Optional[pd.DataFrame]): DataFrame to save. If None, uses self.df
            filename (str): Output filename. Defaults to "cleaned_reviews.csv"
            directory (Optional[Path]): Output directory. Defaults to config.PROCESSED_DATA_DIR

        Returns:
            Path: Path to saved file

        Raises:
            PreprocessorError: If no data is available

        Example:
            >>> filepath = preprocessor.save_cleaned_data()
            >>> print(f"Saved to: {filepath}")
        """
        if df is None:
            if self.df is None:
                raise PreprocessorError("No data available to save")
            df = self.df

        directory = directory or self.config.PROCESSED_DATA_DIR
        directory.mkdir(parents=True, exist_ok=True)

        # Select only the required columns in the correct order
        required_columns = ["review", "rating", "date", "bank", "source"]

        # Check which columns exist
        available_columns = [col for col in required_columns if col in df.columns]

        if len(available_columns) < len(required_columns):
            missing = set(required_columns) - set(available_columns)
            self.logger.warning(
                f"Missing columns: {missing}. Saving available columns only."
            )

        # Save only the required columns
        df_to_save = df[available_columns].copy()

        filepath = directory / filename
        df_to_save.to_csv(filepath, index=False)

        self.logger.info(
            f"Saved {len(df_to_save)} cleaned reviews with {len(available_columns)} columns to {filepath}"
        )

        return filepath

    def save_quality_report(
        self,
        report: Optional[Dict[str, Any]] = None,
        filename: str = "data_quality_report.txt",
        directory: Optional[Path] = None,
    ) -> Path:
        """
        Save data quality report to text file.

        Args:
            report (Optional[Dict[str, Any]]): Quality report. If None, uses self.quality_report
            filename (str): Output filename. Defaults to "data_quality_report.txt"
            directory (Optional[Path]): Output directory. Defaults to config.PROCESSED_DATA_DIR

        Returns:
            Path: Path to saved report

        Example:
            >>> report_path = preprocessor.save_quality_report()
        """
        if report is None:
            report = self.quality_report

        if not report:
            self.logger.warning("No quality report available")
            return None  # type: ignore

        directory = directory or self.config.PROCESSED_DATA_DIR
        directory.mkdir(parents=True, exist_ok=True)

        filepath = directory / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("DATA QUALITY REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Total Reviews: {report['total_reviews']}\n")
            f.write(f"Banks: {', '.join(report['banks'])}\n\n")

            if "reviews_per_bank" in report:
                f.write("Reviews per Bank:\n")
                for bank, count in report["reviews_per_bank"].items():
                    f.write(f"  {bank}: {count}\n")
                f.write("\n")

            if report.get("date_range"):
                f.write(
                    f"Date Range: {report['date_range']['start']} to {report['date_range']['end']}\n\n"
                )

            if "rating_distribution" in report:
                f.write("Rating Distribution:\n")
                for rating, count in sorted(report["rating_distribution"].items()):
                    f.write(f"  {rating} stars: {count}\n")
                f.write(f"Average Rating: {report.get('average_rating', 0):.2f}\n\n")

            f.write(f"Missing Data Percentage: {report['missing_percentage']:.2f}%\n")
            if report.get("missing_data"):
                f.write("Missing Data by Column:\n")
                for col, pct in report["missing_data"].items():
                    f.write(f"  {col}: {pct}%\n")
            f.write("\n")

            if "quality_checks" in report:
                f.write("Quality Checks:\n")
                for check, passed in report["quality_checks"].items():
                    status = "✓ PASS" if passed else "✗ FAIL"
                    f.write(f"  {status}: {check}\n")

            f.write("\n" + "=" * 70 + "\n")

        self.logger.info(f"Saved quality report to {filepath}")

        return filepath


if __name__ == "__main__":
    # Example usage and testing
    print("Testing ReviewPreprocessor...\n")

    # Initialize preprocessor
    preprocessor = ReviewPreprocessor()

    try:
        # Load raw data
        print("Loading raw data...")
        df = preprocessor.load_raw_data()
        print(f"✓ Loaded {len(df)} reviews\n")

        # Clean data
        print("Cleaning data...")
        cleaned_df = preprocessor.clean_data(df)
        print(f"✓ Cleaned data: {len(cleaned_df)} reviews\n")

        # Save cleaned data
        print("Saving cleaned data...")
        filepath = preprocessor.save_cleaned_data(cleaned_df)
        print(f"✓ Saved to: {filepath}\n")

        # Save quality report
        print("Saving quality report...")
        report_path = preprocessor.save_quality_report()
        print(f"✓ Report saved to: {report_path}\n")

        # Print summary
        print("=" * 70)
        print("PREPROCESSING SUMMARY")
        print("=" * 70)
        quality = preprocessor.quality_report
        print(f"Total Reviews: {quality['total_reviews']}")
        print(f"Banks: {', '.join(quality['banks'])}")
        print(f"Missing Data: {quality['missing_percentage']:.2f}%")
        if "average_rating" in quality:
            print(f"Average Rating: {quality['average_rating']:.2f}")
        print("=" * 70)

    except PreprocessorError as e:
        print(f"✗ Error: {e}")

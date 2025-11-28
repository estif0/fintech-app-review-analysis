"""
Theme Classification Module for Review Analysis

This module provides theme classification functionality for categorizing
bank app reviews into thematic categories based on keyword matching.

Classes:
    ThemeClassifier: Main class for assigning themes to reviews

Example:
    >>> classifier = ThemeClassifier()
    >>> themes = classifier.classify_review("App crashes frequently, very slow")
    >>> print(themes)
    ['Technical Issues', 'Performance']
"""

import pandas as pd
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter


class ThemeClassifier:
    """
    A class for classifying reviews into thematic categories.

    Uses keyword-based rule matching to assign one or more themes to each review
    based on the presence of theme-specific keywords.

    Attributes:
        theme_definitions (Dict[str, Dict]): Theme definitions with keywords
        logger (logging.Logger): Logger instance for tracking operations

    Example:
        >>> classifier = ThemeClassifier()
        >>> review = "Good app but slow loading"
        >>> themes = classifier.classify_review(review)
        >>> 'Performance' in themes
        True
    """

    def __init__(self, custom_themes: Optional[Dict[str, Dict]] = None) -> None:
        """
        Initialize the ThemeClassifier with theme definitions.

        Args:
            custom_themes (Optional[Dict[str, Dict]]): Custom theme definitions.
                If None, uses default themes. Each theme should have:
                - 'keywords': List of keywords
                - 'description': Theme description
        """
        self.logger = self._setup_logger()

        # Use custom themes or default
        if custom_themes:
            self.theme_definitions = custom_themes
        else:
            self.theme_definitions = self._get_default_themes()

        # Build keyword to theme mapping for efficient lookup
        self.keyword_to_themes = self._build_keyword_mapping()

        self.logger.info(
            f"ThemeClassifier initialized with {len(self.theme_definitions)} themes"
        )

    def _setup_logger(self) -> logging.Logger:
        """
        Set up logging configuration.

        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _get_default_themes(self) -> Dict[str, Dict]:
        """
        Get default theme definitions based on keyword analysis.

        Returns:
            Dict[str, Dict]: Dictionary of theme definitions
        """
        themes = {
            "User Experience": {
                "description": "Overall user experience, ease of use, and interface quality",
                "keywords": [
                    "good",
                    "best",
                    "great",
                    "amazing",
                    "excellent",
                    "super",
                    "nice",
                    "easy",
                    "simple",
                    "fast",
                    "quick",
                    "smooth",
                    "user",
                    "interface",
                    "design",
                    "clean",
                    "friendly",
                ],
            },
            "Technical Issues": {
                "description": "Crashes, bugs, errors, and technical problems",
                "keywords": [
                    "crash",
                    "bug",
                    "error",
                    "problem",
                    "issue",
                    "doesnt work",
                    "not working",
                    "broken",
                    "fail",
                    "freeze",
                    "hang",
                    "stuck",
                    "glitch",
                    "cant",
                    "wont",
                    "developer",
                    "fix",
                    "worest",
                ],
            },
            "Performance": {
                "description": "Speed, loading times, and app responsiveness",
                "keywords": [
                    "slow",
                    "fast",
                    "speed",
                    "loading",
                    "lag",
                    "quick",
                    "performance",
                    "wait",
                    "time",
                    "delay",
                    "responsive",
                ],
            },
            "Features & Functionality": {
                "description": "App features, services, and capabilities",
                "keywords": [
                    "feature",
                    "service",
                    "transaction",
                    "transfer",
                    "balance",
                    "account",
                    "history",
                    "statement",
                    "bill",
                    "payment",
                    "deposit",
                    "withdrawal",
                    "loan",
                    "cardless",
                ],
            },
            "Updates & Improvements": {
                "description": "App updates, improvements, and version changes",
                "keywords": [
                    "update",
                    "new",
                    "version",
                    "improve",
                    "better",
                    "change",
                    "upgrade",
                    "latest",
                    "recent",
                    "added",
                ],
            },
            "Authentication & Security": {
                "description": "Login, password, security, and access issues",
                "keywords": [
                    "login",
                    "password",
                    "pin",
                    "fingerprint",
                    "biometric",
                    "security",
                    "safe",
                    "secure",
                    "authentication",
                    "verify",
                    "otp",
                    "code",
                    "locked",
                    "unlock",
                    "sign",
                ],
            },
            "Customer Support": {
                "description": "Customer service, support, and communication",
                "keywords": [
                    "support",
                    "help",
                    "customer",
                    "service",
                    "contact",
                    "call",
                    "email",
                    "response",
                    "assist",
                    "thank",
                    "please",
                ],
            },
            "Negative Experience": {
                "description": "Strong negative feedback and dissatisfaction",
                "keywords": [
                    "worst",
                    "terrible",
                    "horrible",
                    "awful",
                    "bad",
                    "poor",
                    "useless",
                    "garbage",
                    "trash",
                    "hate",
                    "never",
                    "ever",
                    "worst ever",
                    "dont",
                ],
            },
        }
        return themes

    def _build_keyword_mapping(self) -> Dict[str, Set[str]]:
        """
        Build a mapping from keywords to themes for efficient lookup.

        Returns:
            Dict[str, Set[str]]: Mapping of keywords to theme names
        """
        keyword_map = {}
        for theme_name, theme_data in self.theme_definitions.items():
            for keyword in theme_data["keywords"]:
                keyword_lower = keyword.lower()
                if keyword_lower not in keyword_map:
                    keyword_map[keyword_lower] = set()
                keyword_map[keyword_lower].add(theme_name)
        return keyword_map

    def classify_review(
        self,
        review_text: str,
        preprocessed_text: Optional[str] = None,
        min_theme_confidence: int = 1,
    ) -> List[str]:
        """
        Classify a single review into themes.

        Args:
            review_text (str): Original review text
            preprocessed_text (Optional[str]): Preprocessed review text.
                If None, uses original text.
            min_theme_confidence (int): Minimum keyword matches for theme assignment.
                Defaults to 1.

        Returns:
            List[str]: List of theme names assigned to the review

        Example:
            >>> classifier = ThemeClassifier()
            >>> classifier.classify_review("App crashes often, very buggy")
            ['Technical Issues', 'Negative Experience']
        """
        if not review_text and not preprocessed_text:
            return []

        # Use preprocessed text if available, otherwise original
        text = (preprocessed_text or review_text).lower()

        # Count theme matches
        theme_scores = Counter()

        # Check for keyword matches
        for keyword, themes in self.keyword_to_themes.items():
            if keyword in text:
                for theme in themes:
                    theme_scores[theme] += 1

        # Filter themes by minimum confidence
        assigned_themes = [
            theme
            for theme, score in theme_scores.items()
            if score >= min_theme_confidence
        ]

        return sorted(assigned_themes)

    def classify_reviews(
        self,
        reviews: List[str],
        preprocessed_texts: Optional[List[str]] = None,
        min_theme_confidence: int = 1,
        show_progress: bool = True,
    ) -> List[List[str]]:
        """
        Classify multiple reviews into themes.

        Args:
            reviews (List[str]): List of review texts
            preprocessed_texts (Optional[List[str]]): List of preprocessed texts.
                If None, uses original reviews.
            min_theme_confidence (int): Minimum keyword matches for theme assignment.
            show_progress (bool): Whether to show progress. Defaults to True.

        Returns:
            List[List[str]]: List of theme lists for each review

        Example:
            >>> classifier = ThemeClassifier()
            >>> reviews = ["Great app!", "Slow loading"]
            >>> classifier.classify_reviews(reviews, show_progress=False)
            [['User Experience'], ['Performance']]
        """
        if show_progress:
            self.logger.info(f"Classifying {len(reviews)} reviews...")

        results = []
        for idx, review in enumerate(reviews):
            preprocessed = preprocessed_texts[idx] if preprocessed_texts else None
            themes = self.classify_review(review, preprocessed, min_theme_confidence)
            results.append(themes)

            # Log progress every 100 reviews
            if show_progress and (idx + 1) % 100 == 0:
                self.logger.info(f"Classified {idx + 1}/{len(reviews)} reviews")

        if show_progress:
            self.logger.info(f"✓ Completed classifying {len(reviews)} reviews")

        return results

    def classify_dataframe(
        self,
        df: pd.DataFrame,
        review_column: str = "review",
        preprocessed_column: Optional[str] = "preprocessed_text",
        output_column: str = "themes",
        min_theme_confidence: int = 1,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Classify reviews in a DataFrame and add theme column.

        Args:
            df (pd.DataFrame): DataFrame with review data
            review_column (str): Column name with review text. Defaults to 'review'.
            preprocessed_column (Optional[str]): Column with preprocessed text.
                If None, uses review_column. Defaults to 'preprocessed_text'.
            output_column (str): Name for theme output column. Defaults to 'themes'.
            min_theme_confidence (int): Minimum keyword matches for theme assignment.
            show_progress (bool): Whether to show progress. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame with added theme column

        Raises:
            ValueError: If review_column doesn't exist

        Example:
            >>> classifier = ThemeClassifier()
            >>> df = pd.DataFrame({'review': ['Great!', 'Buggy app']})
            >>> df_themed = classifier.classify_dataframe(df, show_progress=False)
            >>> 'themes' in df_themed.columns
            True
        """
        if review_column not in df.columns:
            raise ValueError(
                f"Column '{review_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        if show_progress:
            self.logger.info(f"Classifying {len(df)} reviews in DataFrame...")

        # Get reviews and preprocessed texts
        reviews = df[review_column].tolist()
        preprocessed_texts = None

        if preprocessed_column and preprocessed_column in df.columns:
            preprocessed_texts = df[preprocessed_column].fillna("").tolist()

        # Classify all reviews
        themes = self.classify_reviews(
            reviews, preprocessed_texts, min_theme_confidence, show_progress
        )

        # Add to DataFrame
        df_copy = df.copy()
        df_copy[output_column] = themes

        if show_progress:
            self.logger.info(f"✓ Added '{output_column}' column to DataFrame")

        return df_copy

    def get_theme_statistics(
        self,
        df: pd.DataFrame,
        theme_column: str = "themes",
        group_by: Optional[str] = None,
    ) -> Dict:
        """
        Calculate theme distribution statistics.

        Args:
            df (pd.DataFrame): DataFrame with theme classifications
            theme_column (str): Column containing theme lists. Defaults to 'themes'.
            group_by (Optional[str]): Column to group by (e.g., 'bank').
                If None, calculates overall statistics.

        Returns:
            Dict: Dictionary with theme statistics

        Example:
            >>> classifier = ThemeClassifier()
            >>> df = pd.DataFrame({
            ...     'themes': [['User Experience'], ['Technical Issues']],
            ...     'bank': ['BOA', 'CBE']
            ... })
            >>> stats = classifier.get_theme_statistics(df)
            >>> 'theme_counts' in stats
            True
        """
        if theme_column not in df.columns:
            raise ValueError(f"Column '{theme_column}' not found")

        if group_by:
            # Group statistics
            stats = {}
            for group_name, group_df in df.groupby(group_by):
                stats[group_name] = self._calculate_theme_stats(group_df, theme_column)
            return stats
        else:
            # Overall statistics
            return self._calculate_theme_stats(df, theme_column)

    def _calculate_theme_stats(self, df: pd.DataFrame, theme_column: str) -> Dict:
        """
        Calculate theme statistics for a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with themes
            theme_column (str): Column containing theme lists

        Returns:
            Dict: Statistics dictionary
        """
        # Count all themes
        all_themes = []
        for themes_list in df[theme_column]:
            if isinstance(themes_list, list):
                all_themes.extend(themes_list)

        theme_counts = Counter(all_themes)

        # Calculate percentages
        total_reviews = len(df)
        theme_percentages = {
            theme: (count / total_reviews * 100)
            for theme, count in theme_counts.items()
        }

        # Reviews per theme count
        themes_per_review = df[theme_column].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        stats = {
            "total_reviews": total_reviews,
            "theme_counts": dict(theme_counts),
            "theme_percentages": theme_percentages,
            "avg_themes_per_review": themes_per_review.mean(),
            "reviews_without_themes": (themes_per_review == 0).sum(),
            "top_themes": theme_counts.most_common(10),
        }

        return stats

    def get_theme_definitions(self) -> Dict[str, str]:
        """
        Get theme names and descriptions.

        Returns:
            Dict[str, str]: Mapping of theme names to descriptions

        Example:
            >>> classifier = ThemeClassifier()
            >>> themes = classifier.get_theme_definitions()
            >>> 'User Experience' in themes
            True
        """
        return {
            name: data["description"] for name, data in self.theme_definitions.items()
        }


def main():
    """
    Main function to demonstrate theme classification on reviews.

    Loads preprocessed reviews, classifies themes, and displays results.
    """
    import os

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # File paths
    input_file = "data/processed/preprocessed_reviews.csv"
    output_file = "data/processed/themed_reviews.csv"

    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    # Load reviews
    logger.info(f"Loading reviews from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} reviews")

    # Initialize classifier
    classifier = ThemeClassifier()

    # Display theme definitions
    logger.info("\n=== Theme Definitions ===")
    for theme, description in classifier.get_theme_definitions().items():
        print(f"{theme}: {description}")

    # Classify reviews
    df_themed = classifier.classify_dataframe(df)

    # Calculate statistics
    logger.info("\n=== Overall Theme Statistics ===")
    overall_stats = classifier.get_theme_statistics(df_themed)

    print(f"Total reviews: {overall_stats['total_reviews']}")
    print(f"Reviews without themes: {overall_stats['reviews_without_themes']}")
    print(f"Average themes per review: {overall_stats['avg_themes_per_review']:.2f}")
    print(f"\nTop 10 themes:")
    for theme, count in overall_stats["top_themes"]:
        pct = overall_stats["theme_percentages"][theme]
        print(f"  {theme}: {count} ({pct:.1f}%)")

    # Statistics by bank
    logger.info("\n=== Theme Statistics by Bank ===")
    bank_stats = classifier.get_theme_statistics(df_themed, group_by="bank")

    for bank, stats in bank_stats.items():
        print(f"\n{bank}:")
        print(f"  Total reviews: {stats['total_reviews']}")
        print(f"  Avg themes per review: {stats['avg_themes_per_review']:.2f}")
        print(f"  Top 5 themes:")
        for theme, count in stats["top_themes"][:5]:
            pct = stats["theme_percentages"][theme]
            print(f"    {theme}: {count} ({pct:.1f}%)")

    # Save results
    logger.info(f"\nSaving themed reviews to {output_file}")
    # Convert list to comma-separated string for CSV
    df_themed["themes_str"] = df_themed["themes"].apply(
        lambda x: ", ".join(x) if isinstance(x, list) else ""
    )
    df_themed.to_csv(output_file, index=False)

    logger.info("✓ Theme classification complete!")


if __name__ == "__main__":
    main()

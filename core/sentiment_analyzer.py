"""
Sentiment Analysis Module for Bank App Reviews

This module provides sentiment analysis functionality using VADER (Valence Aware
Dictionary and sEntiment Reasoner) for analyzing customer reviews of mobile
banking applications.

Classes:
    SentimentAnalyzer: Main class for performing sentiment analysis on reviews

Example:
    >>> analyzer = SentimentAnalyzer()
    >>> reviews = ["Great app!", "Terrible experience"]
    >>> results = analyzer.analyze_reviews(reviews)
    >>> print(results[0]['sentiment_label'])
    'Positive'
"""

import pandas as pd
import logging
from typing import List, Dict, Optional, Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """
    A class for analyzing sentiment in text reviews using VADER.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and
    rule-based sentiment analysis tool specifically attuned to sentiments
    expressed in social media.

    Attributes:
        analyzer (SentimentIntensityAnalyzer): VADER sentiment analyzer instance
        positive_threshold (float): Threshold for positive sentiment classification
        negative_threshold (float): Threshold for negative sentiment classification
        logger (logging.Logger): Logger instance for tracking operations

    Example:
        >>> analyzer = SentimentAnalyzer()
        >>> result = analyzer.analyze_text("This app is amazing!")
        >>> print(result['sentiment_label'])
        'Positive'
        >>> print(result['sentiment_score'])
        0.8439
    """

    def __init__(
        self,
        positive_threshold: float = 0.05,
        negative_threshold: float = -0.05,
        use_rating_boost: bool = True,
    ) -> None:
        """
        Initialize the SentimentAnalyzer with VADER.

        Args:
            positive_threshold (float): Minimum compound score for positive sentiment.
                Defaults to 0.05 (VADER recommended).
            negative_threshold (float): Maximum compound score for negative sentiment.
                Defaults to -0.05 (VADER recommended).
            use_rating_boost (bool): Whether to use rating-based adjustments.
                Defaults to True.

        Raises:
            ValueError: If thresholds are invalid (positive <= negative)
        """
        if positive_threshold <= negative_threshold:
            raise ValueError(
                f"Positive threshold ({positive_threshold}) must be greater "
                f"than negative threshold ({negative_threshold})"
            )

        self.analyzer = SentimentIntensityAnalyzer()
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.use_rating_boost = use_rating_boost
        self.logger = self._setup_logger()

        # Define negative problem indicators for banking apps
        self.negative_patterns = [
            "not working",
            "doesn't work",
            "doesnt work",
            "won't work",
            "wont work",
            "can't",
            "cant",
            "couldn't",
            "couldnt",
            "worst",
            "terrible",
            "horrible",
            "awful",
            "crash",
            "bug",
            "broken",
            "fail",
            "problem",
            "issue",
            "slow",
            "loading",
            "freeze",
            "stuck",
            "error",
            "useless",
            "waste",
            "disappointed",
            "frustrated",
            "frustrating",
        ]

        self.logger.info(
            f"SentimentAnalyzer initialized with thresholds: "
            f"positive={positive_threshold}, negative={negative_threshold}, "
            f"rating_boost={use_rating_boost}"
        )

    def _setup_logger(self) -> logging.Logger:
        """
        Set up logging configuration for the sentiment analyzer.

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

    def analyze_text(self, text: str, rating: Optional[int] = None) -> Dict[str, any]:
        """
        Analyze sentiment of a single text string.

        Uses VADER to compute sentiment scores and classify the text as
        Positive, Negative, or Neutral. Optionally incorporates rating
        information for improved accuracy.

        Args:
            text (str): The text to analyze
            rating (Optional[int]): Star rating (1-5) if available. Used to
                adjust sentiment classification for edge cases.

        Returns:
            Dict[str, any]: Dictionary containing:
                - sentiment_score (float): Compound score from -1 (negative) to 1 (positive)
                - sentiment_label (str): 'Positive', 'Negative', or 'Neutral'
                - pos_score (float): Positive sentiment component (0-1)
                - neu_score (float): Neutral sentiment component (0-1)
                - neg_score (float): Negative sentiment component (0-1)
                - rating_adjusted (bool): Whether rating influenced classification

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> result = analyzer.analyze_text("Love this bank app!")
            >>> result['sentiment_label']
            'Positive'
        """
        if not text or not isinstance(text, str):
            self.logger.warning(f"Invalid text input: {text}")
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "Neutral",
                "pos_score": 0.0,
                "neu_score": 1.0,
                "neg_score": 0.0,
                "rating_adjusted": False,
            }

        # Get sentiment scores from VADER
        scores = self.analyzer.polarity_scores(text)
        compound_score = scores["compound"]
        rating_adjusted = False

        # Classify sentiment based on compound score
        if compound_score >= self.positive_threshold:
            label = "Positive"
        elif compound_score <= self.negative_threshold:
            label = "Negative"
        else:
            label = "Neutral"

        # Apply rating-based adjustment if enabled and rating provided
        if self.use_rating_boost and rating is not None:
            original_label = label

            # Check for negative problem patterns in text
            text_lower = text.lower()
            has_negative_pattern = any(
                pattern in text_lower for pattern in self.negative_patterns
            )

            # Adjust sentiment based on rating and patterns
            if rating <= 2:
                # 1-2 star reviews should be negative
                if label != "Negative":
                    # Strong override for 1-star + negative patterns
                    if rating == 1 and has_negative_pattern:
                        label = "Negative"
                        rating_adjusted = True
                    # Moderate override for low rating + neutral/weak positive
                    elif rating <= 2 and compound_score < 0.5:
                        label = "Negative"
                        rating_adjusted = True

            elif rating >= 4:
                # 4-5 star reviews should be positive
                if label == "Neutral" and compound_score > -0.3:
                    label = "Positive"
                    rating_adjusted = True
                # Don't override clearly negative high-star reviews (sarcasm/complaints)
                elif label == "Negative" and compound_score < -0.5:
                    pass  # Keep as negative (likely legitimate complaint)

            # Adjust compound score if classification changed
            if rating_adjusted and label != original_label:
                if label == "Negative" and compound_score >= 0:
                    # Adjust score to match negative classification
                    compound_score = min(compound_score, self.negative_threshold - 0.05)
                elif label == "Positive" and compound_score <= 0:
                    # Adjust score to match positive classification
                    compound_score = max(compound_score, self.positive_threshold + 0.05)

        return {
            "sentiment_score": round(compound_score, 4),
            "sentiment_label": label,
            "pos_score": round(scores["pos"], 4),
            "neu_score": round(scores["neu"], 4),
            "neg_score": round(scores["neg"], 4),
            "rating_adjusted": rating_adjusted,
        }

    def analyze_reviews(
        self,
        reviews: List[str],
        ratings: Optional[List[int]] = None,
        show_progress: bool = True,
    ) -> List[Dict[str, any]]:
        """
        Analyze sentiment for a list of reviews.

        Args:
            reviews (List[str]): List of review texts to analyze
            ratings (Optional[List[int]]): List of ratings corresponding to reviews.
                If provided, used to improve sentiment classification.
            show_progress (bool): Whether to show progress messages. Defaults to True.

        Returns:
            List[Dict[str, any]]: List of sentiment analysis results, one per review

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> reviews = ["Great app!", "Terrible bugs", "It's okay"]
            >>> ratings = [5, 1, 3]
            >>> results = analyzer.analyze_reviews(reviews, ratings)
            >>> len(results)
            3
        """
        if show_progress:
            self.logger.info(f"Analyzing sentiment for {len(reviews)} reviews...")

        results = []
        for idx, review in enumerate(reviews):
            rating = ratings[idx] if ratings and idx < len(ratings) else None
            result = self.analyze_text(review, rating)
            results.append(result)

            # Log progress every 100 reviews
            if show_progress and (idx + 1) % 100 == 0:
                self.logger.info(f"Processed {idx + 1}/{len(reviews)} reviews")

        if show_progress:
            # Count rating adjustments
            adjusted_count = sum(1 for r in results if r.get("rating_adjusted", False))
            self.logger.info(
                f"✓ Completed sentiment analysis for {len(reviews)} reviews"
            )
            if adjusted_count > 0:
                self.logger.info(
                    f"  Rating-based adjustments applied: {adjusted_count} reviews"
                )

        return results

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "review",
        rating_column: Optional[str] = "rating",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Analyze sentiment for reviews in a pandas DataFrame.

        Adds sentiment analysis columns to the DataFrame:
        - sentiment_score: Compound sentiment score (-1 to 1)
        - sentiment_label: Classification (Positive/Negative/Neutral)
        - pos_score: Positive component score
        - neu_score: Neutral component score
        - neg_score: Negative component score
        - rating_adjusted: Whether rating influenced classification

        Args:
            df (pd.DataFrame): DataFrame containing reviews
            text_column (str): Name of the column containing review text.
                Defaults to 'review'.
            rating_column (Optional[str]): Name of rating column for hybrid analysis.
                Defaults to 'rating'. Set to None to disable rating boost.
            show_progress (bool): Whether to show progress messages. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame with added sentiment analysis columns

        Raises:
            ValueError: If text_column doesn't exist in DataFrame

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> df = pd.DataFrame({'review': ["Great!", "Bad!"], 'rating': [5, 1]})
            >>> df_analyzed = analyzer.analyze_dataframe(df)
            >>> 'sentiment_label' in df_analyzed.columns
            True
        """
        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        if show_progress:
            self.logger.info(f"Analyzing {len(df)} reviews from DataFrame...")

        # Get reviews and ratings
        reviews = df[text_column].tolist()
        ratings = None
        if rating_column and rating_column in df.columns:
            ratings = df[rating_column].tolist()
            if show_progress:
                self.logger.info(
                    f"Using rating column '{rating_column}' for hybrid analysis"
                )

        # Analyze all reviews
        results = self.analyze_reviews(reviews, ratings, show_progress=show_progress)

        # Add sentiment columns to DataFrame
        df_copy = df.copy()
        df_copy["sentiment_score"] = [r["sentiment_score"] for r in results]
        df_copy["sentiment_label"] = [r["sentiment_label"] for r in results]
        df_copy["pos_score"] = [r["pos_score"] for r in results]
        df_copy["neu_score"] = [r["neu_score"] for r in results]
        df_copy["neg_score"] = [r["neg_score"] for r in results]
        df_copy["rating_adjusted"] = [r.get("rating_adjusted", False) for r in results]

        if show_progress:
            self.logger.info("✓ Sentiment columns added to DataFrame")

        return df_copy

    def get_sentiment_statistics(
        self, df: pd.DataFrame, group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Calculate sentiment statistics from analyzed DataFrame.

        Computes aggregate statistics like average sentiment score,
        sentiment label distribution, and coverage.

        Args:
            df (pd.DataFrame): DataFrame with sentiment analysis results
            group_by (Optional[str]): Column name to group statistics by
                (e.g., 'bank', 'rating'). Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with sentiment statistics

        Raises:
            ValueError: If required sentiment columns are missing

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> df_analyzed = analyzer.analyze_dataframe(df)
            >>> stats = analyzer.get_sentiment_statistics(df_analyzed, group_by='bank')
            >>> print(stats)
        """
        required_cols = ["sentiment_score", "sentiment_label"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                "Run analyze_dataframe() first."
            )

        if group_by:
            if group_by not in df.columns:
                raise ValueError(f"Group column '{group_by}' not found in DataFrame")

            # Group statistics
            stats = (
                df.groupby(group_by)
                .agg(
                    {
                        "sentiment_score": ["mean", "std", "min", "max"],
                        "sentiment_label": lambda x: x.value_counts().to_dict(),
                    }
                )
                .round(4)
            )

            # Add sentiment label counts as separate columns
            sentiment_counts = (
                df.groupby(group_by)["sentiment_label"]
                .value_counts()
                .unstack(fill_value=0)
            )
            stats = pd.concat([stats, sentiment_counts], axis=1)

        else:
            # Overall statistics
            stats = pd.DataFrame(
                {
                    "avg_sentiment": [df["sentiment_score"].mean()],
                    "std_sentiment": [df["sentiment_score"].std()],
                    "min_sentiment": [df["sentiment_score"].min()],
                    "max_sentiment": [df["sentiment_score"].max()],
                    "positive_count": [(df["sentiment_label"] == "Positive").sum()],
                    "neutral_count": [(df["sentiment_label"] == "Neutral").sum()],
                    "negative_count": [(df["sentiment_label"] == "Negative").sum()],
                    "total_reviews": [len(df)],
                }
            ).round(4)

            # Add percentages
            stats["positive_pct"] = (
                stats["positive_count"] / stats["total_reviews"] * 100
            ).round(2)
            stats["neutral_pct"] = (
                stats["neutral_count"] / stats["total_reviews"] * 100
            ).round(2)
            stats["negative_pct"] = (
                stats["negative_count"] / stats["total_reviews"] * 100
            ).round(2)

        return stats

    def classify_by_rating(
        self, df: pd.DataFrame, rating_column: str = "rating"
    ) -> pd.DataFrame:
        """
        Analyze sentiment-rating correlation.

        Computes average sentiment scores for each rating level to
        validate sentiment analysis against user ratings.

        Args:
            df (pd.DataFrame): DataFrame with sentiment and rating data
            rating_column (str): Name of rating column. Defaults to 'rating'.

        Returns:
            pd.DataFrame: Sentiment statistics grouped by rating

        Example:
            >>> analyzer = SentimentAnalyzer()
            >>> df_analyzed = analyzer.analyze_dataframe(df)
            >>> rating_analysis = analyzer.classify_by_rating(df_analyzed)
        """
        if rating_column not in df.columns:
            raise ValueError(f"Rating column '{rating_column}' not found")

        if "sentiment_score" not in df.columns:
            raise ValueError(
                "Sentiment analysis not performed. Run analyze_dataframe() first."
            )

        return self.get_sentiment_statistics(df, group_by=rating_column)


def main():
    """
    Main function to demonstrate sentiment analysis on cleaned reviews.

    Loads cleaned reviews, performs sentiment analysis, and saves results.
    """
    import os

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # File paths
    input_file = "data/processed/cleaned_reviews.csv"
    output_file = "data/processed/sentiment_analyzed_reviews.csv"
    stats_file = "data/processed/sentiment_statistics.csv"

    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    # Load cleaned reviews
    logger.info(f"Loading cleaned reviews from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} reviews")

    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()

    # Analyze sentiment
    df_analyzed = analyzer.analyze_dataframe(df)

    # Calculate statistics
    logger.info("\n=== Overall Sentiment Statistics ===")
    overall_stats = analyzer.get_sentiment_statistics(df_analyzed)
    print(overall_stats.to_string())

    logger.info("\n=== Sentiment by Bank ===")
    bank_stats = analyzer.get_sentiment_statistics(df_analyzed, group_by="bank")
    print(bank_stats.to_string())

    logger.info("\n=== Sentiment by Rating ===")
    rating_stats = analyzer.classify_by_rating(df_analyzed)
    print(rating_stats.to_string())

    # Save results
    logger.info(f"\nSaving analyzed reviews to {output_file}")
    df_analyzed.to_csv(output_file, index=False)

    logger.info(f"Saving statistics to {stats_file}")
    overall_stats.to_csv(stats_file, index=False)

    # Calculate coverage
    coverage = (len(df_analyzed) / len(df) * 100) if len(df) > 0 else 0
    logger.info(f"\n✓ Sentiment analysis complete!")
    logger.info(f"✓ Coverage: {coverage:.2f}% ({len(df_analyzed)}/{len(df)} reviews)")
    logger.info(f"✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()

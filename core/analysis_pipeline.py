"""
End-to-End Analysis Pipeline for Bank App Reviews

This module provides a complete analysis pipeline that integrates:
- Sentiment analysis (VADER with rating-based adjustments)
- Text preprocessing (cleaning, tokenization, lemmatization)
- Keyword extraction (TF-IDF)
- Theme classification (rule-based multi-label)

Classes:
    AnalysisPipeline: Main pipeline orchestrator

Example:
    >>> pipeline = AnalysisPipeline()
    >>> results = pipeline.run('data/processed/cleaned_reviews.csv')
    >>> print(f"Analyzed {len(results)} reviews")
"""

import pandas as pd
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sentiment_analyzer import SentimentAnalyzer
from core.text_preprocessor import TextPreprocessor
from core.keyword_extractor import KeywordExtractor
from core.theme_classifier import ThemeClassifier


class AnalysisPipeline:
    """
    End-to-end analysis pipeline for bank app reviews.

    Orchestrates the complete analysis workflow from cleaned data
    to enriched dataset with sentiment, themes, and keywords.

    Attributes:
        sentiment_analyzer (SentimentAnalyzer): Sentiment analysis module
        text_preprocessor (TextPreprocessor): Text preprocessing module
        keyword_extractor (KeywordExtractor): Keyword extraction module
        theme_classifier (ThemeClassifier): Theme classification module
        logger (logging.Logger): Pipeline logger

    Example:
        >>> pipeline = AnalysisPipeline()
        >>> results = pipeline.run('data/processed/cleaned_reviews.csv')
    """

    def __init__(
        self,
        output_dir: str = "data/processed",
        use_rating_boost: bool = True,
        custom_themes: Optional[Dict] = None,
    ) -> None:
        """
        Initialize the analysis pipeline with all modules.

        Args:
            output_dir (str): Directory for output files. Defaults to 'data/processed'.
            use_rating_boost (bool): Enable rating-based sentiment adjustment.
                Defaults to True.
            custom_themes (Optional[Dict]): Custom theme definitions.
                If None, uses default themes.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all analysis modules
        self.sentiment_analyzer = SentimentAnalyzer(use_rating_boost=use_rating_boost)
        self.text_preprocessor = TextPreprocessor()
        self.keyword_extractor = KeywordExtractor()
        self.theme_classifier = ThemeClassifier(custom_themes=custom_themes)

        # Setup logging
        self.logger = self._setup_logger()
        self.logger.info("AnalysisPipeline initialized with all modules")

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

    def run(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        save_intermediate: bool = True,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Run the complete analysis pipeline.

        Pipeline steps:
        1. Load cleaned reviews
        2. Sentiment analysis (VADER + rating boost)
        3. Text preprocessing (cleaning, tokenization, lemmatization)
        4. Keyword extraction (TF-IDF)
        5. Theme classification (rule-based multi-label)
        6. Save enriched dataset

        Args:
            input_file (str): Path to cleaned reviews CSV
            output_file (Optional[str]): Path for output file.
                If None, uses 'data/processed/analyzed_reviews.csv'
            save_intermediate (bool): Save intermediate results.
                Defaults to True.
            show_progress (bool): Show progress messages. Defaults to True.

        Returns:
            pd.DataFrame: Enriched dataset with all analysis results

        Example:
            >>> pipeline = AnalysisPipeline()
            >>> df = pipeline.run('data/processed/cleaned_reviews.csv')
            >>> print(df.columns)
        """
        start_time = datetime.now()
        self.logger.info("=" * 70)
        self.logger.info("STARTING END-TO-END ANALYSIS PIPELINE")
        self.logger.info("=" * 70)

        # Step 1: Load data
        self.logger.info("\n[STEP 1/5] Loading cleaned reviews...")
        df = self._load_data(input_file)

        # Step 2: Sentiment analysis
        self.logger.info("\n[STEP 2/5] Running sentiment analysis...")
        df = self._run_sentiment_analysis(df, show_progress)
        if save_intermediate:
            self._save_intermediate(df, "sentiment_analyzed_reviews.csv")

        # Step 3: Text preprocessing
        self.logger.info("\n[STEP 3/5] Preprocessing text...")
        df = self._run_text_preprocessing(df, show_progress)
        if save_intermediate:
            self._save_intermediate(df, "preprocessed_reviews.csv")

        # Step 4: Keyword extraction
        self.logger.info("\n[STEP 4/5] Extracting keywords...")
        keyword_results = self._run_keyword_extraction(df, show_progress)
        if save_intermediate:
            self._save_keyword_results(keyword_results)

        # Step 5: Theme classification
        self.logger.info("\n[STEP 5/5] Classifying themes...")
        df = self._run_theme_classification(df, show_progress)

        # Save final results
        output_path = output_file or str(self.output_dir / "analyzed_reviews.csv")
        self._save_final_results(df, output_path)

        # Generate summary
        self._generate_summary(df, start_time)

        self.logger.info("\n" + "=" * 70)
        self.logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 70)

        return df

    def _load_data(self, input_file: str) -> pd.DataFrame:
        """Load cleaned reviews from CSV file."""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        df = pd.read_csv(input_file)
        self.logger.info(f"✓ Loaded {len(df)} reviews from {input_file}")

        # Validate required columns
        required_cols = ["review", "rating", "date", "bank", "source"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return df

    def _run_sentiment_analysis(
        self, df: pd.DataFrame, show_progress: bool
    ) -> pd.DataFrame:
        """Run sentiment analysis on reviews."""
        df_analyzed = self.sentiment_analyzer.analyze_dataframe(
            df,
            text_column="review",
            rating_column="rating",
            show_progress=show_progress,
        )

        # Log statistics
        sentiment_dist = df_analyzed["sentiment_label"].value_counts().to_dict()
        adjusted_count = df_analyzed["rating_adjusted"].sum()

        self.logger.info(f"✓ Sentiment distribution: {sentiment_dist}")
        self.logger.info(f"✓ Rating-based adjustments: {adjusted_count} reviews")

        return df_analyzed

    def _run_text_preprocessing(
        self, df: pd.DataFrame, show_progress: bool
    ) -> pd.DataFrame:
        """Run text preprocessing on reviews."""
        df_preprocessed = self.text_preprocessor.preprocess_dataframe(
            df,
            text_column="review",
            output_column="preprocessed_text",
            show_progress=show_progress,
        )

        # Get statistics
        stats = self.text_preprocessor.get_token_statistics(
            df_preprocessed["preprocessed_text"].tolist()
        )

        self.logger.info(f"✓ Preprocessed {len(df_preprocessed)} texts")
        self.logger.info(
            f"✓ Average tokens per review: {stats['avg_tokens_per_text']:.2f}"
        )
        self.logger.info(f"✓ Unique tokens: {stats['unique_tokens']}")

        return df_preprocessed

    def _run_keyword_extraction(self, df: pd.DataFrame, show_progress: bool) -> Dict:
        """Run TF-IDF keyword extraction."""
        # Fit on all preprocessed texts
        texts = df["preprocessed_text"].fillna("").tolist()
        self.keyword_extractor.fit(texts)

        # Extract overall keywords
        overall_keywords = self.keyword_extractor.extract_keywords(texts, top_n=50)

        # Extract keywords by bank
        bank_keywords = self.keyword_extractor.extract_keywords_by_group(
            df,
            text_column="preprocessed_text",
            group_column="bank",
            top_n=30,
        )

        # Extract bigrams and trigrams
        bigrams = self.keyword_extractor.extract_bigrams_trigrams(texts, n=2, top_n=20)
        trigrams = self.keyword_extractor.extract_bigrams_trigrams(texts, n=3, top_n=20)

        self.logger.info(f"✓ Extracted {len(overall_keywords)} overall keywords")
        self.logger.info(f"✓ Extracted keywords for {len(bank_keywords)} banks")
        self.logger.info(
            f"✓ Extracted {len(bigrams)} bigrams, {len(trigrams)} trigrams"
        )

        return {
            "overall_keywords": overall_keywords,
            "bank_keywords": bank_keywords,
            "bigrams": bigrams,
            "trigrams": trigrams,
        }

    def _run_theme_classification(
        self, df: pd.DataFrame, show_progress: bool
    ) -> pd.DataFrame:
        """Run theme classification on reviews."""
        df_themed = self.theme_classifier.classify_dataframe(
            df,
            review_column="review",
            preprocessed_column="preprocessed_text",
            output_column="themes",
            show_progress=show_progress,
        )

        # Get statistics
        overall_stats = self.theme_classifier.get_theme_statistics(df_themed)
        bank_stats = self.theme_classifier.get_theme_statistics(
            df_themed, group_by="bank"
        )

        self.logger.info(
            f"✓ Assigned themes to {overall_stats['total_reviews']} reviews"
        )
        self.logger.info(
            f"✓ Average themes per review: {overall_stats['avg_themes_per_review']:.2f}"
        )
        self.logger.info(
            f"✓ Reviews without themes: {overall_stats['reviews_without_themes']}"
        )

        # Log top themes
        top_themes = overall_stats["top_themes"][:5]
        self.logger.info(f"✓ Top 5 themes:")
        for theme, count in top_themes:
            pct = overall_stats["theme_percentages"][theme]
            self.logger.info(f"    - {theme}: {count} ({pct:.1f}%)")

        return df_themed

    def _save_intermediate(self, df: pd.DataFrame, filename: str) -> None:
        """Save intermediate results."""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        self.logger.info(f"  → Saved intermediate results: {output_path}")

    def _save_keyword_results(self, keyword_results: Dict) -> None:
        """Save keyword extraction results."""
        # Save overall keywords
        overall_path = self.output_dir / "keywords_overall.csv"
        pd.DataFrame(
            keyword_results["overall_keywords"], columns=["keyword", "tfidf_score"]
        ).to_csv(overall_path, index=False)

        # Save bank keywords
        bank_keywords_path = self.output_dir / "keywords_by_bank.csv"
        bank_rows = []
        for bank, keywords in keyword_results["bank_keywords"].items():
            for keyword, score in keywords:
                bank_rows.append(
                    {"bank": bank, "keyword": keyword, "tfidf_score": score}
                )
        pd.DataFrame(bank_rows).to_csv(bank_keywords_path, index=False)

        # Save n-grams
        bigrams_path = self.output_dir / "bigrams.csv"
        pd.DataFrame(
            keyword_results["bigrams"], columns=["bigram", "frequency"]
        ).to_csv(bigrams_path, index=False)

        trigrams_path = self.output_dir / "trigrams.csv"
        pd.DataFrame(
            keyword_results["trigrams"], columns=["trigram", "frequency"]
        ).to_csv(trigrams_path, index=False)

        self.logger.info(f"  → Saved keyword results to {self.output_dir}/")

    def _save_final_results(self, df: pd.DataFrame, output_path: str) -> None:
        """Save final enriched dataset."""
        # Convert themes list to comma-separated string for CSV
        df_output = df.copy()
        df_output["themes_str"] = df_output["themes"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else ""
        )

        # Select and order columns
        output_columns = [
            "review",
            "rating",
            "date",
            "bank",
            "source",
            "sentiment_score",
            "sentiment_label",
            "pos_score",
            "neu_score",
            "neg_score",
            "rating_adjusted",
            "preprocessed_text",
            "themes",
            "themes_str",
        ]

        # Only include columns that exist
        output_columns = [col for col in output_columns if col in df_output.columns]

        df_output[output_columns].to_csv(output_path, index=False)
        self.logger.info(f"\n✓ Saved final results: {output_path}")
        self.logger.info(f"  Columns: {len(output_columns)}, Rows: {len(df_output)}")

    def _generate_summary(self, df: pd.DataFrame, start_time: datetime) -> None:
        """Generate and log pipeline summary."""
        duration = (datetime.now() - start_time).total_seconds()

        self.logger.info("\n" + "=" * 70)
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info("=" * 70)

        # Overall statistics
        self.logger.info(f"\nTotal reviews processed: {len(df)}")
        self.logger.info(f"Processing time: {duration:.2f} seconds")
        self.logger.info(f"Average time per review: {duration/len(df)*1000:.2f} ms")

        # Sentiment statistics
        sentiment_dist = df["sentiment_label"].value_counts().to_dict()
        self.logger.info(f"\nSentiment distribution:")
        for label, count in sentiment_dist.items():
            pct = count / len(df) * 100
            self.logger.info(f"  {label}: {count} ({pct:.1f}%)")

        # Theme statistics
        themes_per_review = df["themes"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        self.logger.info(f"\nTheme statistics:")
        self.logger.info(f"  Avg themes per review: {themes_per_review.mean():.2f}")
        self.logger.info(
            f"  Reviews with themes: {(themes_per_review > 0).sum()} ({(themes_per_review > 0).sum()/len(df)*100:.1f}%)"
        )

        # Bank breakdown
        self.logger.info(f"\nReviews by bank:")
        for bank, count in df["bank"].value_counts().items():
            self.logger.info(f"  {bank}: {count}")

    def get_analysis_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive analysis summary statistics.

        Args:
            df (pd.DataFrame): Analyzed DataFrame from pipeline.run()

        Returns:
            Dict: Summary statistics including sentiment, themes, and keywords

        Example:
            >>> pipeline = AnalysisPipeline()
            >>> df = pipeline.run('data/processed/cleaned_reviews.csv')
            >>> summary = pipeline.get_analysis_summary(df)
            >>> print(summary['overall']['total_reviews'])
        """
        summary = {
            "overall": {
                "total_reviews": len(df),
                "date_range": {
                    "earliest": df["date"].min(),
                    "latest": df["date"].max(),
                },
                "banks": df["bank"].nunique(),
            },
            "sentiment": {
                "distribution": df["sentiment_label"].value_counts().to_dict(),
                "average_score": df["sentiment_score"].mean(),
                "rating_adjusted_count": df["rating_adjusted"].sum(),
            },
            "themes": {
                "avg_per_review": df["themes"]
                .apply(lambda x: len(x) if isinstance(x, list) else 0)
                .mean(),
                "reviews_with_themes": (
                    df["themes"].apply(lambda x: len(x) if isinstance(x, list) else 0)
                    > 0
                ).sum(),
            },
            "by_bank": {},
        }

        # Per-bank statistics
        for bank in df["bank"].unique():
            bank_df = df[df["bank"] == bank]
            summary["by_bank"][bank] = {
                "total_reviews": len(bank_df),
                "sentiment_distribution": bank_df["sentiment_label"]
                .value_counts()
                .to_dict(),
                "avg_sentiment_score": bank_df["sentiment_score"].mean(),
                "avg_themes_per_review": bank_df["themes"]
                .apply(lambda x: len(x) if isinstance(x, list) else 0)
                .mean(),
            }

        return summary


def main():
    """
    Main function to run the complete analysis pipeline.

    Demonstrates pipeline usage and generates enriched dataset.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize pipeline
    pipeline = AnalysisPipeline(
        output_dir="data/processed",
        use_rating_boost=True,
    )

    # Run pipeline
    input_file = "data/processed/cleaned_reviews.csv"
    output_file = "data/processed/analyzed_reviews.csv"

    df_analyzed = pipeline.run(
        input_file=input_file,
        output_file=output_file,
        save_intermediate=True,
        show_progress=True,
    )

    # Get and display summary
    summary = pipeline.get_analysis_summary(df_analyzed)

    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nTotal Reviews: {summary['overall']['total_reviews']}")
    print(f"Banks Analyzed: {summary['overall']['banks']}")
    print(
        f"Date Range: {summary['overall']['date_range']['earliest']} to {summary['overall']['date_range']['latest']}"
    )

    print(f"\nSentiment Overview:")
    for label, count in summary["sentiment"]["distribution"].items():
        pct = count / summary["overall"]["total_reviews"] * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    print(f"\nTheme Overview:")
    print(f"  Avg themes per review: {summary['themes']['avg_per_review']:.2f}")
    print(f"  Reviews with themes: {summary['themes']['reviews_with_themes']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

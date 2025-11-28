"""
Keyword Extraction Module using TF-IDF

This module provides keyword and phrase extraction functionality using
TF-IDF (Term Frequency-Inverse Document Frequency) for identifying
important terms and patterns in review text.

Classes:
    KeywordExtractor: Main class for extracting keywords and phrases

Example:
    >>> extractor = KeywordExtractor()
    >>> reviews = ["great app easy to use", "terrible app crashes often"]
    >>> keywords = extractor.extract_keywords(reviews, top_n=5)
    >>> print(keywords[:3])
    [('app', 0.85), ('crash', 0.72), ('great', 0.68)]
"""

import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class KeywordExtractor:
    """
    A class for extracting keywords and phrases using TF-IDF.

    Uses scikit-learn's TfidfVectorizer to identify important terms
    based on their frequency and uniqueness across documents.

    Attributes:
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer
        feature_names (List[str]): List of terms in vocabulary
        logger (logging.Logger): Logger instance for tracking operations

    Example:
        >>> extractor = KeywordExtractor()
        >>> reviews = ["fast easy app", "slow buggy app"]
        >>> keywords = extractor.extract_keywords(reviews, top_n=3)
        >>> keywords[0][0]  # Top keyword
        'fast' or 'slow' or 'buggy'
    """

    def __init__(
        self,
        max_features: Optional[int] = None,
        ngram_range: Tuple[int, int] = (1, 3),
        min_df: int = 2,
        max_df: float = 0.85,
    ) -> None:
        """
        Initialize the KeywordExtractor with TF-IDF parameters.

        Args:
            max_features (Optional[int]): Maximum number of features to extract.
                If None, uses all features. Defaults to None.
            ngram_range (Tuple[int, int]): Range of n-grams to extract.
                (1,1) for unigrams, (1,2) for unigrams+bigrams, etc.
                Defaults to (1,3) for up to trigrams.
            min_df (int): Minimum document frequency for a term to be included.
                Defaults to 2 (must appear in at least 2 documents).
            max_df (float): Maximum document frequency (as proportion).
                Terms appearing in more than this fraction are ignored.
                Defaults to 0.85 (85% of documents).
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.logger = self._setup_logger()

        # Initialize vectorizer (will be fitted later)
        self.vectorizer = None
        self.feature_names = None

        self.logger.info(
            f"KeywordExtractor initialized with ngram_range={ngram_range}, "
            f"min_df={min_df}, max_df={max_df}"
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

    def fit(self, texts: List[str]) -> "KeywordExtractor":
        """
        Fit the TF-IDF vectorizer on the given texts.

        Args:
            texts (List[str]): List of preprocessed texts

        Returns:
            KeywordExtractor: Self for method chaining

        Example:
            >>> extractor = KeywordExtractor()
            >>> extractor.fit(["good app", "bad app"])
            <KeywordExtractor object>
        """
        self.logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} texts...")

        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            lowercase=True,
            stop_words=None,  # Already preprocessed
        )

        # Fit vectorizer
        self.vectorizer.fit(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        self.logger.info(f"✓ Fitted vectorizer with {len(self.feature_names)} features")

        return self

    def extract_keywords(
        self, texts: List[str], top_n: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Extract top keywords from texts based on TF-IDF scores.

        Args:
            texts (List[str]): List of preprocessed texts
            top_n (int): Number of top keywords to return. Defaults to 50.

        Returns:
            List[Tuple[str, float]]: List of (keyword, score) tuples sorted by score

        Raises:
            ValueError: If vectorizer is not fitted

        Example:
            >>> extractor = KeywordExtractor()
            >>> extractor.fit(["good app", "bad app", "great app"])
            >>> keywords = extractor.extract_keywords(["good app"], top_n=2)
            >>> len(keywords)
            2
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit() first.")

        # Transform texts
        tfidf_matrix = self.vectorizer.transform(texts)

        # Sum TF-IDF scores across all documents
        scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()

        # Get keyword-score pairs
        keyword_scores = list(zip(self.feature_names, scores))

        # Sort by score descending
        keyword_scores.sort(key=lambda x: x[1], reverse=True)

        return keyword_scores[:top_n]

    def extract_keywords_by_group(
        self,
        df: pd.DataFrame,
        text_column: str = "preprocessed_text",
        group_column: str = "bank",
        top_n: int = 50,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract keywords grouped by a category (e.g., by bank).

        Args:
            df (pd.DataFrame): DataFrame with text and group columns
            text_column (str): Name of column with preprocessed text.
                Defaults to 'preprocessed_text'.
            group_column (str): Name of column to group by.
                Defaults to 'bank'.
            top_n (int): Number of top keywords per group. Defaults to 50.

        Returns:
            Dict[str, List[Tuple[str, float]]]: Dictionary mapping group names
                to lists of (keyword, score) tuples

        Raises:
            ValueError: If vectorizer is not fitted or columns don't exist

        Example:
            >>> extractor = KeywordExtractor()
            >>> df = pd.DataFrame({
            ...     'preprocessed_text': ['good', 'bad', 'great'],
            ...     'bank': ['A', 'B', 'A']
            ... })
            >>> extractor.fit(df['preprocessed_text'].tolist())
            >>> keywords_by_bank = extractor.extract_keywords_by_group(df, top_n=2)
            >>> 'A' in keywords_by_bank
            True
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit() first.")

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found")

        if group_column not in df.columns:
            raise ValueError(f"Column '{group_column}' not found")

        self.logger.info(
            f"Extracting keywords by {group_column} (top {top_n} per group)..."
        )

        keywords_by_group = {}

        for group_name in df[group_column].unique():
            # Filter texts for this group
            group_texts = df[df[group_column] == group_name][text_column].tolist()

            # Extract keywords
            keywords = self.extract_keywords(group_texts, top_n=top_n)
            keywords_by_group[group_name] = keywords

            self.logger.info(f"  {group_name}: {len(keywords)} keywords extracted")

        self.logger.info("✓ Keyword extraction by group complete")

        return keywords_by_group

    def extract_bigrams_trigrams(
        self, texts: List[str], n: int = 2, top_n: int = 30
    ) -> List[Tuple[str, int]]:
        """
        Extract top n-grams (bigrams or trigrams) by frequency.

        Args:
            texts (List[str]): List of preprocessed texts
            n (int): N-gram size (2 for bigrams, 3 for trigrams). Defaults to 2.
            top_n (int): Number of top n-grams to return. Defaults to 30.

        Returns:
            List[Tuple[str, int]]: List of (ngram, count) tuples sorted by frequency

        Example:
            >>> extractor = KeywordExtractor()
            >>> texts = ["good fast app", "fast great app"]
            >>> bigrams = extractor.extract_bigrams_trigrams(texts, n=2, top_n=2)
            >>> bigrams[0][0]  # Most common bigram
            'fast app' or 'good fast' or 'great app'
        """
        self.logger.info(f"Extracting top {top_n} {n}-grams...")

        ngrams = []
        for text in texts:
            tokens = text.split()
            if len(tokens) >= n:
                # Generate n-grams
                for i in range(len(tokens) - n + 1):
                    ngram = " ".join(tokens[i : i + n])
                    ngrams.append(ngram)

        # Count occurrences
        ngram_counts = Counter(ngrams)
        top_ngrams = ngram_counts.most_common(top_n)

        self.logger.info(f"✓ Extracted {len(top_ngrams)} {n}-grams")

        return top_ngrams

    def get_keyword_context(
        self,
        df: pd.DataFrame,
        keyword: str,
        text_column: str = "review",
        max_examples: int = 5,
    ) -> List[str]:
        """
        Get example reviews containing a specific keyword.

        Args:
            df (pd.DataFrame): DataFrame with review text
            keyword (str): Keyword to search for
            text_column (str): Column containing review text. Defaults to 'review'.
            max_examples (int): Maximum number of examples. Defaults to 5.

        Returns:
            List[str]: List of example reviews containing the keyword

        Example:
            >>> extractor = KeywordExtractor()
            >>> df = pd.DataFrame({'review': ['good app', 'bad app', 'good service']})
            >>> examples = extractor.get_keyword_context(df, 'good', max_examples=2)
            >>> len(examples)
            2
        """
        # Search for keyword (case-insensitive)
        mask = df[text_column].str.lower().str.contains(keyword.lower(), na=False)
        examples = df[mask][text_column].head(max_examples).tolist()

        return examples


def main():
    """
    Main function to demonstrate keyword extraction on reviews.

    Loads preprocessed reviews, extracts keywords, and displays results.
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

    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        logger.info("Please run text_preprocessor.py first")
        return

    # Load preprocessed reviews
    logger.info(f"Loading preprocessed reviews from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} reviews")

    # Get preprocessed texts and handle NaN values
    df["preprocessed_text"] = df["preprocessed_text"].fillna("")
    texts = df["preprocessed_text"].tolist()

    # Filter out empty texts
    non_empty_texts = [t for t in texts if t.strip()]
    logger.info(f"Using {len(non_empty_texts)} non-empty texts for extraction")

    # Initialize and fit keyword extractor
    extractor = KeywordExtractor(ngram_range=(1, 3), min_df=2, max_df=0.85)
    extractor.fit(non_empty_texts)

    # Extract overall keywords
    logger.info("\n=== Top 30 Keywords (Overall) ===")
    keywords = extractor.extract_keywords(non_empty_texts, top_n=30)
    for keyword, score in keywords:
        print(f"  {keyword}: {score:.4f}")

    # Extract keywords by bank
    logger.info("\n=== Top 20 Keywords by Bank ===")
    keywords_by_bank = extractor.extract_keywords_by_group(
        df, text_column="preprocessed_text", group_column="bank", top_n=20
    )

    for bank, keywords in keywords_by_bank.items():
        print(f"\n{bank}:")
        for keyword, score in keywords[:10]:  # Show top 10
            print(f"  {keyword}: {score:.4f}")

    # Extract bigrams and trigrams
    logger.info("\n=== Top 15 Bigrams ===")
    bigrams = extractor.extract_bigrams_trigrams(non_empty_texts, n=2, top_n=15)
    for bigram, count in bigrams:
        print(f"  {bigram}: {count}")

    logger.info("\n=== Top 15 Trigrams ===")
    trigrams = extractor.extract_bigrams_trigrams(non_empty_texts, n=3, top_n=15)
    for trigram, count in trigrams:
        print(f"  {trigram}: {count}")

    logger.info("\n✓ Keyword extraction complete!")


if __name__ == "__main__":
    main()

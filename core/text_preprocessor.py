"""
Text Preprocessing Module for NLP Analysis

This module provides text preprocessing functionality for natural language processing
tasks including tokenization, stopword removal, lemmatization, and text normalization.

Classes:
    TextPreprocessor: Main class for preprocessing text data

Example:
    >>> preprocessor = TextPreprocessor()
    >>> text = "The app is AMAZING!!! Love it so much!!!"
    >>> clean_text = preprocessor.clean_text(text)
    >>> tokens = preprocessor.tokenize(clean_text)
    >>> print(tokens)
    ['app', 'amazing', 'love', 'much']
"""

import re
import string
import logging
from typing import List, Dict, Optional, Set
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    """
    A class for preprocessing text data for NLP analysis.

    Provides methods for cleaning, tokenizing, removing stopwords, and
    lemmatizing text to prepare it for sentiment analysis, topic modeling,
    and keyword extraction.

    Attributes:
        stop_words (Set[str]): Set of stopwords to remove
        lemmatizer (WordNetLemmatizer): NLTK lemmatizer for word normalization
        custom_stop_words (Set[str]): Additional domain-specific stopwords
        logger (logging.Logger): Logger instance for tracking operations

    Example:
        >>> preprocessor = TextPreprocessor()
        >>> text = "The banking app crashes frequently!"
        >>> clean = preprocessor.preprocess_text(text)
        >>> print(clean)
        'banking app crash frequently'
    """

    def __init__(
        self,
        language: str = "english",
        custom_stop_words: Optional[List[str]] = None,
        remove_numbers: bool = True,
        min_word_length: int = 2,
    ) -> None:
        """
        Initialize the TextPreprocessor with configuration parameters.

        Args:
            language (str): Language for stopwords. Defaults to 'english'.
            custom_stop_words (Optional[List[str]]): Additional stopwords to remove.
                Defaults to None.
            remove_numbers (bool): Whether to remove numeric tokens. Defaults to True.
            min_word_length (int): Minimum word length to keep. Defaults to 2.

        Raises:
            LookupError: If NLTK data is not downloaded
        """
        self.language = language
        self.remove_numbers = remove_numbers
        self.min_word_length = min_word_length
        self.logger = self._setup_logger()

        # Download required NLTK data if not present
        self._download_nltk_data()

        # Initialize stopwords
        self.stop_words = set(stopwords.words(language))

        # Add custom stopwords (domain-specific for banking apps)
        self.custom_stop_words = set(custom_stop_words) if custom_stop_words else set()
        self._add_default_custom_stopwords()

        # Combine all stopwords
        self.stop_words.update(self.custom_stop_words)

        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        self.logger.info(
            f"TextPreprocessor initialized with {len(self.stop_words)} stopwords"
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

    def _download_nltk_data(self) -> None:
        """Download required NLTK data packages if not present."""
        required_packages = [
            "punkt",
            "stopwords",
            "wordnet",
            "averaged_perceptron_tagger",
        ]

        for package in required_packages:
            try:
                # Try to find the package
                if package == "punkt":
                    nltk.data.find("tokenizers/punkt")
                elif package == "averaged_perceptron_tagger":
                    nltk.data.find("taggers/averaged_perceptron_tagger")
                else:
                    nltk.data.find(f"corpora/{package}")
            except LookupError:
                self.logger.info(f"Downloading NLTK package: {package}")
                nltk.download(package, quiet=True)

    def _add_default_custom_stopwords(self) -> None:
        """Add default custom stopwords for banking app reviews."""
        default_custom = {
            "app",
            "application",
            "mobile",
            "phone",
            "bank",
            "banking",
            "use",
            "using",
            "used",
            "get",
            "got",
            "would",
            "could",
            "also",
            "really",
            "much",
            "even",
            "still",
            "just",
            "one",
        }
        self.custom_stop_words.update(default_custom)

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.

        Performs lowercase conversion, special character removal,
        whitespace normalization, and optional number removal.

        Args:
            text (str): Raw text to clean

        Returns:
            str: Cleaned text

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.clean_text("The App is GREAT!!! 123")
            'the app is great'
        """
        if not text or not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove mentions and hashtags (social media artifacts)
        text = re.sub(r"@\w+|#\w+", "", text)

        # Remove numbers if specified
        if self.remove_numbers:
            text = re.sub(r"\d+", "", text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text (str): Text to tokenize

        Returns:
            List[str]: List of tokens

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.tokenize("great banking app")
            ['great', 'banking', 'app']
        """
        if not text:
            return []

        # Use NLTK word tokenizer
        tokens = word_tokenize(text)

        # Filter by minimum length
        tokens = [t for t in tokens if len(t) >= self.min_word_length]

        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from token list.

        Args:
            tokens (List[str]): List of tokens

        Returns:
            List[str]: Filtered list of tokens

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.remove_stopwords(['the', 'app', 'is', 'good'])
            ['good']
        """
        return [token for token in tokens if token not in self.stop_words]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base form.

        Reduces words to their dictionary form (e.g., 'running' -> 'run',
        'better' -> 'good').

        Args:
            tokens (List[str]): List of tokens to lemmatize

        Returns:
            List[str]: Lemmatized tokens

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.lemmatize(['crashes', 'running', 'better'])
            ['crash', 'running', 'better']
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preprocess_text(
        self, text: str, remove_stopwords: bool = True, lemmatize: bool = True
    ) -> str:
        """
        Complete text preprocessing pipeline.

        Applies cleaning, tokenization, stopword removal (optional),
        and lemmatization (optional) in sequence.

        Args:
            text (str): Raw text to preprocess
            remove_stopwords (bool): Whether to remove stopwords. Defaults to True.
            lemmatize (bool): Whether to lemmatize tokens. Defaults to True.

        Returns:
            str: Preprocessed text as a single string

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor.preprocess_text("The apps are crashing frequently!")
            'crash frequently'
        """
        # Clean text
        clean = self.clean_text(text)

        # Tokenize
        tokens = self.tokenize(clean)

        # Remove stopwords
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)

        # Lemmatize
        if lemmatize:
            tokens = self.lemmatize(tokens)

        # Join back to string
        return " ".join(tokens)

    def preprocess_texts(
        self,
        texts: List[str],
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        show_progress: bool = True,
    ) -> List[str]:
        """
        Preprocess multiple texts.

        Args:
            texts (List[str]): List of texts to preprocess
            remove_stopwords (bool): Whether to remove stopwords. Defaults to True.
            lemmatize (bool): Whether to lemmatize. Defaults to True.
            show_progress (bool): Whether to show progress. Defaults to True.

        Returns:
            List[str]: List of preprocessed texts

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> texts = ["Great app!", "Terrible bugs"]
            >>> preprocessor.preprocess_texts(texts, show_progress=False)
            ['great', 'terrible bug']
        """
        if show_progress:
            self.logger.info(f"Preprocessing {len(texts)} texts...")

        preprocessed = []
        for idx, text in enumerate(texts):
            processed = self.preprocess_text(text, remove_stopwords, lemmatize)
            preprocessed.append(processed)

            # Log progress every 100 texts
            if show_progress and (idx + 1) % 100 == 0:
                self.logger.info(f"Processed {idx + 1}/{len(texts)} texts")

        if show_progress:
            self.logger.info(f"✓ Completed preprocessing {len(texts)} texts")

        return preprocessed

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "review",
        output_column: str = "preprocessed_text",
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Preprocess text column in a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing text data
            text_column (str): Name of column with raw text. Defaults to 'review'.
            output_column (str): Name for preprocessed text column.
                Defaults to 'preprocessed_text'.
            remove_stopwords (bool): Whether to remove stopwords. Defaults to True.
            lemmatize (bool): Whether to lemmatize. Defaults to True.
            show_progress (bool): Whether to show progress. Defaults to True.

        Returns:
            pd.DataFrame: DataFrame with added preprocessed text column

        Raises:
            ValueError: If text_column doesn't exist

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> df = pd.DataFrame({'review': ["Great!", "Bad!"]})
            >>> df_processed = preprocessor.preprocess_dataframe(df, show_progress=False)
            >>> 'preprocessed_text' in df_processed.columns
            True
        """
        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        if show_progress:
            self.logger.info(f"Preprocessing {len(df)} texts from DataFrame...")

        # Get texts
        texts = df[text_column].tolist()

        # Preprocess
        preprocessed = self.preprocess_texts(
            texts, remove_stopwords, lemmatize, show_progress
        )

        # Add to DataFrame
        df_copy = df.copy()
        df_copy[output_column] = preprocessed

        if show_progress:
            self.logger.info(f"✓ Added '{output_column}' column to DataFrame")

        return df_copy

    def get_token_statistics(self, texts: List[str], top_n: int = 20) -> Dict[str, any]:
        """
        Calculate token statistics from preprocessed texts.

        Args:
            texts (List[str]): List of preprocessed texts
            top_n (int): Number of top tokens to return. Defaults to 20.

        Returns:
            Dict[str, any]: Dictionary with statistics including:
                - total_tokens: Total number of tokens
                - unique_tokens: Number of unique tokens
                - avg_tokens_per_text: Average tokens per text
                - top_tokens: List of (token, count) tuples

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> texts = ["good", "bad good", "good good good"]
            >>> stats = preprocessor.get_token_statistics(texts, top_n=2)
            >>> stats['top_tokens'][0][0]
            'good'
        """
        from collections import Counter

        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = text.split()
            all_tokens.extend(tokens)

        # Calculate statistics
        token_counts = Counter(all_tokens)

        stats = {
            "total_tokens": len(all_tokens),
            "unique_tokens": len(token_counts),
            "avg_tokens_per_text": len(all_tokens) / len(texts) if texts else 0,
            "top_tokens": token_counts.most_common(top_n),
        }

        return stats


def main():
    """
    Main function to demonstrate text preprocessing on reviews.

    Loads sentiment-analyzed reviews, preprocesses text, and saves results.
    """
    import os

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # File paths
    input_file = "data/processed/sentiment_analyzed_reviews.csv"
    output_file = "data/processed/preprocessed_reviews.csv"

    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    # Load reviews
    logger.info(f"Loading reviews from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} reviews")

    # Initialize preprocessor
    preprocessor = TextPreprocessor()

    # Preprocess reviews
    df_preprocessed = preprocessor.preprocess_dataframe(df)

    # Calculate statistics
    logger.info("\n=== Text Preprocessing Statistics ===")
    stats = preprocessor.get_token_statistics(
        df_preprocessed["preprocessed_text"].tolist(), top_n=30
    )

    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Unique tokens: {stats['unique_tokens']:,}")
    print(f"Average tokens per review: {stats['avg_tokens_per_text']:.2f}")
    print(f"\nTop 30 tokens:")
    for token, count in stats["top_tokens"]:
        print(f"  {token}: {count}")

    # Save results
    logger.info(f"\nSaving preprocessed reviews to {output_file}")
    df_preprocessed.to_csv(output_file, index=False)

    logger.info("✓ Text preprocessing complete!")


if __name__ == "__main__":
    main()

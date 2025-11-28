"""
Unit Tests for Theme Classification Module

Tests the ThemeClassifier class functionality including theme assignment,
statistics, and DataFrame operations.
"""

import unittest
import pandas as pd
import tempfile
import os
from core.theme_classifier import ThemeClassifier


class TestThemeClassifierInitialization(unittest.TestCase):
    """Tests for ThemeClassifier initialization."""

    def test_default_initialization(self):
        """Test classifier initializes with default themes."""
        classifier = ThemeClassifier()
        self.assertIsNotNone(classifier.theme_definitions)
        self.assertGreater(len(classifier.theme_definitions), 0)
        self.assertIn("User Experience", classifier.theme_definitions)

    def test_custom_themes_initialization(self):
        """Test classifier initializes with custom themes."""
        custom = {
            "Test Theme": {
                "description": "Test description",
                "keywords": ["test", "example"],
            }
        }
        classifier = ThemeClassifier(custom_themes=custom)
        self.assertEqual(len(classifier.theme_definitions), 1)
        self.assertIn("Test Theme", classifier.theme_definitions)

    def test_keyword_mapping_built(self):
        """Test keyword to theme mapping is built correctly."""
        classifier = ThemeClassifier()
        self.assertIsNotNone(classifier.keyword_to_themes)
        self.assertGreater(len(classifier.keyword_to_themes), 0)
        # Check that 'good' maps to User Experience
        self.assertIn("good", classifier.keyword_to_themes)
        self.assertIn("User Experience", classifier.keyword_to_themes["good"])

    def test_logger_setup(self):
        """Test logger is configured."""
        classifier = ThemeClassifier()
        self.assertIsNotNone(classifier.logger)
        self.assertEqual(classifier.logger.name, "core.theme_classifier")


class TestThemeDefinitions(unittest.TestCase):
    """Tests for theme definitions and structure."""

    def setUp(self):
        """Set up test classifier."""
        self.classifier = ThemeClassifier()

    def test_all_themes_have_keywords(self):
        """Test all themes have keyword lists."""
        for theme_name, theme_data in self.classifier.theme_definitions.items():
            self.assertIn("keywords", theme_data)
            self.assertIsInstance(theme_data["keywords"], list)
            self.assertGreater(len(theme_data["keywords"]), 0)

    def test_all_themes_have_descriptions(self):
        """Test all themes have descriptions."""
        for theme_name, theme_data in self.classifier.theme_definitions.items():
            self.assertIn("description", theme_data)
            self.assertIsInstance(theme_data["description"], str)
            self.assertGreater(len(theme_data["description"]), 0)

    def test_get_theme_definitions(self):
        """Test getting theme definitions."""
        definitions = self.classifier.get_theme_definitions()
        self.assertIsInstance(definitions, dict)
        self.assertEqual(len(definitions), len(self.classifier.theme_definitions))

        for theme, description in definitions.items():
            self.assertIsInstance(description, str)
            self.assertGreater(len(description), 0)

    def test_keywords_are_lowercase(self):
        """Test keyword mapping uses lowercase."""
        for keyword in self.classifier.keyword_to_themes.keys():
            self.assertEqual(keyword, keyword.lower())


class TestSingleReviewClassification(unittest.TestCase):
    """Tests for classifying individual reviews."""

    def setUp(self):
        """Set up test classifier."""
        self.classifier = ThemeClassifier()

    def test_classify_positive_review(self):
        """Test classifying positive review."""
        review = "This is a great app, very easy to use and fast!"
        themes = self.classifier.classify_review(review)
        self.assertIn("User Experience", themes)
        self.assertIn("Performance", themes)

    def test_classify_negative_review(self):
        """Test classifying negative review."""
        review = "Worst app ever, crashes all the time, terrible!"
        themes = self.classifier.classify_review(review)
        self.assertIn("Technical Issues", themes)
        self.assertIn("Negative Experience", themes)

    def test_classify_feature_review(self):
        """Test classifying feature-focused review."""
        review = "Good features, can transfer money and check balance easily"
        themes = self.classifier.classify_review(review)
        self.assertIn("Features & Functionality", themes)
        self.assertIn("User Experience", themes)

    def test_classify_performance_review(self):
        """Test classifying performance review."""
        review = "App is very slow, takes too long to load"
        themes = self.classifier.classify_review(review)
        self.assertIn("Performance", themes)

    def test_classify_update_review(self):
        """Test classifying update-related review."""
        review = "Latest update improved the app significantly, much better now"
        themes = self.classifier.classify_review(review)
        self.assertIn("Updates & Improvements", themes)

    def test_classify_security_review(self):
        """Test classifying security review."""
        review = "Login issues, password reset doesn't work, can't access account"
        themes = self.classifier.classify_review(review)
        self.assertIn("Authentication & Security", themes)

    def test_classify_support_review(self):
        """Test classifying customer support review."""
        review = "Thank you for the help, customer service was great"
        themes = self.classifier.classify_review(review)
        self.assertIn("Customer Support", themes)

    def test_classify_empty_review(self):
        """Test classifying empty review."""
        themes = self.classifier.classify_review("")
        self.assertEqual(themes, [])

    def test_classify_with_preprocessed_text(self):
        """Test classification with preprocessed text."""
        original = "The app is really slow!!!"
        preprocessed = "app really slow"
        themes = self.classifier.classify_review(original, preprocessed)
        self.assertIn("Performance", themes)

    def test_classify_no_matching_keywords(self):
        """Test review with no matching keywords."""
        review = "xyz abc def"
        themes = self.classifier.classify_review(review)
        self.assertEqual(themes, [])

    def test_classify_multiple_themes(self):
        """Test review matching multiple themes."""
        review = "Great update! The new features are fast and secure. Thank you!"
        themes = self.classifier.classify_review(review)
        self.assertGreater(len(themes), 2)  # Should match multiple themes

    def test_min_theme_confidence(self):
        """Test minimum theme confidence filtering."""
        review = "good"  # Only one keyword match
        themes_min1 = self.classifier.classify_review(review, min_theme_confidence=1)
        themes_min2 = self.classifier.classify_review(review, min_theme_confidence=2)

        self.assertGreater(len(themes_min1), 0)
        self.assertEqual(len(themes_min2), 0)  # Shouldn't meet min=2

    def test_themes_are_sorted(self):
        """Test that returned themes are sorted alphabetically."""
        review = "Great app with good features"
        themes = self.classifier.classify_review(review)
        self.assertEqual(themes, sorted(themes))


class TestBatchReviewClassification(unittest.TestCase):
    """Tests for classifying multiple reviews."""

    def setUp(self):
        """Set up test classifier and reviews."""
        self.classifier = ThemeClassifier()
        self.reviews = [
            "Great app!",
            "Terrible, crashes all the time",
            "Good features and fast",
            "",
        ]

    def test_classify_reviews_list(self):
        """Test classifying list of reviews."""
        results = self.classifier.classify_reviews(self.reviews, show_progress=False)
        self.assertEqual(len(results), len(self.reviews))
        self.assertIsInstance(results, list)

        for theme_list in results:
            self.assertIsInstance(theme_list, list)

    def test_classify_with_preprocessed_list(self):
        """Test classifying with preprocessed texts."""
        preprocessed = ["great app", "terrible crash time", "good feature fast", ""]
        results = self.classifier.classify_reviews(
            self.reviews, preprocessed_texts=preprocessed, show_progress=False
        )
        self.assertEqual(len(results), len(self.reviews))

    def test_empty_reviews_list(self):
        """Test classifying empty list."""
        results = self.classifier.classify_reviews([], show_progress=False)
        self.assertEqual(results, [])

    def test_show_progress_false(self):
        """Test classification without progress logging."""
        results = self.classifier.classify_reviews(self.reviews, show_progress=False)
        self.assertEqual(len(results), len(self.reviews))


class TestDataFrameClassification(unittest.TestCase):
    """Tests for classifying reviews in DataFrames."""

    def setUp(self):
        """Set up test classifier and DataFrame."""
        self.classifier = ThemeClassifier()
        self.df = pd.DataFrame(
            {
                "review": [
                    "Great app, very easy to use",
                    "Crashes frequently, worst app",
                    "Good features, but slow",
                    "Latest update is much better",
                ],
                "preprocessed_text": [
                    "great app easy use",
                    "crash frequently worst app",
                    "good feature slow",
                    "latest update much better",
                ],
                "bank": ["BOA", "CBE", "Dashen", "BOA"],
            }
        )

    def test_classify_dataframe_basic(self):
        """Test basic DataFrame classification."""
        result = self.classifier.classify_dataframe(self.df, show_progress=False)
        self.assertIn("themes", result.columns)
        self.assertEqual(len(result), len(self.df))

        for themes in result["themes"]:
            self.assertIsInstance(themes, list)

    def test_classify_with_custom_columns(self):
        """Test classification with custom column names."""
        result = self.classifier.classify_dataframe(
            self.df,
            review_column="review",
            preprocessed_column="preprocessed_text",
            output_column="my_themes",
            show_progress=False,
        )
        self.assertIn("my_themes", result.columns)

    def test_classify_without_preprocessed(self):
        """Test classification without preprocessed column."""
        df_no_prep = self.df.drop("preprocessed_text", axis=1)
        result = self.classifier.classify_dataframe(
            df_no_prep, preprocessed_column=None, show_progress=False
        )
        self.assertIn("themes", result.columns)

    def test_missing_review_column_raises_error(self):
        """Test error when review column is missing."""
        with self.assertRaises(ValueError) as context:
            self.classifier.classify_dataframe(
                self.df, review_column="nonexistent", show_progress=False
            )
        self.assertIn("not found", str(context.exception))

    def test_dataframe_not_modified(self):
        """Test original DataFrame is not modified."""
        original_columns = list(self.df.columns)
        self.classifier.classify_dataframe(self.df, show_progress=False)
        self.assertEqual(list(self.df.columns), original_columns)

    def test_empty_dataframe(self):
        """Test classifying empty DataFrame."""
        empty_df = pd.DataFrame(columns=["review", "preprocessed_text"])
        result = self.classifier.classify_dataframe(empty_df, show_progress=False)
        self.assertEqual(len(result), 0)
        self.assertIn("themes", result.columns)


class TestThemeStatistics(unittest.TestCase):
    """Tests for theme statistics calculation."""

    def setUp(self):
        """Set up test data."""
        self.classifier = ThemeClassifier()
        self.df = pd.DataFrame(
            {
                "themes": [
                    ["User Experience", "Performance"],
                    ["Technical Issues"],
                    ["User Experience", "Features & Functionality"],
                    [],
                ],
                "bank": ["BOA", "CBE", "BOA", "Dashen"],
            }
        )

    def test_get_overall_statistics(self):
        """Test overall statistics calculation."""
        stats = self.classifier.get_theme_statistics(self.df)

        self.assertIn("total_reviews", stats)
        self.assertIn("theme_counts", stats)
        self.assertIn("theme_percentages", stats)
        self.assertIn("avg_themes_per_review", stats)
        self.assertIn("reviews_without_themes", stats)
        self.assertIn("top_themes", stats)

        self.assertEqual(stats["total_reviews"], 4)
        self.assertEqual(stats["reviews_without_themes"], 1)

    def test_theme_counts_correct(self):
        """Test theme counts are calculated correctly."""
        stats = self.classifier.get_theme_statistics(self.df)

        counts = stats["theme_counts"]
        self.assertEqual(counts["User Experience"], 2)
        self.assertEqual(counts["Performance"], 1)
        self.assertEqual(counts["Technical Issues"], 1)

    def test_theme_percentages_correct(self):
        """Test theme percentages are correct."""
        stats = self.classifier.get_theme_statistics(self.df)

        percentages = stats["theme_percentages"]
        self.assertEqual(percentages["User Experience"], 50.0)  # 2/4 * 100
        self.assertEqual(percentages["Performance"], 25.0)  # 1/4 * 100

    def test_avg_themes_per_review(self):
        """Test average themes per review."""
        stats = self.classifier.get_theme_statistics(self.df)
        # (2 + 1 + 2 + 0) / 4 = 1.25
        self.assertAlmostEqual(stats["avg_themes_per_review"], 1.25)

    def test_top_themes_ordered(self):
        """Test top themes are ordered by count."""
        stats = self.classifier.get_theme_statistics(self.df)
        top_themes = stats["top_themes"]

        # Should be sorted by count descending
        counts = [count for _, count in top_themes]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_grouped_statistics(self):
        """Test statistics grouped by column."""
        stats = self.classifier.get_theme_statistics(self.df, group_by="bank")

        self.assertIn("BOA", stats)
        self.assertIn("CBE", stats)
        self.assertIn("Dashen", stats)

        self.assertEqual(stats["BOA"]["total_reviews"], 2)
        self.assertEqual(stats["CBE"]["total_reviews"], 1)

    def test_missing_theme_column_raises_error(self):
        """Test error when theme column is missing."""
        with self.assertRaises(ValueError) as context:
            self.classifier.get_theme_statistics(self.df, theme_column="nonexistent")
        self.assertIn("not found", str(context.exception))


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def setUp(self):
        """Set up test classifier."""
        self.classifier = ThemeClassifier()

    def test_none_review_text(self):
        """Test handling None review text."""
        themes = self.classifier.classify_review(None)
        self.assertEqual(themes, [])

    def test_very_long_review(self):
        """Test handling very long review."""
        review = " ".join(["good"] * 1000)
        themes = self.classifier.classify_review(review)
        self.assertIn("User Experience", themes)

    def test_special_characters(self):
        """Test handling special characters."""
        review = "!@#$%^&*()_+ great app []{}|"
        themes = self.classifier.classify_review(review)
        self.assertIn("User Experience", themes)

    def test_mixed_case_keywords(self):
        """Test case-insensitive keyword matching."""
        reviews = ["GREAT app", "Great APP", "great app"]
        for review in reviews:
            themes = self.classifier.classify_review(review)
            self.assertIn("User Experience", themes)

    def test_unicode_characters(self):
        """Test handling unicode characters."""
        review = "great app 👍 🎉"
        themes = self.classifier.classify_review(review)
        self.assertIn("User Experience", themes)

    def test_dataframe_with_nan_values(self):
        """Test DataFrame with NaN values."""
        df = pd.DataFrame(
            {
                "review": ["good app", None, "bad app", ""],
                "preprocessed_text": ["good app", "none", None, ""],
            }
        )
        result = self.classifier.classify_dataframe(df, show_progress=False)
        self.assertEqual(len(result), 4)
        self.assertIn("themes", result.columns)

    def test_statistics_with_no_themes(self):
        """Test statistics when no themes assigned."""
        df = pd.DataFrame({"themes": [[], [], []]})
        stats = self.classifier.get_theme_statistics(df)

        self.assertEqual(stats["total_reviews"], 3)
        self.assertEqual(stats["reviews_without_themes"], 3)
        self.assertEqual(stats["avg_themes_per_review"], 0.0)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow."""

    def setUp(self):
        """Set up test classifier."""
        self.classifier = ThemeClassifier()

    def test_full_pipeline(self):
        """Test complete classification pipeline."""
        # Create test data
        df = pd.DataFrame(
            {
                "review": [
                    "Great app with excellent features!",
                    "Crashes all the time, worst experience",
                    "Good update, much better performance now",
                    "Login doesn't work, password issues",
                    "Fast and easy to use, love it!",
                ],
                "preprocessed_text": [
                    "great app excellent feature",
                    "crash time worst experience",
                    "good update much better performance",
                    "login doesnt work password issue",
                    "fast easy use love",
                ],
                "bank": ["BOA", "CBE", "Dashen", "BOA", "CBE"],
            }
        )

        # Classify
        result = self.classifier.classify_dataframe(df, show_progress=False)

        # Verify structure
        self.assertIn("themes", result.columns)
        self.assertEqual(len(result), len(df))

        # Verify themes assigned
        has_themes = result["themes"].apply(lambda x: len(x) > 0)
        self.assertTrue(has_themes.all())

        # Get statistics
        stats = self.classifier.get_theme_statistics(result)
        self.assertGreater(stats["total_reviews"], 0)
        self.assertGreater(len(stats["theme_counts"]), 0)

        # Get grouped statistics
        bank_stats = self.classifier.get_theme_statistics(result, group_by="bank")
        self.assertEqual(len(bank_stats), 3)

    def test_custom_themes_full_pipeline(self):
        """Test full pipeline with custom themes."""
        custom_themes = {
            "Positive": {
                "description": "Positive sentiment",
                "keywords": ["good", "great", "excellent", "love"],
            },
            "Negative": {
                "description": "Negative sentiment",
                "keywords": ["bad", "worst", "terrible", "hate"],
            },
        }

        classifier = ThemeClassifier(custom_themes=custom_themes)

        df = pd.DataFrame(
            {
                "review": ["Good app!", "Worst app ever", "I love it"],
                "preprocessed_text": ["good app", "worst app ever", "love"],
            }
        )

        result = classifier.classify_dataframe(df, show_progress=False)

        # Check all reviews got classified
        for themes in result["themes"]:
            self.assertGreater(len(themes), 0)

        # Verify only custom themes used
        all_themes = set()
        for themes_list in result["themes"]:
            all_themes.update(themes_list)

        self.assertTrue(all_themes.issubset({"Positive", "Negative"}))

    def test_save_and_analyze_results(self):
        """Test saving classified data and analyzing."""
        df = pd.DataFrame(
            {
                "review": ["Great!", "Bad!", "Good features"],
                "preprocessed_text": ["great", "bad", "good feature"],
            }
        )

        # Classify
        result = self.classifier.classify_dataframe(df, show_progress=False)

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_file = f.name
            result.to_csv(temp_file, index=False)

        try:
            # Load and verify
            loaded = pd.read_csv(temp_file)
            self.assertIn("themes", loaded.columns)
            self.assertEqual(len(loaded), len(df))
        finally:
            os.unlink(temp_file)


if __name__ == "__main__":
    unittest.main()

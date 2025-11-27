"""
Unit tests for the Config module.

This module tests the configuration settings, path management, and helper methods
of the Config class.
"""

import pytest
from pathlib import Path
from core.config import Config


class TestConfig:
    """Test suite for the Config class."""

    def test_bank_app_ids_exist(self):
        """Test that all bank app IDs are defined."""
        assert "CBE" in Config.BANK_APP_IDS
        assert "BOA" in Config.BANK_APP_IDS
        assert "Dashen" in Config.BANK_APP_IDS

    def test_bank_app_ids_format(self):
        """Test that app IDs have correct format."""
        for bank, app_id in Config.BANK_APP_IDS.items():
            assert isinstance(app_id, str)
            assert len(app_id) > 0
            # Google Play package names typically follow this pattern
            assert "." in app_id

    def test_bank_names_exist(self):
        """Test that all bank names are defined."""
        assert "CBE" in Config.BANK_NAMES
        assert "BOA" in Config.BANK_NAMES
        assert "Dashen" in Config.BANK_NAMES

    def test_bank_names_match_ids(self):
        """Test that bank names keys match app IDs keys."""
        assert set(Config.BANK_NAMES.keys()) == set(Config.BANK_APP_IDS.keys())

    def test_scraping_params_structure(self):
        """Test that scraping parameters have correct structure."""
        required_keys = ["language", "country", "count", "sort_by"]
        for key in required_keys:
            assert key in Config.SCRAPING_PARAMS

    def test_scraping_params_values(self):
        """Test that scraping parameters have valid values."""
        params = Config.SCRAPING_PARAMS

        # Language should be 2-letter code
        assert len(params["language"]) == 2
        assert isinstance(params["language"], str)

        # Country should be 2-letter code
        assert len(params["country"]) == 2
        assert isinstance(params["country"], str)

        # Count should be positive integer
        assert isinstance(params["count"], int)
        assert params["count"] > 0

    def test_paths_are_path_objects(self):
        """Test that all directory paths are Path objects."""
        assert isinstance(Config.BASE_DIR, Path)
        assert isinstance(Config.DATA_DIR, Path)
        assert isinstance(Config.RAW_DATA_DIR, Path)
        assert isinstance(Config.PROCESSED_DATA_DIR, Path)
        assert isinstance(Config.CLEANED_DATA_DIR, Path)

    def test_base_dir_exists(self):
        """Test that base directory exists (should be project root)."""
        assert Config.BASE_DIR.exists()
        assert Config.BASE_DIR.is_dir()

    def test_data_quality_thresholds(self):
        """Test that data quality thresholds are reasonable."""
        assert Config.MIN_REVIEWS_PER_BANK >= 100
        assert Config.MAX_MISSING_DATA_PERCENT >= 0
        assert Config.MAX_MISSING_DATA_PERCENT <= 100

    def test_get_app_id_valid_bank(self):
        """Test getting app ID for valid bank codes."""
        cbe_id = Config.get_app_id("CBE")
        assert isinstance(cbe_id, str)
        assert cbe_id == Config.BANK_APP_IDS["CBE"]

        boa_id = Config.get_app_id("BOA")
        assert boa_id == Config.BANK_APP_IDS["BOA"]

        dashen_id = Config.get_app_id("Dashen")
        assert dashen_id == Config.BANK_APP_IDS["Dashen"]

    def test_get_app_id_invalid_bank(self):
        """Test that invalid bank code raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config.get_app_id("INVALID")

        assert "Unknown bank code" in str(exc_info.value)
        assert "INVALID" in str(exc_info.value)

    def test_get_bank_name_valid_bank(self):
        """Test getting bank name for valid bank codes."""
        cbe_name = Config.get_bank_name("CBE")
        assert isinstance(cbe_name, str)
        assert cbe_name == Config.BANK_NAMES["CBE"]
        assert "Commercial Bank of Ethiopia" in cbe_name

    def test_get_bank_name_invalid_bank(self):
        """Test that invalid bank code raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            Config.get_bank_name("INVALID")

        assert "Unknown bank code" in str(exc_info.value)

    def test_get_all_banks(self):
        """Test getting all bank codes."""
        banks = Config.get_all_banks()

        assert isinstance(banks, list)
        assert len(banks) == 3
        assert "CBE" in banks
        assert "BOA" in banks
        assert "Dashen" in banks

    def test_ensure_directories_creates_dirs(self, tmp_path):
        """Test that ensure_directories creates all required directories."""
        # This test would need to mock the directories or use a temp path
        # For now, we just test that the method doesn't raise errors
        Config.ensure_directories()

        # Verify at least some directories exist
        assert Config.RAW_DATA_DIR.exists()
        assert Config.PROCESSED_DATA_DIR.exists()
        assert Config.CLEANED_DATA_DIR.exists()

    def test_db_config_structure(self):
        """Test that database configuration has required keys."""
        required_keys = ["host", "port", "database", "user", "password"]
        for key in required_keys:
            assert key in Config.DB_CONFIG

    def test_db_config_default_values(self):
        """Test that database configuration has sensible defaults."""
        assert Config.DB_CONFIG["host"] in [
            "localhost",
            "127.0.0.1",
        ] or Config.DB_CONFIG["host"].startswith("DB_HOST")
        assert Config.DB_CONFIG["database"] == "bank_reviews"

    def test_log_level_valid(self):
        """Test that log level is a valid logging level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert Config.LOG_LEVEL in valid_levels

    def test_log_format_contains_essential_fields(self):
        """Test that log format contains essential fields."""
        assert "%(asctime)s" in Config.LOG_FORMAT
        assert "%(name)s" in Config.LOG_FORMAT
        assert "%(levelname)s" in Config.LOG_FORMAT
        assert "%(message)s" in Config.LOG_FORMAT


class TestConfigInstance:
    """Test suite for the config singleton instance."""

    def test_config_instance_exists(self):
        """Test that the config singleton instance is created."""
        from core.config import config

        assert config is not None

    def test_config_instance_is_config_class(self):
        """Test that config instance is of type Config."""
        from core.config import config

        assert isinstance(config, Config)

    def test_config_instance_has_all_banks(self):
        """Test that config instance can access all banks."""
        from core.config import config

        banks = config.get_all_banks()
        assert len(banks) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

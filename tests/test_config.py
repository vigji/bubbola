"""Tests for the configuration management system."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from bubbola.config import ConfigManager, get_config, get_env


class TestConfigManager:
    """Test the ConfigManager class."""

    def test_get_with_default(self):
        """Test getting a configuration value with default."""
        with patch.object(ConfigManager, "_get_config_paths") as mock_paths:
            mock_paths.return_value = []  # No config files found

            config = ConfigManager()

            # Test with existing environment variable
            with patch.dict(os.environ, {"TEST_KEY": "test_value"}):
                value = config.get("TEST_KEY", "default")
                assert value == "test_value"

            # Test with missing environment variable
            with patch.dict(os.environ, {}, clear=True):
                value = config.get("MISSING_KEY", "default")
                assert value == "default"

    def test_get_required_success(self):
        """Test getting a required configuration value successfully."""
        with patch.object(ConfigManager, "_get_config_paths") as mock_paths:
            mock_paths.return_value = []  # No config files found

            config = ConfigManager()

            with patch.dict(os.environ, {"REQUIRED_KEY": "required_value"}):
                value = config.get_required("REQUIRED_KEY")
                assert value == "required_value"

    def test_get_required_missing(self):
        """Test getting a required configuration value that's missing."""
        with patch.object(ConfigManager, "_get_config_paths") as mock_paths:
            mock_paths.return_value = []  # No config files found

            config = ConfigManager()

            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(
                    ValueError,
                    match="Required configuration key 'MISSING_KEY' not found",
                ):
                    config.get_required("MISSING_KEY")

    def test_get_aws_credentials(self):
        """Test getting AWS credentials."""
        with patch.object(ConfigManager, "_get_config_paths") as mock_paths:
            mock_paths.return_value = []  # No config files found

            config = ConfigManager()

            with patch.dict(
                os.environ,
                {
                    "AWS_ACCESS_KEY_ID": "test_access_key",
                    "AWS_SECRET_ACCESS_KEY": "test_secret_key",
                },
            ):
                access_key, secret_key = config.get_aws_credentials()
                assert access_key == "test_access_key"
                assert secret_key == "test_secret_key"

    def test_get_openai_key(self):
        """Test getting OpenAI API key."""
        with patch.object(ConfigManager, "_get_config_paths") as mock_paths:
            mock_paths.return_value = []  # No config files found

            config = ConfigManager()

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test_openai_key"}):
                key = config.get_openai_key()
                assert key == "test_openai_key"

    def test_get_deepinfra_token(self):
        """Test getting DeepInfra token."""
        with patch.object(ConfigManager, "_get_config_paths") as mock_paths:
            mock_paths.return_value = []  # No config files found

            config = ConfigManager()

            with patch.dict(os.environ, {"DEEPINFRA_TOKEN": "test_deepinfra_token"}):
                token = config.get_deepinfra_token()
                assert token == "test_deepinfra_token"

    def test_get_openrouter_key(self):
        """Test getting OpenRouter API key."""
        with patch.object(ConfigManager, "_get_config_paths") as mock_paths:
            mock_paths.return_value = []  # No config files found

            config = ConfigManager()

            with patch.dict(os.environ, {"OPENROUTER": "test_openrouter_key"}):
                key = config.get_openrouter_key()
                assert key == "test_openrouter_key"

    def test_validate_config_success(self):
        """Test configuration validation with all keys present."""
        with patch.object(ConfigManager, "_get_config_paths") as mock_paths:
            mock_paths.return_value = []  # No config files found

            config = ConfigManager()

            with patch.dict(
                os.environ,
                {
                    "AWS_ACCESS_KEY_ID": "test_access_key",
                    "AWS_SECRET_ACCESS_KEY": "test_secret_key",
                    "OPENAI_API_KEY": "test_openai_key",
                    "DEEPINFRA_TOKEN": "test_deepinfra_token",
                    "OPENROUTER": "test_openrouter_key",
                },
            ):
                assert config.validate_config() is True

    def test_validate_config_missing_keys(self):
        """Test configuration validation with missing keys."""
        with patch.object(ConfigManager, "_get_config_paths") as mock_paths:
            mock_paths.return_value = []  # No config files found

            config = ConfigManager()

            with patch.dict(
                os.environ,
                {
                    "AWS_ACCESS_KEY_ID": "test_access_key",
                    # Missing other keys
                },
            ):
                assert config.validate_config() is False

    def test_load_config_from_file(self, tmp_path):
        """Test loading configuration from a file."""
        # Create a temporary config file
        config_file = tmp_path / "config.env"
        config_content = """AWS_ACCESS_KEY_ID=file_access_key
AWS_SECRET_ACCESS_KEY=file_secret_key
OPENAI_API_KEY=file_openai_key
DEEPINFRA_TOKEN=file_deepinfra_token
OPENROUTER=file_openrouter_key
"""
        config_file.write_text(config_content)

        # Mock the config paths to include our test file
        with patch.object(ConfigManager, "_get_config_paths") as mock_paths:
            mock_paths.return_value = [config_file]

            # Clear environment variables to ensure file values are used
            with patch.dict(os.environ, {}, clear=True):
                config = ConfigManager()

                # Test that values are loaded from file
                assert config.get("AWS_ACCESS_KEY_ID") == "file_access_key"
                assert config.get("OPENAI_API_KEY") == "file_openai_key"

    def test_config_priority(self, tmp_path):
        """Test that environment variables take priority over config file."""
        # Create a temporary config file
        config_file = tmp_path / "config.env"
        config_content = """AWS_ACCESS_KEY_ID=file_access_key
OPENAI_API_KEY=file_openai_key
"""
        config_file.write_text(config_content)

        # Mock the config paths to include our test file
        with patch.object(ConfigManager, "_get_config_paths") as mock_paths:
            mock_paths.return_value = [config_file]

            # Set environment variable with different value
            with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "env_access_key"}):
                config = ConfigManager()

                # Environment variable should take priority
                assert config.get("AWS_ACCESS_KEY_ID") == "env_access_key"
                # File value should be used for keys not in environment
                assert config.get("OPENAI_API_KEY") == "file_openai_key"


class TestConvenienceFunctions:
    """Test the convenience functions."""

    def test_get_env(self):
        """Test the get_env convenience function."""
        with patch.object(ConfigManager, "_get_config_paths") as mock_paths:
            mock_paths.return_value = []  # No config files found

            with patch.dict(os.environ, {"TEST_KEY": "test_value"}):
                value = get_env("TEST_KEY", "default")
                assert value == "test_value"

            with patch.dict(os.environ, {}, clear=True):
                value = get_env("MISSING_KEY", "default")
                assert value == "default"

    def test_get_config(self):
        """Test the get_config convenience function."""
        config = get_config()
        assert isinstance(config, ConfigManager)


class TestConfigPaths:
    """Test configuration path detection."""

    def test_find_project_root(self, tmp_path):
        """Test finding project root by looking for pyproject.toml."""
        # Create a mock project structure
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / "pyproject.toml").write_text("")

        subdir = project_root / "src" / "bubbola"
        subdir.mkdir(parents=True)

        config = ConfigManager()

        # Change to subdirectory and test finding project root
        with patch.object(Path, "cwd", return_value=subdir):
            found_root = config._find_project_root()
            assert found_root == project_root

    def test_find_project_root_not_found(self, tmp_path):
        """Test finding project root when pyproject.toml is not found."""
        # Create a directory without pyproject.toml
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        config = ConfigManager()

        with patch.object(Path, "cwd", return_value=test_dir):
            found_root = config._find_project_root()
            assert found_root is None


if __name__ == "__main__":
    pytest.main([__file__])

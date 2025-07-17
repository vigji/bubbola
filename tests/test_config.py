"""Tests for the configuration management system."""

import os
from unittest.mock import patch

import pytest

from bubbola.config import get_env, get_required


class TestConfigFunctions:
    """Test the configuration functions."""

    def test_get_env_with_default(self):
        """Test getting environment variable with default."""
        # Reset the loaded flag to ensure fresh test
        if hasattr(get_env, "_loaded"):
            delattr(get_env, "_loaded")

        with patch.dict(os.environ, {"TEST_KEY": "test_value"}):
            value = get_env("TEST_KEY", "default")
            assert value == "test_value"

        with patch.dict(os.environ, {}, clear=True):
            value = get_env("MISSING_KEY", "default")
            assert value == "default"

    def test_get_required_success(self):
        """Test getting a required environment variable successfully."""
        # Reset the loaded flag
        if hasattr(get_env, "_loaded"):
            delattr(get_env, "_loaded")

        with patch.dict(os.environ, {"REQUIRED_KEY": "required_value"}):
            value = get_required("REQUIRED_KEY")
            assert value == "required_value"

    def test_get_required_missing(self):
        """Test getting a required environment variable that's missing."""
        # Reset the loaded flag
        if hasattr(get_env, "_loaded"):
            delattr(get_env, "_loaded")

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError,
                match="Required environment variable 'MISSING_KEY' not found or not configured",
            ):
                get_required("MISSING_KEY")

    def test_get_required_placeholder(self):
        """Test getting a required environment variable that has placeholder value."""
        # Reset the loaded flag
        if hasattr(get_env, "_loaded"):
            delattr(get_env, "_loaded")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "your_openai_api_key_here"}):
            with pytest.raises(
                ValueError,
                match="Required environment variable 'OPENAI_API_KEY' not found or not configured",
            ):
                get_required("OPENAI_API_KEY")

    def test_environment_variable_priority(self):
        """Test that environment variables take priority over config file."""
        # Reset the loaded flag
        if hasattr(get_env, "_loaded"):
            delattr(get_env, "_loaded")

        # Set environment variable
        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "env_key"}):
            value = get_env("AWS_ACCESS_KEY_ID")
            # Environment variable should be usedgit add *
            assert value == "env_key"

    def test_config_loading_works(self):
        """Test that config loading doesn't crash."""
        # Reset the loaded flag
        if hasattr(get_env, "_loaded"):
            delattr(get_env, "_loaded")

        # This should not crash
        value = get_env("SOME_RANDOM_KEY", "default_value")
        assert value == "default_value"


if __name__ == "__main__":
    pytest.main([__file__])

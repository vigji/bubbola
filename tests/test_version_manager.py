"""Tests for the version manager script."""

# Import the functions from the version manager script
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.append(str(Path(__file__).parent.parent / "scripts"))
from version_manager import (
    bump_version,
    get_current_version,
    update_init_file,
    update_pyproject_toml,
    validate_version,
)


class TestVersionManager:
    """Test cases for version manager functions."""

    def test_validate_version(self):
        """Test version validation."""
        # Valid versions
        assert validate_version("1.0.0") is True
        assert validate_version("10.20.30") is True
        assert validate_version("0.0.1") is True

        # Invalid versions
        assert validate_version("1.0") is False
        assert validate_version("1.0.0.0") is False
        assert validate_version("v1.0.0") is False
        assert validate_version("1.0.0-alpha") is False
        assert validate_version("") is False

    def test_bump_version(self):
        """Test version bumping logic."""
        # Patch bumps
        assert bump_version("1.0.0", "patch") == "1.0.1"
        assert bump_version("1.0.9", "patch") == "1.0.10"

        # Minor bumps
        assert bump_version("1.0.0", "minor") == "1.1.0"
        assert bump_version("1.5.9", "minor") == "1.6.0"

        # Major bumps
        assert bump_version("1.0.0", "major") == "2.0.0"
        assert bump_version("5.9.8", "major") == "6.0.0"

        # Invalid bump type
        with pytest.raises(ValueError):
            bump_version("1.0.0", "invalid")

    def test_get_current_version(self):
        """Test getting current version from __init__.py"""
        # Create a temporary init file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="__init__.py", delete=False
        ) as f:
            f.write(
                '"""Test module."""\n\n__version__ = "1.2.3"\n__author__ = "Test"\n'
            )
            temp_path = f.name

        try:
            # Mock the path to point to our temp file
            with patch("version_manager.Path") as mock_path:
                mock_init = mock_path.return_value
                mock_init.exists.return_value = True
                mock_init.read_text.return_value = Path(temp_path).read_text()

                version = get_current_version()
                assert version == "1.2.3"
        finally:
            Path(temp_path).unlink()

    def test_update_init_file(self):
        """Test updating version in __init__.py"""
        # Create a temporary init file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="__init__.py", delete=False
        ) as f:
            original_content = (
                '"""Test module."""\n\n__version__ = "1.0.0"\n__author__ = "Test"\n'
            )
            f.write(original_content)
            temp_path = f.name

        try:
            # Mock the path to point to our temp file
            with patch("version_manager.Path") as mock_path:
                mock_init = mock_path.return_value
                mock_init.read_text.return_value = Path(temp_path).read_text()

                # Mock write_text to capture what would be written
                written_content = None

                def mock_write(content):
                    nonlocal written_content
                    written_content = content

                mock_init.write_text = mock_write

                update_init_file("2.0.0")

                assert '__version__ = "2.0.0"' in written_content
                assert '__author__ = "Test"' in written_content
        finally:
            Path(temp_path).unlink()

    def test_update_pyproject_toml(self):
        """Test updating version in pyproject.toml"""
        # Create a temporary pyproject.toml file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            original_content = """[project]
name = "test"
version = "1.0.0"
description = "Test project"
"""
            f.write(original_content)
            temp_path = f.name

        try:
            # Mock the path to point to our temp file
            with patch("version_manager.Path") as mock_path:
                mock_toml = mock_path.return_value
                mock_toml.read_text.return_value = Path(temp_path).read_text()

                # Mock write_text to capture what would be written
                written_content = None

                def mock_write(content):
                    nonlocal written_content
                    written_content = content

                mock_toml.write_text = mock_write

                update_pyproject_toml("2.0.0")

                assert 'version = "2.0.0"' in written_content
                assert 'name = "test"' in written_content
        finally:
            Path(temp_path).unlink()

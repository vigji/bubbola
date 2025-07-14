"""Tests for the core application logic."""

from pathlib import Path

from bubbola.core import BubbolaApp


class TestBubbolaApp:
    """Test cases for BubbolaApp."""

    def test_app_initialization(self, tmp_path: Path) -> None:
        """Test that the app initializes correctly."""
        app = BubbolaApp()
        assert app.config_path.parent.exists()

    def test_run_with_no_args(self) -> None:
        """Test running the app with no arguments."""
        app = BubbolaApp()
        result = app.run([])
        assert result == 0

    def test_run_version_command(self) -> None:
        """Test the version command."""
        app = BubbolaApp()
        result = app.run(["version"])
        assert result == 0

    def test_run_help_command(self) -> None:
        """Test the help command."""
        app = BubbolaApp()
        result = app.run(["help"])
        assert result == 0

    def test_run_unknown_command(self) -> None:
        """Test running with an unknown command."""
        app = BubbolaApp()
        result = app.run(["unknown"])
        assert result == 1

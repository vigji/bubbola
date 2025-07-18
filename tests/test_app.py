"""Tests for the BubbolaApp class."""

from pathlib import Path
from unittest.mock import Mock, patch

from bubbola.app import BubbolaApp


class TestBubbolaApp:
    """Test cases for BubbolaApp."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch("bubbola.app.load_config"):
            self.app = BubbolaApp()

    def test_init_creates_config_directory(self):
        """Test that __init__ creates the config directory."""
        with patch("bubbola.app.load_config"), patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = Path("/fake/home")
            with patch("pathlib.Path.mkdir") as mock_mkdir:
                BubbolaApp()
                mock_mkdir.assert_called_once_with(exist_ok=True)

    def test_init_fallback_to_current_directory(self):
        """Test that __init__ falls back to current directory when home fails."""
        with patch("bubbola.app.load_config"), patch("pathlib.Path.home") as mock_home:
            mock_home.side_effect = RuntimeError("Home directory not available")
            with (
                patch("pathlib.Path.cwd") as mock_cwd,
                patch("pathlib.Path.mkdir") as mock_mkdir,
            ):
                mock_cwd.return_value = Path("/fake/current")
                app = BubbolaApp()
                assert (
                    app.config_path
                    == Path("/fake/current") / ".bubbola" / "config.json"
                )
                mock_mkdir.assert_called_once_with(exist_ok=True)

    def test_run_no_args_shows_help(self, capsys):
        """Test that run with no args shows help message."""
        result = self.app.run([])
        captured = capsys.readouterr()
        assert "Bubbola - Un'applicazione Python" in captured.out
        assert "Utilizzo: bubbola <comando>" in captured.out
        assert result == 0

    def test_run_version_command(self, capsys):
        """Test the version command."""
        with patch("bubbola.__version__", "1.2.3"):
            result = self.app.run(["version"])
            captured = capsys.readouterr()
            assert "Bubbola version 1.2.3" in captured.out
            assert result == 0

    def test_run_help_command(self, capsys):
        """Test the help command."""
        result = self.app.run(["help"])
        captured = capsys.readouterr()
        assert "Comandi disponibili:" in captured.out
        assert "version" in captured.out
        assert "help" in captured.out
        assert "sanitize" in captured.out
        assert "list" in captured.out
        assert "extract" in captured.out
        assert result == 0

    def test_run_unknown_command(self, capsys):
        """Test that unknown commands show error message."""
        result = self.app.run(["unknown"])
        captured = capsys.readouterr()
        assert "Comando sconosciuto: unknown" in captured.out
        assert "Usa 'bubbola help'" in captured.out
        assert result == 1

    def test_sanitize_command_no_args(self, capsys):
        """Test sanitize command with no arguments."""
        result = self.app.run(["sanitize"])
        captured = capsys.readouterr()
        assert "Utilizzo: bubbola sanitize" in captured.out
        assert result == 1

    def test_sanitize_command_nonexistent_input(self, capsys):
        """Test sanitize command with nonexistent input path."""
        result = self.app.run(["sanitize", "nonexistent.pdf"])
        captured = capsys.readouterr()
        assert "Errore: Il percorso dell'input non esiste" in captured.out
        assert result == 1

    def test_sanitize_command_invalid_max_size(self, capsys):
        """Test sanitize command with invalid max-size argument."""
        with patch("pathlib.Path.exists", return_value=True):
            result = self.app.run(["sanitize", "test.pdf", "--max-size", "invalid"])
            captured = capsys.readouterr()
            assert "Errore: Valore max-size non valido" in captured.out
            assert result == 1

    def test_sanitize_command_negative_max_size(self, capsys):
        """Test sanitize command with negative max-size argument."""
        with patch("pathlib.Path.exists", return_value=True):
            result = self.app.run(["sanitize", "test.pdf", "--max-size", "-100"])
            captured = capsys.readouterr()
            assert "Errore: Valore max-size non valido" in captured.out
            assert result == 1

    def test_sanitize_command_unknown_argument(self, capsys):
        """Test sanitize command with unknown argument."""
        with patch("pathlib.Path.exists", return_value=True):
            result = self.app.run(["sanitize", "test.pdf", "--unknown"])
            captured = capsys.readouterr()
            assert "Errore: Argomento sconosciuto: --unknown" in captured.out
            assert result == 1

    @patch("bubbola.image_data_loader.save_sanitized_images")
    def test_sanitize_command_success(self, mock_save, capsys):
        """Test successful sanitize command."""
        mock_save.return_value = Path("/output/path")
        with patch("pathlib.Path.exists", return_value=True):
            result = self.app.run(["sanitize", "test.pdf"])
            captured = capsys.readouterr()
            assert "Immagini sanitizzate salvate in: /output/path" in captured.out
            assert result == 0
            mock_save.assert_called_once_with(Path("test.pdf"), None, None)

    @patch("bubbola.image_data_loader.save_sanitized_images")
    def test_sanitize_command_with_options(self, mock_save, capsys):
        """Test sanitize command with output and max-size options."""
        mock_save.return_value = Path("/output/path")
        with patch("pathlib.Path.exists", return_value=True):
            result = self.app.run(
                [
                    "sanitize",
                    "test.pdf",
                    "--output",
                    "/custom/output",
                    "--max-size",
                    "1500",
                ]
            )
            captured = capsys.readouterr()
            assert "Immagini sanitizzate salvate in: /output/path" in captured.out
            assert result == 0
            mock_save.assert_called_once_with(
                Path("test.pdf"), Path("/custom/output"), 1500
            )

    @patch("bubbola.image_data_loader.save_sanitized_images")
    def test_sanitize_command_exception(self, mock_save, capsys):
        """Test sanitize command when save_sanitized_images raises an exception."""
        mock_save.side_effect = Exception("Test error")
        with patch("pathlib.Path.exists", return_value=True):
            result = self.app.run(["sanitize", "test.pdf"])
            captured = capsys.readouterr()
            assert "Errore durante la sanitizzazione: Test error" in captured.out
            assert result == 1

    def test_list_command_with_args(self, capsys):
        """Test list command with arguments (should show usage)."""
        result = self.app.run(["list", "extra"])
        captured = capsys.readouterr()
        assert "Utilizzo: bubbola list" in captured.out
        assert result == 1

    @patch("bubbola.batch_processor.BatchProcessor")
    def test_list_command_success(self, mock_batch_processor, capsys):
        """Test successful list command."""
        mock_processor = Mock()
        mock_flow = Mock()
        mock_flow.name = "test_flow"
        mock_flow.description = "Test flow description"
        mock_flow.model_name = "test_model"
        mock_flow.data_model.__name__ = "TestModel"
        mock_flow.external_file_options = {"suppliers_csv": "Suppliers file"}
        mock_processor.list_flows.return_value = [mock_flow]
        mock_batch_processor.return_value = mock_processor

        result = self.app.run(["list"])
        captured = capsys.readouterr()
        assert "Flussi di estrazione disponibili:" in captured.out
        assert "test_flow: Test flow description" in captured.out
        assert "Modello: test_model" in captured.out
        assert "Modello dati: TestModel" in captured.out
        assert "--suppliers_csv: Suppliers file" in captured.out
        assert result == 0

    @patch("bubbola.batch_processor.BatchProcessor")
    def test_list_command_no_flows(self, mock_batch_processor, capsys):
        """Test list command when no flows are available."""
        mock_processor = Mock()
        mock_processor.list_flows.return_value = []
        mock_batch_processor.return_value = mock_processor

        result = self.app.run(["list"])
        captured = capsys.readouterr()
        assert "Nessun flusso trovato." in captured.out
        assert result == 0

    @patch("bubbola.batch_processor.BatchProcessor")
    def test_list_command_exception(self, mock_batch_processor, capsys):
        """Test list command when BatchProcessor raises an exception."""
        mock_batch_processor.side_effect = Exception("Test error")

        result = self.app.run(["list"])
        captured = capsys.readouterr()
        assert "Errore durante il caricamento dei flussi: Test error" in captured.out
        assert result == 1

    def test_extract_command_no_args(self, capsys):
        """Test extract command with no arguments."""
        result = self.app.run(["extract"])
        captured = capsys.readouterr()
        assert "Utilizzo: bubbola extract" in captured.out
        assert "--input" in captured.out
        assert "--flow" in captured.out
        assert result == 1

    def test_extract_command_missing_input(self, capsys):
        """Test extract command with missing --input argument."""
        with patch("bubbola.config.validate_config"):
            result = self.app.run(["extract", "--flow", "test_flow"])
            captured = capsys.readouterr()
            assert (
                "Errore: --input è obbligatorio per il comando extract" in captured.out
            )
            assert result == 1

    def test_extract_command_missing_flow(self, capsys):
        """Test extract command with missing --flow argument."""
        with patch("bubbola.config.validate_config"):
            result = self.app.run(["extract", "--input", "test_path"])
            captured = capsys.readouterr()
            assert (
                "Errore: --flow è obbligatorio per il comando extract" in captured.out
            )
            assert result == 1

    def test_extract_command_unknown_argument(self, capsys):
        """Test extract command with unknown argument."""
        with patch("bubbola.config.validate_config"):
            result = self.app.run(["extract", "--unknown", "value"])
            captured = capsys.readouterr()
            assert "Errore: Argomento sconosciuto: --unknown" in captured.out
            assert result == 1

    @patch("bubbola.batch_processor.BatchProcessor")
    @patch("bubbola.config.validate_config")
    def test_extract_command_dry_run_failure(
        self, mock_validate, mock_batch_processor, capsys
    ):
        """Test extract command when dry run fails."""
        mock_processor = Mock()
        mock_processor.process_batch.return_value = 1
        mock_batch_processor.return_value = mock_processor

        result = self.app.run(
            ["extract", "--input", "test_path", "--flow", "test_flow"]
        )
        captured = capsys.readouterr()
        assert "Errore durante la stima dei costi. Interruzione." in captured.out
        assert result == 1

    @patch("bubbola.batch_processor.BatchProcessor")
    @patch("bubbola.config.validate_config")
    def test_extract_command_with_yes_flag(
        self, mock_validate, mock_batch_processor, capsys
    ):
        """Test extract command with --yes flag."""
        mock_processor = Mock()
        mock_processor.process_batch.side_effect = [
            0,
            0,
        ]  # dry run success, real run success
        mock_batch_processor.return_value = mock_processor

        result = self.app.run(
            ["extract", "--input", "test_path", "--flow", "test_flow", "--yes"]
        )
        captured = capsys.readouterr()
        assert "=== STIMA COSTI ===" in captured.out
        assert "=== ELABORAZIONE REALE ===" in captured.out
        assert result == 0

    @patch("bubbola.batch_processor.BatchProcessor")
    @patch("bubbola.config.validate_config")
    def test_extract_command_with_external_files(
        self, mock_validate, mock_batch_processor, capsys
    ):
        """Test extract command with external files."""
        mock_processor = Mock()
        mock_processor.process_batch.side_effect = [0, 0]
        mock_batch_processor.return_value = mock_processor

        result = self.app.run(
            [
                "extract",
                "--input",
                "test_path",
                "--flow",
                "test_flow",
                "--suppliers-csv",
                "suppliers.csv",
                "--prices-csv",
                "prices.csv",
                "--yes",
            ]
        )
        assert result == 0

        # Check that external files were passed correctly
        calls = mock_processor.process_batch.call_args_list
        expected_external_files = {
            "suppliers_csv": Path("suppliers.csv"),
            "prices_csv": Path("prices.csv"),
        }
        for call in calls:
            assert call[1]["external_files"] == expected_external_files

    @patch("bubbola.batch_processor.BatchProcessor")
    @patch("bubbola.config.validate_config")
    @patch("builtins.input")
    def test_extract_command_user_confirmation_yes(
        self, mock_input, mock_validate, mock_batch_processor, capsys
    ):
        """Test extract command with user confirming."""
        mock_processor = Mock()
        mock_processor.process_batch.side_effect = [0, 0]
        mock_batch_processor.return_value = mock_processor
        mock_input.return_value = "y"

        result = self.app.run(
            ["extract", "--input", "test_path", "--flow", "test_flow"]
        )
        captured = capsys.readouterr()
        assert "=== STIMA COSTI ===" in captured.out
        assert "=== ELABORAZIONE REALE ===" in captured.out
        assert result == 0
        # Verify input was called
        mock_input.assert_called_once_with(
            "Procedere con l'elaborazione reale? (y/N): "
        )

    @patch("bubbola.batch_processor.BatchProcessor")
    @patch("bubbola.config.validate_config")
    @patch("builtins.input")
    def test_extract_command_user_confirmation_no(
        self, mock_input, mock_validate, mock_batch_processor, capsys
    ):
        """Test extract command with user declining."""
        mock_processor = Mock()
        mock_processor.process_batch.return_value = 0
        mock_batch_processor.return_value = mock_processor
        mock_input.return_value = "n"

        result = self.app.run(
            ["extract", "--input", "test_path", "--flow", "test_flow"]
        )
        captured = capsys.readouterr()
        assert "=== STIMA COSTI ===" in captured.out
        assert "Elaborazione annullata dall'utente." in captured.out
        assert result == 0
        # Verify input was called
        mock_input.assert_called_once_with(
            "Procedere con l'elaborazione reale? (y/N): "
        )

    @patch("bubbola.batch_processor.BatchProcessor")
    @patch("bubbola.config.validate_config")
    @patch("builtins.input")
    def test_extract_command_eof_error(
        self, mock_input, mock_validate, mock_batch_processor, capsys
    ):
        """Test extract command when input raises EOFError."""
        mock_processor = Mock()
        mock_processor.process_batch.return_value = 0
        mock_batch_processor.return_value = mock_processor
        mock_input.side_effect = EOFError()

        result = self.app.run(
            ["extract", "--input", "test_path", "--flow", "test_flow"]
        )
        captured = capsys.readouterr()
        assert (
            "Errore: Impossibile leggere l'input. Usa --yes per bypassare la conferma."
            in captured.out
        )
        assert result == 1

    @patch("bubbola.batch_processor.BatchProcessor")
    @patch("bubbola.config.validate_config")
    def test_extract_command_exception(
        self, mock_validate, mock_batch_processor, capsys
    ):
        """Test extract command when BatchProcessor raises an exception."""
        mock_batch_processor.side_effect = Exception("Test error")

        result = self.app.run(
            ["extract", "--input", "test_path", "--flow", "test_flow"]
        )
        captured = capsys.readouterr()
        assert "Errore durante l'elaborazione: Test error" in captured.out
        assert result == 1

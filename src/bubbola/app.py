"""Application logic for Bubbola."""

from pathlib import Path


class BubbolaApp:
    """Main application class for Bubbola."""

    def __init__(self) -> None:
        """Initialize the application."""
        self.config_path = Path.home() / ".bubbola" / "config.json"
        self.config_path.parent.mkdir(exist_ok=True)

    def run(self, argv: list[str]) -> int:
        """Run the application with given arguments."""
        if not argv:
            print("Bubbola - A Python application")
            print("Usage: bubbola <command> [options]")
            return 0

        command = argv[0]

        if command == "version":
            from bubbola import __version__

            print(f"Bubbola version {__version__}")
            return 0
        elif command == "help":
            print("Available commands:")
            print("  version   - Show version information")
            print("  help      - Show this help message")
            print("  sanitize  - Convert PDFs/images to sanitized single page images")
            return 0
        elif command == "sanitize":
            return self._handle_sanitize_command(argv[1:])
        else:
            print(f"Unknown command: {command}")
            print("Use 'bubbola help' for available commands.")
            return 1

    def _handle_sanitize_command(self, args: list[str]) -> int:
        """Handle the sanitize command."""
        if not args:
            print(
                "Usage: bubbola sanitize <input_path> [--output <destination>] [--max-size <pixels>]"
            )
            print(
                "  input_path: Path to PDF file, image file, or folder containing files"
            )
            print(
                "  --output: Destination folder for saved images (default: single_pages)"
            )
            print("  --max-size: Maximum edge size in pixels for resizing (optional)")
            return 1

        input_path = Path(args[0])
        if not input_path.exists():
            print(f"Error: Input path does not exist: {input_path}")
            return 1

        # Parse optional arguments
        destination = None
        max_edge_size = None

        i = 1
        while i < len(args):
            if args[i] == "--output" and i + 1 < len(args):
                destination = Path(args[i + 1])
                i += 2
            elif args[i] == "--max-size" and i + 1 < len(args):
                try:
                    max_edge_size = int(args[i + 1])
                    if max_edge_size <= 0:
                        raise ValueError("Max size must be positive")
                except ValueError as e:
                    print(f"Error: Invalid max-size value: {e}")
                    return 1
                i += 2
            else:
                print(f"Error: Unknown argument: {args[i]}")
                return 1

        try:
            from bubbola.image_data_loader import save_sanitized_images

            output_path = save_sanitized_images(input_path, destination, max_edge_size)
            print(f"Successfully sanitized images to: {output_path}")
            return 0
        except Exception as e:
            print(f"Error during sanitization: {e}")
            return 1

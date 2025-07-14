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
            from . import __version__
            print(f"Bubbola version {__version__}")
            return 0
        elif command == "help":
            print("Available commands:")
            print("  version - Show version information")
            print("  help    - Show this help message")
            return 0
        else:
            print(f"Unknown command: {command}")
            print("Use 'bubbola help' for available commands.")
            return 1 
"""Command-line interface for Bubbola."""

import sys
from pathlib import Path
from typing import Optional

from .core import BubbolaApp


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for the CLI application."""
    if argv is None:
        argv = sys.argv[1:]
    
    try:
        app = BubbolaApp()
        return app.run(argv)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 
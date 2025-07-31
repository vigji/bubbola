"""Bubbola - A Python application for PDF and image processing."""

import sys

__version__ = "0.1.3"
__author__ = "Luigi Petrucco"
__email__ = "luigi.petrucco@gmail.com"


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI application."""
    if argv is None:
        argv = sys.argv[1:]

    try:
        from bubbola.app import BubbolaApp

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

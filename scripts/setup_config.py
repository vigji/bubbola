#!/usr/bin/env python3
"""Simple setup script for Bubbola configuration."""

import getpass
import sys
from pathlib import Path


def create_config():
    """Create ~/.bubbola/config.env with user input."""
    # Try to get home directory, fallback to current directory if it fails
    try:
        home_dir = Path.home()
        config_dir = home_dir / ".bubbola"
    except (RuntimeError, OSError):
        # Fallback for CI environments where home directory cannot be determined
        config_dir = Path.cwd() / ".bubbola"

    config_file = config_dir / "config.env"

    config_dir.mkdir(exist_ok=True)

    print("Setting up Bubbola configuration...")
    print("Please provide your API keys:")
    print()

    openai_key = getpass.getpass("OpenAI API Key: ").strip()
    deepinfra_token = getpass.getpass("DeepInfra Token: ").strip()
    openrouter_key = getpass.getpass("OpenRouter API Key: ").strip()

    config_content = f"""# Bubbola Configuration
OPENAI_API_KEY={openai_key}
DEEPINFRA_TOKEN={deepinfra_token}
OPENROUTER={openrouter_key}
"""

    with open(config_file, "w", encoding="utf-8") as f:
        f.write(config_content)

    print(f"\nConfiguration saved to: {config_file}")


if __name__ == "__main__":
    try:
        create_config()
    except KeyboardInterrupt:
        print("\nSetup cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

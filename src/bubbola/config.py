"""Simple configuration management for Bubbola application."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Embedded template configuration (can be overridden by user config)
EMBEDDED_CONFIG = """# Bubbola Configuration
# This file contains your API keys for the Bubbola application.
#
# INSTRUCTIONS:
# 1. Replace the placeholder values below with your actual API keys
# 2. Save this file
# 3. Restart the application
#
# You can get these keys from:
# - OpenAI: https://platform.openai.com/api-keys
# - DeepInfra: https://deepinfra.com/ (API Tokens)
# - OpenRouter: https://openrouter.ai/ (API Keys)

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# DeepInfra Token
DEEPINFRA_TOKEN=your_deepinfra_token_here

# OpenRouter API Key
OPENROUTER=your_openrouter_api_key_here
"""


def create_config_from_env():
    """Create config file from environment variables if running in CI."""
    if not os.getenv("CI"):
        return  # Only run in CI environments

    # Try to get home directory, fallback to current directory if it fails
    try:
        home_dir = Path.home()
        config_path = home_dir / ".bubbola" / "config.env"
    except (RuntimeError, OSError):
        # Fallback for CI environments where home directory cannot be determined
        config_path = Path.cwd() / ".bubbola" / "config.env"

    config_path.parent.mkdir(exist_ok=True)

    config_content = []
    possible_keys = [
        "OPENAI_API_KEY",
        "DEEPINFRA_TOKEN",
        "OPENROUTER",
    ]

    for key in possible_keys:
        value = os.getenv(key)
        if value:
            config_content.append(f"{key}={value}")

    if config_content:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write("# Bubbola Configuration (auto-generated from environment)\n")
            f.write("\n".join(config_content))
        print(f"Created CI configuration from environment variables: {config_path}")


def load_config():
    """Load configuration with fallback to embedded template."""
    # In CI, try to create config from environment variables first
    create_config_from_env()

    # Try to get home directory, fallback to current directory if it fails
    try:
        home_dir = Path.home()
        config_path = home_dir / ".bubbola" / "config.env"
    except (RuntimeError, OSError):
        # Fallback for CI environments where home directory cannot be determined
        config_path = Path.cwd() / ".bubbola" / "config.env"

    if config_path.exists():
        load_dotenv(config_path)
        print(f"Loaded configuration from: {config_path}")
    else:
        # Create config directory and template file
        config_path.parent.mkdir(exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(EMBEDDED_CONFIG)
        print(f"Created configuration template at: {config_path}")
        print("Please edit this file with your actual API keys")


def get_env(key: str, default: str | None = None) -> str | None:
    """Get environment variable with fallback to config file.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Environment variable value or default
    """
    # First check if environment variable is set (prioritizes GitHub secrets)
    env_value = os.getenv(key)
    if env_value is not None:
        return env_value

    # Load config file only if environment variable is not set
    if not hasattr(get_env, "_loaded"):
        load_config()
        get_env._loaded = True

    return os.getenv(key, default)


def get_required(key: str) -> str:
    """Get a required environment variable.

    Args:
        key: Environment variable name

    Returns:
        Environment variable value

    Raises:
        ValueError: If the key is not found
    """
    value = get_env(key)
    if value is None or value == f"your_{key.lower()}_here":
        raise ValueError(
            f"Required environment variable '{key}' not found or not configured"
        )
    return value


def validate_config():
    """Ensure at least one API key is present and not a placeholder. Exit if not valid."""
    possible_keys = [
        "OPENAI_API_KEY",
        "DEEPINFRA_TOKEN",
        "OPENROUTER",
    ]

    # Check if at least one key is properly configured
    has_valid_key = False
    missing_or_placeholder = []

    for key in possible_keys:
        value = get_env(key)
        if value is None or value.strip() == "":
            missing_or_placeholder.append(key)
        elif value == f"your_{key.lower()}_here":
            missing_or_placeholder.append(key)
        else:
            has_valid_key = True

    if not has_valid_key:
        print("\n[ERROR] Bubbola configuration is not valid:")
        print(
            f"  All keys are missing or have placeholder values: {', '.join(missing_or_placeholder)}"
        )
        print(
            "\nPlease edit ~/.bubbola/config.env and provide at least one actual API key before running any command."
        )
        sys.exit(1)

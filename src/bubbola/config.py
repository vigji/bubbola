"""Simple configuration management for Bubbola application."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Embedded template configuration (can be overridden by user config)
EMBEDDED_CONFIG = """# Bubbola Configuration
# This file contains your API keys and secrets for the Bubbola application.
#
# INSTRUCTIONS:
# 1. Replace the placeholder values below with your actual API keys
# 2. Save this file
# 3. Restart the application
#
# You can get these keys from:
# - AWS: https://console.aws.amazon.com/iam/ (Access Keys)
# - OpenAI: https://platform.openai.com/api-keys
# - DeepInfra: https://deepinfra.com/ (API Tokens)
# - OpenRouter: https://openrouter.ai/ (API Keys)

# AWS Credentials
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here

# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# DeepInfra Token
DEEPINFRA_TOKEN=your_deepinfra_token_here

# OpenRouter API Key
OPENROUTER=your_openrouter_api_key_here
"""


def load_config():
    """Load configuration with fallback to embedded template."""
    config_path = Path.home() / ".bubbola" / "config.env"

    if config_path.exists():
        load_dotenv(config_path)
        print(f"Loaded configuration from: {config_path}")
    else:
        # Create config directory and template file
        config_path.parent.mkdir(exist_ok=True)

        # Check if we're running as a binary
        if getattr(sys, "frozen", False):
            # Running as binary - create template for user to edit
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(EMBEDDED_CONFIG)
            print(f"Created configuration template at: {config_path}")
            print("Please edit this file with your actual API keys")
        else:
            # Running in development - just warn
            print(f"Configuration file not found: {config_path}")
            print("Please create ~/.bubbola/config.env with your API keys")


def get_env(key: str, default: str | None = None) -> str | None:
    """Get environment variable with fallback to config file.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Environment variable value or default
    """
    # Load config if not already loaded
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

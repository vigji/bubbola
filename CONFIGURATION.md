# Bubbola Configuration

Simple configuration setup for Bubbola.

## Setup

### Development
Create your configuration file:

```bash
python scripts/setup_config.py
```

This will create `~/.bubbola/config.env` with your API keys.

### Binary Distribution
When you run the binary for the first time, it automatically creates a configuration template at `~/.bubbola/config.env`. Just edit this file with your actual API keys.

## Manual Setup

Create `~/.bubbola/config.env` manually:

```env
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
OPENAI_API_KEY=your_openai_api_key
DEEPINFRA_TOKEN=your_deepinfra_token
OPENROUTER=your_openrouter_api_key
```

## How It Works

- **Development**: Uses `~/.bubbola/config.env` if it exists
- **Binary**: Automatically creates a template config file on first run
- **Fallback**: Template contains placeholder values that must be replaced
- **Cross-platform**: Works on Windows, macOS, and Linux

## Usage

The application automatically loads configuration:

```python
from bubbola.config import get_env, get_required

# Get optional value with default
api_key = get_env("OPENAI_API_KEY", "default_key")

# Get required value (raises ValueError if missing or not configured)
api_key = get_required("OPENAI_API_KEY")
```

## Security

- Keep your `~/.bubbola/config.env` file secure
- Don't commit it to version control
- Set file permissions to 600 on Unix systems (or restrict access on Windows)
- The binary includes a template, not actual secrets 
# Bubbola

A Python application with executable distribution using PyInstaller.

## Project Structure

```
bubbola/
├── src/
│   └── bubbola/           # Main package
│       ├── __init__.py    # Package initialization
│       ├── cli.py         # Command-line interface
│       └── core.py        # Core application logic
├── tests/                 # Test suite
│   ├── __init__.py
│   └── test_core.py
├── build.spec             # PyInstaller configuration
├── pyproject.toml         # Project configuration
├── Makefile              # Development tasks
└── README.md             # This file
```

## Development Setup

**Requirements:** Python 3.12

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd bubbola
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   uv sync
   ```
   This will automatically create a new virtual environment with Python 3.12+ and install all dependencies (including development dependencies).

**Important:** UV automatically creates and manages virtual environments. If you're currently in an existing environment (like conda), UV will create a separate environment for this project. You don't need to manually create or activate virtual environments - UV handles this automatically.


## Development Tools

### UV (Ultrafast Python Package Installer)

This project uses UV for dependency management and virtual environment handling:

- **Sync dependencies:** `uv sync`
- **Install in development mode:** `uv pip install -e .`
- **Run with UV:** `uv run python -m bubbola.cli`

### Ruff (Fast Python Linter)

This project uses Ruff for linting and formatting:

- **Format code:** `ruff format src/ tests/`
- **Lint code:** `ruff check src/ tests/`
- **Auto-fix issues:** `ruff check --fix src/ tests/`

## Development Workflow

1. **Start development:**
   ```bash
   uv sync
   ```

2. **Make changes to the code**

3. **Test your changes:**
   ```bash
   make test
   make lint
   ```

4. **Format and lint code:**
   ```bash
   make format
   ```

5. **Build and test executable:**
   ```bash
   make build
   make run-exe
   ```

## Configuration

The application stores configuration in `~/.bubbola/config.json`. The directory is created automatically on first run.

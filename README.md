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
├── uv.toml               # UV configuration
├── Makefile              # Development tasks
└── README.md             # This file
```

## Features

- Modern Python project structure with `src/` layout
- PyInstaller integration for executable distribution
- UV for fast dependency management and virtual environments
- Ruff for fast linting and formatting
- Type hints and static analysis with mypy
- Cross-platform compatibility

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd bubbola
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   uv sync
   ```

3. **Install development dependencies:**
   ```bash
   make install-dev
   # or manually:
   # uv pip install -e ".[dev]"
   ```

## Usage

### Development

- **Run the application:**
  ```bash
  make run
  # or
  uv run python -m bubbola.cli
  ```

- **Run tests:**
  ```bash
  make test
  ```

- **Format and lint code:**
  ```bash
  make format
  ```

- **Lint code only:**
  ```bash
  make lint
  ```

- **Clean build artifacts:**
  ```bash
  make clean
  ```

### Building Executables

1. **Build the executable:**
   ```bash
   make build
   ```

2. **Run the built executable:**
   ```bash
   make run-exe
   ```

The executable will be created in the `dist/bubbola/` directory.

## Available Commands

- `bubbola version` - Show version information
- `bubbola help` - Show help message

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
   make install-dev
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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting: `make test && make lint`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

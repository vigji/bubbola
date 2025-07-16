# Bubbola

A Python application for PDF and image processing with executable distribution using PyInstaller.

## Project Structure

```
bubbola/
├── src/
│   └── bubbola/           # Main package
│       ├── __init__.py    # Package initialization
│       ├── cli.py         # Command-line interface
│       ├── core.py        # Core application logic
│       ├── image_data_loader.py  # Image processing utilities
│       ├── load_results.py       # Results loading utilities
│       ├── model_creator.py      # Model creation utilities
│       └── models.py             # Data models
├── scripts/
│   └── build_binary.py    # Binary build script
├── tests/                 # Test suite
├── .github/workflows/     # GitHub Actions workflows
│   └── build-binaries.yml # Cross-platform build workflow
├── bubbola.spec           # PyInstaller configuration
├── pyproject.toml         # Project configuration
├── Makefile              # Development tasks
└── README.md             # This file
```

## Development Setup

**Requirements:** Python 3.11

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd bubbola
   ```

2. **Install dependencies:**
   
   **Option A: Using pip (recommended for binary builds)**
   ```bash
   pip install -e ".[dev,build]"
   ```
   
   **Option B: Using UV**
   ```bash
   uv sync
   uv pip install -e ".[build]"
   ```
   
   This will install the package in development mode along with all development and build dependencies.

**Important:** For building binaries, we recommend using pip as it provides better compatibility with PyInstaller. UV is great for development but may have issues with some binary packaging scenarios.


## Building Binaries

Bubbola can be compiled into standalone executables for multiple platforms using PyInstaller. The build system uses a **single, unified approach** that works both locally and in GitHub Actions.

**Note:** We use `make build` for local development and the same build script is used automatically in GitHub Actions.

### Local Binary Build

**Prerequisites:**
- Python 3.11
- Build dependencies installed: `pip install -e ".[build]"`

**Recommended Build Method:**
```bash
# Build for current platform
make build

# Clean build (removes previous artifacts)
make build-clean
```

**Direct Build Script:**
```bash
# Using the build script directly
python scripts/build_binary.py

# With custom output directory
python scripts/build_binary.py --output-dir ./my-binaries

# Clean build
python scripts/build_binary.py --clean
```



### Supported Platforms

The build system supports the following platforms:

- **macOS**: x86_64 and ARM64 (Apple Silicon)
- **Windows**: x64 (Windows 10/11)
- **Linux**: x86_64 (Ubuntu 22.04+)

### Binary Output

Built binaries are placed in the `dist/` directory:
- **macOS**: `dist/bubbola` (no extension)
- **Windows**: `dist/bubbola.exe`
- **Linux**: `dist/bubbola` (no extension)

### Testing Built Binaries

```bash
# Test the built binary
./dist/bubbola --help

# Test specific functionality
./dist/bubbola version
./dist/bubbola sanitize tests/assets/0088_001.pdf
```

### Troubleshooting Build Issues

**Common Issues:**

1. **Missing dependencies**: Ensure all build dependencies are installed
   ```bash
   pip install -e ".[build]"
   ```

2. **Large binary size**: This is normal for Python applications with dependencies like PyMuPDF and Pillow

3. **Platform-specific issues**:
   - **macOS**: May require code signing for distribution
   - **Windows**: May trigger antivirus warnings (false positive)
   - **Linux**: May require additional system libraries

4. **Import errors**: Check that all modules are included in `bubbola.spec`

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

#### Git Hooks for Ruff

To automatically run ruff checks before commits, you can set up a pre-commit hook:

1. **Install pre-commit:**
   ```bash
   uv add --dev pre-commit
   ```

2. **Create `.pre-commit-config.yaml`:**
   ```yaml
   repos:
     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: v0.12.3
       hooks:
         - id: ruff
           args: [--fix]
         - id: ruff-format
   ```

3. **Install the git hooks:**
   ```bash
   uv run pre-commit install
   ```

## Automated Builds (GitHub Actions)

The project includes automated cross-platform builds via GitHub Actions.

### Triggering Builds

**Automatic builds on:**
- Push to `main` branch (for testing)
- Tag push (e.g., `v1.0.0`) - creates GitHub release
- Manual trigger via GitHub Actions UI

### Build Matrix

The workflow builds for:
- **macOS 15**: x64 and ARM64 architectures
- **Windows 2022**: x64 architecture  
- **Ubuntu 22.04**: x64 architecture

### Creating Releases

To create a new release with binaries:

1. **Create and push a tag:**
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. **GitHub Actions will automatically:**
   - Build binaries for all platforms
   - Create a GitHub release
   - Attach all binaries to the release

### Accessing Built Binaries

- **From GitHub Releases**: Download directly from the releases page
- **From Actions Artifacts**: Available in the Actions tab for 30 days
- **From Release Assets**: Named as `bubbola-{platform}-{arch}`

## Development Workflow

1. **Start development:**
   ```bash
   pip install -e ".[dev,build]"
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
   ./dist/bubbola --help
   ```

## Makefile Targets

The project includes a comprehensive Makefile for common development tasks:

```bash
# Show all available targets
make help

# Installation
make install          # Install in development mode
make install-dev      # Install with dev dependencies

# Building
make build            # Build binary for current platform
make build-clean      # Clean and build binary

# Development
make test             # Run tests
make lint             # Run linting checks
make format           # Format code
make check            # Run all checks (lint + test)

# Cleaning
make clean            # Clean build artifacts
```

## Configuration

The application stores configuration in `~/.bubbola/config.json`. The directory is created automatically on first run.

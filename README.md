# Bubbola

[![CI](https://github.com/vigji/bubbola/workflows/CI/badge.svg)](https://github.com/vigji/bubbola/actions/workflows/ci.yml)
[![Build Binaries](https://github.com/vigji/bubbola/workflows/Build%20Binaries/badge.svg)](https://github.com/vigji/bubbola/actions/workflows/build-binaries.yml)
[![Codecov](https://codecov.io/gh/vigji/bubbola/branch/main/graph/badge.svg)](https://codecov.io/gh/vigji/bubbola)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A Python application for PDF and image processing with AI-powered text recognition and analysis. The tool processes documents to identify and extract relevant information such as measurements, specifications, and other technical data, focusing on delivery notes.

It features an executable distribution using PyInstaller.

## Istruzioni

### Utilizzo del Tool CLI

Bubbola è un tool da riga di comando per l'elaborazione di PDF e immagini. Ecco i comandi principali:

#### Comandi Base

```bash
bubbola --help  # help

bubbola version ext  # versione

bubbola sanitize tests/assets/0088_001.pdf  # prepara un pdf per la lettura
bubbola sanitize tests/assets/single_pages/0088_001_001.png  # prepara un'immagine per la lettura
bubbola sanitize tests/assets/single_pages  # prepara i dati di una cartella intera per la lettura

# Lista i flussi disponibili per il processamento delle immagine
bubbola list

# Processa i dati di una cartella (tests/assets/single_pages) usando il flusso small_test
bubbola extract --input tests/assets/single_pages/ --flow small_test
```

#### Opzioni Disponibili

- `--output <directory>`: Specifica la directory di output (per sanitize)
- `--max-size <pixels>`: Dimensione massima dell'edge in pixel (per sanitize)
- `--input <path>`: Percorso ai file da elaborare (per extract)
- `--flow <name>`: Nome del flusso da utilizzare (per extract)
- `--yes, -y`: Procede automaticamente senza chiedere conferma (per extract)

Altri parametri dipendono dal flusso utilizzato, ad esempio:
- `--suppliers-csv <file>`: File CSV fornitori (per extract)
- `--prices-csv <file>`: File CSV prezzi (per extract)
- `--fattura <file>`: File XML con la fattura elettronica (per extract)

**Nota:** Prima di utilizzare il tool, è necessario configurare le chiavi API. Vedi [CONFIGURATION.md](CONFIGURATION.md) per le istruzioni di configurazione.

## Fussi implementati

Di seguito elenco e istruzioni per i flussi implementati fino ad ora.

### `small_parsing`

Un test, produce semplicemente una breve descrizione dell'immagine.

#### Esempio di utilizzo

```bash
bubbola extract --input /folder/with/pdfs --flow small_parsing
```



### `fattura_check_v1`

Processa le immagini delle bolle relative a una fattura elettronica. Per ogni bolla, prova a verificare se il codice della bolla compare sulla fattura (con un fuzzy match), e se tutti gli articoli della fattura attribuiti alla bolla sono presenti nella bolla. Se le due condizioni non si verificano, la bolla viene analizzata di nuovo, fino a un max di `n_retries` tentativi (3 di default). In testing il modello configurato per il flow (`o4-mini` con reasoning effort `medium`) non ha mai allucinato un falso positivo, quindi la procedura "a retries" dovrebbe essere sicura.

#### Esempio di utilizzo

```bash
bubbola extract --input /folder/with/pdfs --flow fattura_check_v1 --fattura /path/to/fattura_elettronica.xml
```

#### Output

Il flusso produce una cartella di output dentro la cartella di input (a meno che non sia specificato un'altra cartella di output con l'opzione `--output`). Nella cartella si trova:

- Per ogni pagina da processare:
   - l'immagine mandata al modello per quella pagina, eventualmente ridimensionata
   - un json con il risultato del modello per quella pagina
- un file `main_table.json` con il risultato del modello per l'intera fattura
- un file `items_table.json` con il risultato del modello per gli articoli della fattura
- un logfile con la configurazione del flusso e i risultati del processo






## Developers

### Project Structure

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
└── README.md             # This file
```

### Development Setup

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

### Building Binaries

Bubbola can be compiled into standalone executables for multiple platforms using PyInstaller. The build system uses a **single, unified approach** that works both locally and in GitHub Actions.

### Local Binary Build

**Prerequisites:**
- Python 3.11
- Build dependencies installed: `pip install -e ".[build]"`

**Recommended Build Method:**
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

- **Format code:** `uv run ruff format src/ tests/`
- **Lint code:** `uv run ruff check src/ tests/`
- **Auto-fix issues:** `uv run ruff check --fix src/ tests/`

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

## Continuous Integration (CI)

The project includes comprehensive CI checks via GitHub Actions that run on every push and pull request.

### CI Workflow

The CI workflow includes:

- **Linting**: Ruff checks for code quality and style
- **Formatting**: Ruff format verification
- **Type Checking**: MyPy static type analysis
- **Pre-commit Hooks**: Verification that all pre-commit hooks pass
- **Testing**: Pytest with coverage reporting
- **Cross-platform Testing**: Tests run on Ubuntu, macOS, and Windows

### Coverage Reporting

Test coverage is automatically calculated and reported:
- Coverage reports are generated for each CI run
- Coverage data is uploaded to Codecov for tracking
- Coverage thresholds can be configured in `pyproject.toml`

### Local Development

To run the same checks locally:

```bash
# Install development dependencies
uv sync --group dev

# Run all CI checks
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pre-commit run --all-files
uv run pytest --cov=src --cov-report=term-missing
```

## Development Workflow

1. **Start development:**
   ```bash
   pip install -e ".[dev,build]"
   ```

2. **Make changes to the code**

3. **Test your changes:**
   ```bash
   uv run pytest
   uv run ruff check src/ tests/
   ```

4. **Format and lint code:**
   ```bash
   uv run ruff format src/ tests/
   uv run ruff check --fix src/ tests/
   ```

5. **Build and test executable:**
   ```bash
   python scripts/build_binary.py
   ./dist/bubbola --help
   ```

## Common Development Commands

```bash
# Installation and setup
uv sync                    # Sync dependencies
uv pip install -e ".[build]"  # Install in development mode

# Testing
uv run pytest             # Run all tests
uv run pytest tests/test_specific.py  # Run specific test file

# Code quality
uv run ruff check src/ tests/     # Lint code
uv run ruff format src/ tests/    # Format code
uv run ruff check --fix src/ tests/  # Auto-fix issues

# Building (use pip for better PyInstaller compatibility)
pip install -e ".[build]"         # Install build dependencies
python scripts/build_binary.py    # Build binary
python scripts/build_binary.py --clean  # Clean build

# Running the application
uv run python -m bubbola.cli --help  # Run CLI with help
uv run python -m bubbola.cli version  # Run specific command
```

## Configuration

Bubbola uses a simple configuration system. Create your config file:

```bash
python scripts/setup_config.py
```

This creates `~/.bubbola/config.env` with your API keys. See [CONFIGURATION.md](CONFIGURATION.md) for details.

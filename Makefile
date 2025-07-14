.PHONY: help install install-dev test lint format clean build dist

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package in development mode
	uv pip install -e .

install-dev:  ## Install the package with development dependencies
	uv pip install -e ".[dev]"

uv-sync:  ## Sync dependencies with uv
	uv sync

uv-run:  ## Run the application with uv
	uv run python -m bubbola.cli

test:  ## Run tests
	pytest

lint:  ## Run linting checks
	ruff check src/ tests/
	mypy src/

format:  ## Format code with ruff
	ruff format src/ tests/
	ruff check --fix src/ tests/

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean  ## Build the executable with PyInstaller
	pyinstaller build.spec

dist: build  ## Create distribution (alias for build)
	@echo "Distribution created in dist/ directory"

run:  ## Run the application
	uv run python -m bubbola.cli

run-exe:  ## Run the built executable
	./dist/bubbola/bubbola 
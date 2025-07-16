.PHONY: help install install-dev build clean test lint format check

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install the package in development mode"
	@echo "  install-dev - Install development dependencies"
	@echo "  build       - Build standalone binary"
	@echo "  build-clean - Clean and build binary"
	@echo "  clean       - Clean build artifacts"
	@echo "  test        - Run tests"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code"
	@echo "  check       - Run all checks (lint + test)"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,build]"

# Building
build:
	python scripts/build_binary.py

build-clean:
	python scripts/build_binary.py --clean

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -f bubbola.spec
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Testing
test:
	pytest

# Linting and formatting
lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

# All checks
check: lint test 
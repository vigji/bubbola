# Build System Documentation

This document provides quick reference for building Bubbola binaries using our unified build system.

**Unified Approach:** The same build script (`scripts/build_binary.py`) is used both locally and in GitHub Actions, ensuring consistency across all platforms.

## Quick Start

```bash
# Install build dependencies
pip install -e ".[build]"

# Build for current platform
make build

# Test the binary
./dist/bubbola --help
```

## Build Commands

### Recommended: Using Makefile
```bash
make build          # Build binary
make build-clean    # Clean and build
make clean          # Clean artifacts
```

### Alternative: Direct Build Script
```bash
python scripts/build_binary.py
python scripts/build_binary.py --clean
python scripts/build_binary.py --output-dir ./custom-dist
```



## Supported Platforms

| Platform | Architecture | Binary Name | Notes |
|----------|-------------|-------------|-------|
| macOS | x64 | `bubbola` | Intel Macs |
| macOS | ARM64 | `bubbola` | Apple Silicon |
| Windows | x64 | `bubbola.exe` | Windows 10/11 |
| Linux | x64 | `bubbola` | Ubuntu 22.04+ |

## GitHub Actions

### Automatic Builds
- **Push to main**: Builds for testing
- **Tag push**: Creates release with binaries
- **Manual trigger**: Via GitHub Actions UI

### Creating Releases
```bash
git tag v1.0.0
git push origin v1.0.0
```

## Troubleshooting

### Common Issues

1. **Import errors**: Check `bubbola.spec` includes all modules
2. **Large binary size**: Normal for Python apps with heavy dependencies
3. **macOS code signing**: Use `--codesign-identity -` for ad-hoc signing
4. **Windows antivirus**: May trigger false positives

### Build Dependencies
- Python 3.11
- PyInstaller >= 6.0.0
- pyinstaller-hooks-contrib >= 2023.0

### File Structure
```
dist/
├── bubbola          # macOS/Linux binary
└── bubbola.exe      # Windows binary
```

## Configuration Files

- `bubbola.spec`: PyInstaller configuration
- `scripts/build_binary.py`: Build automation script
- `.github/workflows/build-binaries.yml`: CI/CD workflow
- `pyproject.toml`: Build dependencies in `[project.optional-dependencies.build]` 
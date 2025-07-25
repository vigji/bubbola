#!/usr/bin/env python3
"""Build script for creating standalone binaries of the bubbola application."""

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def get_platform_info() -> dict[str, str]:
    """Get platform-specific information for building."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        return {
            "system": "macos",
            "arch": "x86_64" if machine == "x86_64" else "arm64",
            "ext": "",
            "pyinstaller_arch": "x86_64" if machine == "x86_64" else "arm64",
        }
    elif system == "windows":
        return {
            "system": "windows",
            "arch": "x64" if machine == "amd64" else machine,
            "ext": ".exe",
            "pyinstaller_arch": "x64" if machine == "amd64" else machine,
        }
    elif system == "linux":
        return {
            "system": "linux",
            "arch": "x86_64" if machine == "x86_64" else machine,
            "ext": "",
            "pyinstaller_arch": "x86_64" if machine == "x86_64" else machine,
        }
    else:
        raise ValueError(f"Unsupported platform: {system}")


def build_binary(output_dir: Path, clean: bool = False) -> None:
    """Build the binary using PyInstaller."""
    platform_info = get_platform_info()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean previous builds if requested
    if clean:
        for path in ["build", "dist"]:
            if Path(path).exists():
                shutil.rmtree(path)
        # Don't delete the spec file as it's part of the build configuration

    # PyInstaller command - use __init__.py as entry point
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        "bubbola",
        "--distpath",
        str(output_dir),
        "--workpath",
        "build",
        "--clean",
        "--noconfirm",
        "src/bubbola/__init__.py",
    ]

    # Platform-specific options are handled in the spec file

    print(f"Building for {platform_info['system']} ({platform_info['arch']})...")
    print(f"Command: {' '.join(cmd)}")

    # Run PyInstaller
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    if result.returncode == 0:
        binary_name = f"bubbola{platform_info['ext']}"
        binary_path = output_dir / binary_name

        if binary_path.exists():
            print(f"[SUCCESS] Binary created successfully: {binary_path}")
            print(f"   Size: {binary_path.stat().st_size / (1024 * 1024):.1f} MB")
        else:
            print("[ERROR] Binary not found after build")
            sys.exit(1)
    else:
        print("[ERROR] Build failed:")
        print(result.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build bubbola binary")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("dist"),
        help="Output directory for binaries (default: dist)",
    )
    parser.add_argument(
        "--clean",
        "-c",
        action="store_true",
        help="Clean previous builds before building",
    )

    args = parser.parse_args()

    try:
        build_binary(args.output_dir, args.clean)
    except KeyboardInterrupt:
        print("\nBuild cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Build failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

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
        spec_file = Path("bubbola.spec")
        if spec_file.exists():
            spec_file.unlink()

    # PyInstaller command
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
    ]
    if platform_info["system"] == "macos":
        cmd.append("--onedir")
    else:
        cmd.append("--onefile")
    cmd.extend(
        [
            "--name",
            "bubbola",
            "--distpath",
            str(output_dir),
            "--workpath",
            "build",
            "--specpath",
            ".",
            "--clean",
            "--noconfirm",
            "src/bubbola/cli.py",
        ]
    )

    # Platform-specific options
    if platform_info["system"] == "macos":
        cmd.extend(
            [
                "--target-arch",
                platform_info["pyinstaller_arch"],
                "--codesign-identity",
                "-",  # Ad-hoc signing
                "--no-bundle-python",  # Don't bundle Python to avoid code signing issues
            ]
        )
    elif platform_info["system"] == "windows":
        cmd.extend(
            [
                "--target-arch",
                platform_info["pyinstaller_arch"],
                "--uac-admin",  # Request admin privileges if needed
            ]
        )

    print(f"Building for {platform_info['system']} ({platform_info['arch']})...")
    print(f"Command: {' '.join(cmd)}")

    # Run PyInstaller
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    if result.returncode == 0:
        binary_name = f"bubbola{platform_info['ext']}"
        binary_path = output_dir / binary_name

        if binary_path.exists():
            print(f"✅ Binary created successfully: {binary_path}")
            print(f"   Size: {binary_path.stat().st_size / (1024 * 1024):.1f} MB")
        else:
            print("❌ Binary not found after build")
            sys.exit(1)
    else:
        print("❌ Build failed:")
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

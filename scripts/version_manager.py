#!/usr/bin/env python3
"""Version management script for Bubbola project.

This script handles version bumping following Semantic Versioning (SemVer).
It updates both pyproject.toml and src/bubbola/__init__.py to keep them in sync.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def get_current_version() -> str:
    """Get current version from __init__.py"""
    init_file = Path("src/bubbola/__init__.py")
    if not init_file.exists():
        raise FileNotFoundError("src/bubbola/__init__.py not found")
    
    content = init_file.read_text()
    version_match = re.search(r'__version__ = ["\']([^"\']+)["\']', content)
    if not version_match:
        raise ValueError("Version not found in __init__.py")
    
    return version_match.group(1)


def validate_version(version: str) -> bool:
    """Validate version follows SemVer format (X.Y.Z)"""
    pattern = r'^\d+\.\d+\.\d+$'
    return bool(re.match(pattern, version))


def bump_version(current: str, bump_type: str) -> str:
    """Bump version according to SemVer rules"""
    parts = current.split('.')
    major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    
    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")
    
    return f"{major}.{minor}.{patch}"


def update_init_file(new_version: str) -> None:
    """Update version in __init__.py"""
    init_file = Path("src/bubbola/__init__.py")
    content = init_file.read_text()
    
    new_content = re.sub(
        r'__version__ = ["\'][^"\']+["\']',
        f'__version__ = "{new_version}"',
        content
    )
    
    init_file.write_text(new_content)
    print(f"Updated {init_file}")


def update_pyproject_toml(new_version: str) -> None:
    """Update version in pyproject.toml"""
    pyproject_file = Path("pyproject.toml")
    content = pyproject_file.read_text()
    
    new_content = re.sub(
        r'version = ["\'][^"\']+["\']',
        f'version = "{new_version}"',
        content
    )
    
    pyproject_file.write_text(new_content)
    print(f"Updated {pyproject_file}")


def create_git_tag(version: str, push: bool = False) -> None:
    """Create and optionally push git tag"""
    tag_name = f"v{version}"
    
    # Create tag
    subprocess.run(["git", "tag", "-a", tag_name, "-m", f"Release {version}"], check=True)
    print(f"Created git tag: {tag_name}")
    
    if push:
        subprocess.run(["git", "push", "origin", tag_name], check=True)
        print(f"Pushed tag: {tag_name}")


def commit_version_changes(version: str) -> None:
    """Commit version changes to git"""
    files = ["src/bubbola/__init__.py", "pyproject.toml"]
    
    # Add files
    subprocess.run(["git", "add"] + files, check=True)
    
    # Commit
    commit_msg = f"Bump version to {version}"
    subprocess.run(["git", "commit", "-m", commit_msg], check=True)
    print(f"Committed version changes: {commit_msg}")


def main() -> None:
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Manage Bubbola version")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Show current version
    subparsers.add_parser("current", help="Show current version")
    
    # Set specific version
    set_parser = subparsers.add_parser("set", help="Set specific version")
    set_parser.add_argument("version", help="Version to set (e.g., 1.2.3)")
    set_parser.add_argument("--no-commit", action="store_true", help="Don't commit changes")
    set_parser.add_argument("--no-tag", action="store_true", help="Don't create git tag")
    set_parser.add_argument("--push", action="store_true", help="Push tag to origin")
    
    # Bump version
    bump_parser = subparsers.add_parser("bump", help="Bump version")
    bump_parser.add_argument(
        "type", 
        choices=["major", "minor", "patch"], 
        help="Version component to bump"
    )
    bump_parser.add_argument("--no-commit", action="store_true", help="Don't commit changes")
    bump_parser.add_argument("--no-tag", action="store_true", help="Don't create git tag")
    bump_parser.add_argument("--push", action="store_true", help="Push tag to origin")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        current_version = get_current_version()
        
        if args.command == "current":
            print(f"Current version: {current_version}")
            return
        
        # Determine new version
        if args.command == "set":
            if not validate_version(args.version):
                print(f"Error: Invalid version format '{args.version}'. Use X.Y.Z format.")
                sys.exit(1)
            new_version = args.version
        else:  # bump
            new_version = bump_version(current_version, args.type)
        
        print(f"Updating version: {current_version} -> {new_version}")
        
        # Update files
        update_init_file(new_version)
        update_pyproject_toml(new_version)
        
        # Git operations
        if not args.no_commit:
            commit_version_changes(new_version)
        
        if not args.no_tag:
            create_git_tag(new_version, args.push)
        
        print(f"✅ Version successfully updated to {new_version}")
        
        if not args.no_tag and not args.push:
            print(f"💡 To trigger release, run: git push origin v{new_version}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
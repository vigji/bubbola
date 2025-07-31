#!/usr/bin/env python3
"""Script to verify GitHub releases and their assets.

This script helps verify that releases are properly created with binaries attached.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def run_command(cmd: List[str], capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    try:
        return subprocess.run(cmd, capture_output=capture_output, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command {' '.join(cmd)}: {e}")
        sys.exit(1)


def get_latest_tag() -> str:
    """Get the latest git tag."""
    result = run_command(["git", "describe", "--tags", "--abbrev=0"])
    return result.stdout.strip()


def get_release_info(tag: str) -> Optional[Dict]:
    """Get release information from GitHub API."""
    # Use gh CLI if available, otherwise use curl
    try:
        result = run_command(["gh", "api", f"repos/vigji/bubbola/releases/tags/{tag}"])
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("GitHub CLI not available, trying curl...")
        try:
            result = run_command([
                "curl", "-s", 
                f"https://api.github.com/repos/vigji/bubbola/releases/tags/{tag}"
            ])
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            print(f"Could not fetch release info for tag {tag}")
            return None


def verify_release_assets(release_info: Dict) -> bool:
    """Verify that a release has the expected assets."""
    if not release_info or "assets" not in release_info:
        print("❌ No release info or assets found")
        return False
    
    assets = release_info["assets"]
    expected_assets = [
        "bubbola-macos-x64",
        "bubbola-macos-arm64", 
        "bubbola-windows-x64.exe",
        "bubbola-linux-x64",
        "checksums.txt"
    ]
    
    asset_names = [asset["name"] for asset in assets]
    missing_assets = [name for name in expected_assets if name not in asset_names]
    
    if missing_assets:
        print(f"❌ Missing assets: {missing_assets}")
        return False
    
    print("✅ All expected assets found:")
    for asset in assets:
        size_mb = asset["size"] / (1024 * 1024)
        print(f"   - {asset['name']} ({size_mb:.1f} MB)")
    
    return True


def list_releases() -> None:
    """List all releases with their status."""
    try:
        result = run_command(["gh", "api", "repos/vigji/bubbola/releases"])
        releases = json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("GitHub CLI not available, cannot list releases")
        return
    
    print("📋 Recent releases:")
    for release in releases[:5]:  # Show last 5 releases
        tag = release["tag_name"]
        name = release["name"] or tag
        assets_count = len(release["assets"])
        print(f"   {tag}: {name} ({assets_count} assets)")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Verify GitHub releases")
    parser.add_argument("--tag", help="Specific tag to verify (default: latest)")
    parser.add_argument("--list", action="store_true", help="List recent releases")
    
    args = parser.parse_args()
    
    if args.list:
        list_releases()
        return
    
    # Get tag to verify
    if args.tag:
        tag = args.tag
    else:
        tag = get_latest_tag()
    
    print(f"🔍 Verifying release: {tag}")
    
    # Get release info
    release_info = get_release_info(tag)
    if not release_info:
        print(f"❌ Release {tag} not found or could not be fetched")
        sys.exit(1)
    
    # Verify assets
    success = verify_release_assets(release_info)
    
    if success:
        print(f"✅ Release {tag} is properly configured with all binaries")
    else:
        print(f"❌ Release {tag} is missing required assets")
        sys.exit(1)


if __name__ == "__main__":
    main() 
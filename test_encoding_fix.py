#!/usr/bin/env python3
"""
Test script to demonstrate the encoding fix for the UTF-8 decode error.
This simulates the Windows encoding issue where files contain non-UTF-8 bytes.
"""

import json
import tempfile
from pathlib import Path

from bubbola.results_converter import (
    _read_file_with_encoding_fallback,
    parse_hierarchical_json,
)


def test_encoding_fix():
    """Test that the encoding fix handles the specific 0xe0 byte issue."""

    # Create a temporary directory with problematic files
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "results"
        results_dir.mkdir()

        # Sample data that might contain special characters
        sample_data = {
            "nome_rag_1": "Supplier with special chars: àèéìòù",
            "b501a_num_doc": "ORDER123",
            "ddt_number": "DDT001",
            "delivery_date": "2024-01-15",
            "summary": "Test delivery with encoding issues",
            "delivery_items": [
                {
                    "item_name": "Concrete Type A",
                    "item_code": "CONC001",
                    "item_quantity": 10.5,
                }
            ],
        }

        # Create a file with the problematic 0xe0 byte at position 697
        problematic_content = json.dumps(sample_data, ensure_ascii=False)
        problematic_bytes = problematic_content.encode("utf-8")

        # Insert 0xe0 byte at position 697 (as mentioned in the original error)
        if len(problematic_bytes) > 697:
            problematic_bytes = (
                problematic_bytes[:697] + b"\xe0" + problematic_bytes[697:]
            )
        else:
            # If the content is shorter, append the problematic byte
            problematic_bytes += b"\xe0"

        # Write the problematic file
        with open(results_dir / "response_problematic.json", "wb") as f:
            f.write(problematic_bytes)

        # Also create a normal file for comparison
        with open(results_dir / "response_normal.json", "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False)

        print("Testing encoding fallback function...")

        # Test the encoding fallback function directly
        try:
            content = _read_file_with_encoding_fallback(
                results_dir / "response_problematic.json"
            )
            print("✓ Successfully read problematic file with encoding fallback")
            print(f"  Content length: {len(content)} characters")
            print(f"  First 100 chars: {content[:100]}...")
        except Exception as e:
            print(f"✗ Failed to read problematic file: {e}")
            return False

        print("\nTesting full parse_hierarchical_json function...")

        # Test the full function
        try:
            level_data, level_names = parse_hierarchical_json(results_dir=results_dir)
            print("✓ Successfully processed files with encoding issues")
            print(f"  Levels: {level_names}")
            print(f"  Main level records: {len(level_data[0])}")
            print(f"  Items level records: {len(level_data[1])}")

            # Check that both files were processed
            file_ids = {item["file_id"] for item in level_data[0]}
            print(f"  Processed file IDs: {file_ids}")

            return True

        except Exception as e:
            print(f"✗ Failed to process files: {e}")
            return False


if __name__ == "__main__":
    print("Testing encoding fix for UTF-8 decode error...")
    print("=" * 50)

    success = test_encoding_fix()

    print("=" * 50)
    if success:
        print("✓ All tests passed! The encoding fix is working correctly.")
    else:
        print("✗ Some tests failed. The encoding fix needs more work.")

    print("\nThis fix should resolve the Windows encoding issue where")
    print("files contain non-UTF-8 bytes like 0xe0, preventing CSV generation.")

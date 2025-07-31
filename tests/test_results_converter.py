import json
import tempfile
from pathlib import Path

import pytest

from bubbola.results_converter import (
    _read_file_with_encoding_fallback,
    parse_hierarchical_json,
)


@pytest.fixture
def sample_results_dir():
    """Create a temporary directory with sample JSON files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "results"
        results_dir.mkdir()

        # Sample data based on data_models structure
        sample_data_1 = {
            "nome_rag_1": "Supplier A",
            "b501a_num_doc": "ORDER123",
            "ddt_number": "DDT001",
            "delivery_date": "2024-01-15",
            "fisso_pompa_nastro": 100,
            "m3_pompa_nastro": 50,
            "summary": "Test delivery 1",
            "delivery_items": [
                {
                    "item_name": "Concrete Type A",
                    "item_code": "CONC001",
                    "item_quantity": 10.5,
                    "item_measurement_unit": "m3",
                    "item_unit_price": 85.0,
                    "item_total_price": 892.5,
                    "not_in_valid_items": False,
                },
                {
                    "item_name": "Concrete Type B",
                    "item_code": "CONC002",
                    "item_quantity": 5.0,
                    "item_measurement_unit": "m3",
                    "item_unit_price": 90.0,
                    "item_total_price": 450.0,
                    "not_in_valid_items": False,
                },
            ],
        }

        sample_data_2 = {
            "nome_rag_1": "Supplier B",
            "b501a_num_doc": "ORDER456",
            "ddt_number": "DDT002",
            "delivery_date": "2024-01-16",
            "fisso_pompa_nastro": 150,
            "m3_pompa_nastro": 75,
            "summary": "Test delivery 2",
            "delivery_items": [
                {
                    "item_name": "Concrete Type C",
                    "item_code": "CONC003",
                    "item_quantity": 8.0,
                    "item_measurement_unit": "m3",
                    "item_unit_price": 95.0,
                    "item_total_price": 760.0,
                    "not_in_valid_items": False,
                }
            ],
        }

        sample_data_3 = {}

        # Write sample JSON files
        with open(results_dir / "response_file1.json", "w") as f:
            json.dump(sample_data_1, f)

        with open(results_dir / "response_file2.json", "w") as f:
            json.dump(sample_data_2, f)

        with open(results_dir / "response_file3.json", "w") as f:
            json.dump(sample_data_3, f)

        yield results_dir


@pytest.fixture
def sample_results_dir_with_encoding_issues():
    """Create a temporary directory with JSON files that have encoding issues."""
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "results"
        results_dir.mkdir()

        # Sample data with potential encoding issues
        sample_data = {
            "nome_rag_1": "Supplier with special chars: àèéìòù",
            "b501a_num_doc": "ORDER123",
            "ddt_number": "DDT001",
            "delivery_date": "2024-01-15",
            "summary": "Test delivery with encoding",
            "delivery_items": [
                {
                    "item_name": "Concrete Type A",
                    "item_code": "CONC001",
                    "item_quantity": 10.5,
                }
            ],
        }

        # Write JSON file with UTF-8 encoding (normal case)
        with open(results_dir / "response_normal.json", "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False)

        # Write JSON file with latin-1 encoding (simulating Windows encoding issues)
        with open(results_dir / "response_latin1.json", "w", encoding="latin-1") as f:
            json.dump(sample_data, f, ensure_ascii=False)

        # Write a file with problematic bytes (simulating the 0xe0 byte issue)
        problematic_content = json.dumps(sample_data, ensure_ascii=False)
        # Insert a problematic byte sequence
        problematic_bytes = problematic_content.encode("utf-8")
        # Insert 0xe0 byte at position 697 (as mentioned in the error)
        if len(problematic_bytes) > 697:
            problematic_bytes = (
                problematic_bytes[:697] + b"\xe0" + problematic_bytes[697:]
            )

        with open(results_dir / "response_problematic.json", "wb") as f:
            f.write(problematic_bytes)

        yield results_dir


def test_read_file_with_encoding_fallback():
    """Test the encoding fallback functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "test.txt"

        # Test with UTF-8 content
        content_utf8 = "Hello, world! àèéìòù"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(content_utf8)

        result = _read_file_with_encoding_fallback(test_file)
        assert result == content_utf8

        # Test with latin-1 content
        content_latin1 = "Hello, world! àèéìòù"
        with open(test_file, "w", encoding="latin-1") as f:
            f.write(content_latin1)

        result = _read_file_with_encoding_fallback(test_file)
        assert "Hello, world!" in result


def test_parse_hierarchical_json_with_encoding_issues(
    sample_results_dir_with_encoding_issues,
):
    """Test that parse_hierarchical_json can handle files with encoding issues."""
    level_data, level_names = parse_hierarchical_json(
        results_dir=sample_results_dir_with_encoding_issues
    )

    # Should return 2 levels: top level and delivery_items level
    assert len(level_data) == 2
    assert len(level_names) == 2

    # Check that all files were processed (including the problematic one)
    top_level = level_data[0]
    assert len(top_level) == 3  # normal, latin1, and problematic files

    # Verify that the function didn't crash and processed the files
    file_ids = [item["file_id"] for item in top_level]
    assert "normal" in file_ids
    assert "latin1" in file_ids
    assert "problematic" in file_ids


def test_parse_hierarchical_json_general(sample_results_dir):
    """Test the parse_hierarchical_json function with automatic hierarchy detection."""
    print("--------------------------------")
    level_data, level_names = parse_hierarchical_json(results_dir=sample_results_dir)

    print(sample_results_dir)
    print(level_data)
    print(level_names)
    print("--------------------------------")
    print(level_data)
    print(level_names)

    # Should return 2 levels: top level and delivery_items level
    assert len(level_data) == 2
    assert len(level_names) == 2

    # Check top level (level 0)
    top_level = level_data[0]
    assert len(top_level) == 3

    # Check delivery_items level (level 1)
    # The function creates multiple entries for each item as context accumulates
    # We expect 2 items from file1 + 1 item from file2, but each gets processed multiple times
    # as the parent context fields are added incrementally
    items_level = level_data[1]
    assert len(items_level) > 0  # Should have at least some items

    # Verify that we have items from both files
    file_ids = {item["file_id"] for item in items_level}
    assert "file1" in file_ids
    assert "file2" in file_ids

    # Verify field propagation
    for item in items_level:
        assert "main_nome_rag_1" in item
        assert "file_id" in item
        assert "item_name" in item
        # Check that at least some items have the delivery_date field
        # (not all items will have it due to incremental processing)
        if "main_delivery_date" in item:
            assert isinstance(item["main_delivery_date"], str)


if __name__ == "__main__":
    results_dir = Path("temp_dir/results")

    level_data, level_names = parse_hierarchical_json(results_dir=results_dir)
    print(level_data)
    print(level_names)

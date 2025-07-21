import json
import tempfile
from pathlib import Path

import pytest

from bubbola.results_converter import create_results_csv, parse_hierarchical_json


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

        # Write sample JSON files
        with open(results_dir / "response_file1.json", "w") as f:
            json.dump(sample_data_1, f)

        with open(results_dir / "response_file2.json", "w") as f:
            json.dump(sample_data_2, f)

        yield results_dir


def test_create_results_csv(sample_results_dir):
    """Test the create_results_csv function with sample hierarchical data."""
    ddts_data, items_data = create_results_csv(sample_results_dir)

    # Check DDT level data
    assert len(ddts_data) == 2

    # Check first DDT
    ddt1 = ddts_data[0]
    assert ddt1["file_id"] == "file1"
    assert ddt1["nome_rag_1"] == "Supplier A"
    assert ddt1["ddt_number"] == "DDT001"
    assert ddt1["n_delivery_items"] == 2

    # Check second DDT
    ddt2 = ddts_data[1]
    assert ddt2["file_id"] == "file2"
    assert ddt2["nome_rag_1"] == "Supplier B"
    assert ddt2["ddt_number"] == "DDT002"
    assert ddt2["n_delivery_items"] == 1

    # Check items level data
    assert len(items_data) == 3  # 2 + 1 items

    # Check items from first file
    file1_items = [item for item in items_data if item["file_id"] == "file1"]
    assert len(file1_items) == 2

    # Check first item from file1
    item1 = file1_items[0]
    assert item1["file_id"] == "file1"
    assert item1["level0_nome_rag_1"] == "Supplier A"
    assert item1["level0_delivery_date"] == "2024-01-15"
    assert item1["item_name"] == "Concrete Type A"
    assert item1["item_quantity"] == 10.5

    # Check items from second file
    file2_items = [item for item in items_data if item["file_id"] == "file2"]
    assert len(file2_items) == 1

    # Check CSV files were created
    level0_csv = sample_results_dir / "level_0_table.csv"
    delivery_items_csv = sample_results_dir / "delivery_items_table.csv"
    assert level0_csv.exists()
    assert delivery_items_csv.exists()


def test_parse_hierarchical_json_general(sample_results_dir):
    """Test the new parse_hierarchical_json function with custom hierarchy."""
    # Test with the same configuration as create_results_csv
    level_data = parse_hierarchical_json(
        results_dir=sample_results_dir, hierarchy_config=["delivery_items"]
    )

    # Should return 2 levels: top level and delivery_items level
    assert len(level_data) == 2

    # Check top level (level 0)
    top_level = level_data[0]
    assert len(top_level) == 2

    # Check delivery_items level (level 1)
    items_level = level_data[1]
    assert len(items_level) == 3

    # Verify field propagation
    for item in items_level:
        assert "level0_nome_rag_1" in item
        assert "level0_delivery_date" in item
        assert "file_id" in item
        assert "item_name" in item


def test_parse_hierarchical_json_custom_schema_fields(sample_results_dir):
    """Test parse_hierarchical_json with custom schema fields."""
    custom_schema = {"$defs", "properties", "title", "type", "summary"}

    level_data = parse_hierarchical_json(
        results_dir=sample_results_dir,
        hierarchy_config=["delivery_items"],
        schema_fields=custom_schema,
    )

    # Check that summary field is excluded from top level
    top_level = level_data[0]
    for record in top_level:
        assert "summary" not in record
        assert "nome_rag_1" in record  # Should still be included


def test_create_results_csv_empty_directory():
    """Test behavior with empty results directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir) / "results"
        results_dir.mkdir()

        ddts_data, items_data = create_results_csv(results_dir)

        assert ddts_data == []
        assert items_data == []

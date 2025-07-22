import json
import tempfile
from pathlib import Path

import pytest

from bubbola.results_converter import parse_hierarchical_json


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


def test_parse_hierarchical_json_general(sample_results_dir):
    """Test the parse_hierarchical_json function with automatic hierarchy detection."""
    level_data, level_names = parse_hierarchical_json(results_dir=sample_results_dir)

    # Should return 2 levels: top level and delivery_items level
    assert len(level_data) == 2
    assert len(level_names) == 2

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

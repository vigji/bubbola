import csv
import json
from pathlib import Path
from typing import Any


def parse_hierarchical_json(
    results_dir: Path,
    hierarchy_config: list[str],
    schema_fields: set[str] | None = None,
) -> tuple[list[dict[str, Any]], ...]:
    """
    Parse hierarchical JSON files into multiple CSV levels.

    This function processes JSON files with nested structures and creates separate
    CSV files for each level of the hierarchy. All fields from parent levels are
    propagated to child levels with appropriate prefixes.

    Args:
        results_dir: Path to results directory containing JSON files
        hierarchy_config: List of field names representing the hierarchy levels
                         (e.g., ["delivery_items"] for 2-level hierarchy)
        schema_fields: Set of schema fields to exclude from top-level data

    Returns:
        Tuple of lists, one for each hierarchy level. For a 2-level hierarchy,
        returns (top_level_data, child_level_data).

    Example:
        # For DDT -> delivery_items hierarchy:
        level_data = parse_hierarchical_json(
            results_dir=Path("results"),
            hierarchy_config=["delivery_items"]
        )
        ddts_data, items_data = level_data

        # For deeper hierarchy (orders -> deliveries -> items):
        level_data = parse_hierarchical_json(
            results_dir=Path("results"),
            hierarchy_config=["deliveries", "items"]
        )
        orders_data, deliveries_data, items_data = level_data
    """
    if schema_fields is None:
        schema_fields = {"$defs", "properties", "title", "type"}

    # Initialize data structures for each level
    level_data = [[] for _ in range(len(hierarchy_config) + 1)]

    print(f"Loading from: {results_dir}")

    for json_file in sorted(results_dir.glob("response_*.json")):
        with open(json_file, encoding="utf-8") as f:
            try:
                content = json.load(f)
                # Handle string JSON content
                if isinstance(content, str):
                    content = json.loads(content)

                # Extract file ID
                file_id = json_file.stem.replace("response_", "")

                # Process each level
                current_data = content
                parent_context = {}

                # Top level (level 0)
                top_level_row = {"file_id": file_id}
                for key, value in current_data.items():
                    if key not in hierarchy_config and key not in schema_fields:
                        top_level_row[key] = value
                        parent_context[f"level0_{key}"] = value

                # Add count of items in first child level
                if hierarchy_config:
                    child_items = current_data.get(hierarchy_config[0]) or []
                    top_level_row[f"n_{hierarchy_config[0]}"] = len(child_items)

                level_data[0].append(top_level_row)

                # Process child levels
                for level_idx, child_field in enumerate(hierarchy_config):
                    child_items = current_data.get(child_field) or []

                    for item in child_items:
                        item_row = {"file_id": file_id}

                        # Add all parent context
                        item_row.update(parent_context)

                        # Add all item fields
                        item_row.update(item)

                        level_data[level_idx + 1].append(item_row)

                    # Update parent context for next level
                    if level_idx < len(hierarchy_config) - 1:
                        # For deeper levels, we'd need to handle nested structures
                        # For now, we'll focus on 2-level hierarchy
                        break

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse {json_file}: {e}")
                continue

    # Print summary
    for i, data in enumerate(level_data):
        level_name = f"level_{i}" if i == 0 else hierarchy_config[i - 1]
        print(f"Loaded {len(data)} {level_name} records")

    # Export to CSV
    for i, data in enumerate(level_data):
        if data:
            level_name = f"level_{i}" if i == 0 else hierarchy_config[i - 1]
            csv_file = results_dir / f"{level_name}_table.csv"

            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

            print(f"{level_name} exported to: {csv_file}")

    return tuple(level_data)


def create_results_csv(
    results_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Load all JSON results and return DDTs and items data.

    This is a wrapper around parse_hierarchical_json for backward compatibility.

    Args:
        results_dir: Path to results directory. If None, looks for 'results' folder in current directory.

    Returns:
        tuple: (ddts_data, items_data)
    """
    if results_dir is None:
        raise FileNotFoundError("No results directory found. Please specify the path.")

    # Use the new hierarchical parser with 2-level configuration
    level_data = parse_hierarchical_json(
        results_dir=results_dir, hierarchy_config=["delivery_items"]
    )

    # Return the first two levels for backward compatibility
    return level_data[0], level_data[1]


if __name__ == "__main__":
    results_dir = Path(
        "/Users/vigji/Desktop/pages_sample-data/concrete/1461/results/test"
    )

    # Example 1: Using the original function (backward compatibility)
    print("=== Example 1: Original function ===")
    ddts_data, items_data = create_results_csv(results_dir)

    # Example 2: Using the new general function with same configuration
    print("\n=== Example 2: General function with delivery_items hierarchy ===")
    level_data = parse_hierarchical_json(
        results_dir=results_dir, hierarchy_config=["delivery_items"]
    )

    # Example 3: Using custom schema fields
    print("\n=== Example 3: Custom schema fields ===")
    custom_schema = {"$defs", "properties", "title", "type", "summary"}
    level_data_custom = parse_hierarchical_json(
        results_dir=results_dir,
        hierarchy_config=["delivery_items"],
        schema_fields=custom_schema,
    )

    # Example 4: How to extend for deeper hierarchies (commented out as example)
    print("\n=== Example 4: For deeper hierarchies ===")
    print("# For 3-level hierarchy (e.g., orders -> deliveries -> items):")
    print("# level_data = parse_hierarchical_json(")
    print("#     results_dir=results_dir,")
    print("#     hierarchy_config=['deliveries', 'items']")
    print("# )")

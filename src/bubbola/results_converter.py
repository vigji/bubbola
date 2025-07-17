import csv
import json
from pathlib import Path
from typing import Any

# TODO left at: why some of the docs are not read here? now we do not have empty ones.


def create_results_csv(
    results_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Load all JSON results and return DDTs and items data.

    Args:
        results_dir: Path to results directory. If None, looks for 'results' folder in current directory.

    Returns:
        tuple: (ddts_data, items_data)
    """
    if results_dir is None:
        # Try to find results directory in various locations
        if results_dir is None:
            raise FileNotFoundError(
                "No results directory found. Please specify the path."
            )

    print(f"Loading from: {results_dir}")

    ddts_list = []
    items_list = []

    for json_file in sorted(results_dir.glob("response_*.json")):
        with open(json_file, encoding="utf-8") as f:
            try:
                content = json.load(f)
                # Handle string JSON content
                if isinstance(content, str):
                    content = json.loads(content)

                # Extract file ID
                file_id = json_file.stem.replace("response_", "")

                # DDT row - automatically include all top-level fields except delivery_items and schema fields
                ddt_row = {"file_id": file_id}
                schema_fields = {
                    "$defs",
                    "properties",
                    "title",
                    "type",
                }  # Common schema fields to skip
                for key, value in content.items():
                    if key not in {"delivery_items"} | schema_fields:
                        ddt_row[key] = value

                # Add item count
                delivery_items = content.get("delivery_items") or []
                ddt_row["n_items"] = len(delivery_items)
                ddts_list.append(ddt_row)

                # Items rows - automatically include all item fields plus DDT context
                for item in delivery_items:
                    item_row = {
                        "file_id": file_id,
                        "ddt_nome_rag_1": content.get("nome_rag_1"),
                        "ddt_delivery_date": content.get("delivery_date"),
                    }
                    # Add all item fields
                    item_row.update(item)
                    items_list.append(item_row)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse {json_file}: {e}")
                continue

    print(f"Loaded {len(ddts_list)} DDTs and {len(items_list)} items")

    # Export to CSV using csv module
    if ddts_list:
        ddts_csv = results_dir / "ddts_table.csv"
        with open(ddts_csv, "w", newline="", encoding="utf-8") as f:
            if ddts_list:
                writer = csv.DictWriter(f, fieldnames=ddts_list[0].keys())
                writer.writeheader()
                writer.writerows(ddts_list)
        print(f"DDTs exported to: {ddts_csv}")

    if items_list:
        items_csv = results_dir / "items_table.csv"
        with open(items_csv, "w", newline="", encoding="utf-8") as f:
            if items_list:
                writer = csv.DictWriter(f, fieldnames=items_list[0].keys())
                writer.writeheader()
                writer.writerows(items_list)
        print(f"Items exported to: {items_csv}")

    return ddts_list, items_list


if __name__ == "__main__":
    results_dir = Path(
        "/Users/vigji/Desktop/pages_sample-data/concrete/1461/results/test"
    )
    ddts_data, items_data = create_results_csv(results_dir)

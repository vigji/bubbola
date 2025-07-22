import csv
import json
from pathlib import Path
from typing import Any


def infer_hierarchy_fields_from_json(obj: dict) -> list[str]:
    """
    Infer fields that are lists of dicts (i.e., lower hierarchies) from a JSON object.
    """
    return [
        k
        for k, v in obj.items()
        if isinstance(v, list) and v and all(isinstance(i, dict) for i in v)
    ]


def parse_hierarchical_json(
    results_dir: Path,
) -> tuple[list[list[dict[str, Any]]], list[str]]:
    """
    Parse hierarchical JSON files into multiple CSV levels, inferring hierarchy automatically.

    #TODO this function is a bit of a mess and could help some refactoring

    This function processes JSON files with nested structures and creates separate
    CSV files for each level of the hierarchy. All fields from parent levels are
    propagated to child levels with appropriate prefixes.

    Args:
        results_dir: Path to results directory containing JSON files

    Returns:
        Tuple of (list of lists of dicts, list of level names). Each list corresponds to a hierarchy level.
    """
    # Find the first JSON file to infer hierarchy
    FILE_PREFIX = "response_"
    json_files = sorted(results_dir.glob(f"{FILE_PREFIX}*.json"))
    if not json_files:
        raise FileNotFoundError(f"No {FILE_PREFIX}*.json files found in {results_dir}")
    # Find the first non-empty JSON file
    first_nonempty_content = None
    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            raw_content = f.read()
            if raw_content.strip() and len(raw_content) > 2:
                first_nonempty_content = raw_content
                break
    if first_nonempty_content is None:
        raise FileNotFoundError(
            f"All {FILE_PREFIX}*.json files in {results_dir} are empty. Cannot infer hierarchy."
        )
    current_obj = json.loads(first_nonempty_content)
    # Fix for double encodings:
    if isinstance(current_obj, str):
        current_obj = json.loads(current_obj)

    level_names = []
    hierarchy_fields = []

    while True:
        fields = infer_hierarchy_fields_from_json(current_obj)
        if not fields:
            break
        hierarchy_fields.append(fields)
        level_names.append(fields if len(fields) > 1 else fields[0])
        first_field = fields[0]
        items = current_obj.get(first_field, [])
        if not items or not isinstance(items[0], dict):
            break
        current_obj = items[0]

    flat_level_names = ["main"]
    for names in level_names:
        if isinstance(names, list):
            flat_level_names.extend(names)
        else:
            flat_level_names.append(names)

    level_data = [[] for _ in range(len(flat_level_names))]

    print(f"Loading from: {results_dir}")

    # Store expected hierarchy fields per level, as inferred from the first document
    expected_hierarchy_fields = []
    for fields in hierarchy_fields:
        expected_hierarchy_fields.append(fields)

    for json_file in json_files:
        with open(json_file, encoding="utf-8") as f:
            raw_content = f.read()
            file_id = json_file.stem.replace(FILE_PREFIX, "")
            if not raw_content.strip() or len(raw_content) < 2:
                # Empty file: add a row with only file_id
                level_data[0].append({"file_id": file_id})
                print(
                    f"Warning: {json_file} is empty. Added empty row for file_id {file_id}."
                )
                continue
            try:
                content = json.loads(raw_content)
                if isinstance(content, str):
                    content = json.loads(content)

            except Exception as e:
                # Failed to parse: treat as empty
                level_data[0].append({"file_id": file_id})
                print(
                    f"Warning: Could not parse {json_file}: {e}. Added empty row for file_id {file_id}."
                )
                continue

            def process_level(obj, parent_context, level_idx, file_id):
                row = {"file_id": file_id}
                row.update(parent_context)
                for k, v in obj.items():
                    if not (
                        isinstance(v, list)
                        and v
                        and all(isinstance(i, dict) for i in v)
                    ):
                        row[k] = v
                # Always set n_{field} for all expected lower hierarchy fields at this level
                if level_idx < len(expected_hierarchy_fields):
                    for field in expected_hierarchy_fields[level_idx]:
                        items = obj.get(field)
                        if not (
                            isinstance(items, list)
                            and all(isinstance(i, dict) for i in items)
                        ):
                            n_items = 0
                            items = []
                        else:
                            n_items = len(items)
                        row[f"n_{field}"] = n_items
                        # Only recurse if items is a non-empty list of dicts
                        for item in items:
                            child_context = {
                                f"{flat_level_names[level_idx]}_{k}": v
                                for k, v in row.items()
                            }
                            process_level(item, child_context, level_idx + 1, file_id)
                level_data[level_idx].append(row)

            process_level(content, {}, 0, file_id)

    for i, data in enumerate(level_data):
        print(f"Loaded {len(data)} {flat_level_names[i]} records")

    for i, data in enumerate(level_data):
        if data:
            level_name = flat_level_names[i]
            csv_file = results_dir / f"{level_name}_table.csv"
            # Compute union of all keys across all rows
            all_keys = set()
            for row in data:
                all_keys.update(row.keys())
            fieldnames = list(all_keys)
            with open(csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in data:
                    # Fill missing keys with empty string
                    full_row = {k: row.get(k, "") for k in fieldnames}
                    writer.writerow(full_row)
            print(f"{level_name} exported to: {csv_file}")

    return level_data, flat_level_names


if __name__ == "__main__":
    results_dir = Path(
        "/Users/vigji/Desktop/pages_sample-data/concrete_fixed/1502/results/fattura_check_v1_20250722_134952"
    )

    print("=== Example: Automatic hierarchy detection ===")
    level_data, level_names = parse_hierarchical_json(results_dir=results_dir)
    for name, data in zip(level_names, level_data, strict=False):
        print(f"{name}: {len(data)} records")
        # pprint(data)

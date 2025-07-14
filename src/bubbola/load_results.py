import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd

# TODO left at: why some of the docs are not read here? now we do not have empty ones.


def create_results_csv(results_dir: Optional[Path] = None) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
            raise FileNotFoundError("No results directory found. Please specify the path.")
    
    print(f"Loading from: {results_dir}")
    
    ddts_list = []
    items_list = []
    
    for json_file in sorted(list(results_dir.glob("response_*.json"))):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                content = json.load(f)
                # Handle string JSON content
                if isinstance(content, str):
                    content = json.loads(content)
                
                # Extract file ID
                file_id = json_file.stem.replace("response_", "")
                
                # DDT row - automatically include all top-level fields except delivery_items and schema fields
                ddt_row = {'file_id': file_id}
                schema_fields = {'$defs', 'properties', 'title', 'type'}  # Common schema fields to skip
                for key, value in content.items():
                    if key not in {'delivery_items'} | schema_fields:
                        ddt_row[key] = value
                
                # Add item count
                delivery_items = content.get('delivery_items') or []
                ddt_row['n_items'] = len(delivery_items)
                ddts_list.append(ddt_row)
                
                # Items rows - automatically include all item fields plus DDT context
                for item in delivery_items:
                    item_row = {
                        'file_id': file_id,
                        'ddt_nome_rag_1': content.get('nome_rag_1'),
                        'ddt_delivery_date': content.get('delivery_date'),
                    }
                    # Add all item fields
                    item_row.update(item)
                    items_list.append(item_row)
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse {json_file}: {e}")
                continue
    
    print(f"Loaded {len(ddts_list)} DDTs and {len(items_list)} items")
    
    # Create DataFrames
    ddts_df = pd.DataFrame(ddts_list) if ddts_list else pd.DataFrame()
    items_df = pd.DataFrame(items_list) if items_list else pd.DataFrame()
    
    # Export to CSV using pandas
    if not ddts_df.empty:
        ddts_csv = results_dir / "ddts_table.csv"
        ddts_df.to_csv(ddts_csv, index=False, encoding='utf-8')
        print(f"DDTs exported to: {ddts_csv}")
    
    if not items_df.empty:
        items_csv = results_dir / "items_table.csv"
        items_df.to_csv(items_csv, index=False, encoding='utf-8')
        print(f"Items exported to: {items_csv}")
    
    # Convert back to list of dicts for return compatibility
    ddts_data = ddts_df.to_dict('records') if not ddts_df.empty else []
    items_data = items_df.to_dict('records') if not items_df.empty else []
    
    return ddts_data, items_data


if __name__ == "__main__":
    # Handle command line argument for results directory

    
    ddts_data, items_data = create_results_csv(results_dir)
    # Basic stats
    suppliers = [d.get('nome_rag_1') for d in ddts_data if d.get('nome_rag_1')]
    unique_suppliers = set(suppliers)
    print(f"\nSummary: {len(unique_suppliers)} suppliers, {len(set(i.get('item_name', '') for i in items_data))} unique items")
    
    for supplier in unique_suppliers:
        print(f"  {supplier}: {suppliers.count(supplier)} DDTs")

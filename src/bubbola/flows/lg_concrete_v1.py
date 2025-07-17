"""Delivery notes processing flow."""

from pathlib import Path
from typing import Any

from bubbola.data_models import DeliveryNote


def get_flow(external_files: dict[str, Path] | None = None) -> dict[str, Any]:
    """Get the delivery notes processing flow configuration.

    Args:
        external_files: Optional dict mapping variable names to file paths
                        for external file injection (e.g., {"suppliers_csv": Path("suppliers.csv")})

    Returns:
        Flow configuration dictionary
    """

    # Default suppliers data
    suppliers_data = [
        ["Nome", "Indirizzo", "Partita IVA"],
        [
            "LG Concrete S.r.L.",
            "33050 Castions di Strada (UD) Via Udine,104",
            "03089360303",
        ],
    ]

    # Default prices data
    prices_data = """nome,codice,costo/m3
Cemento C12/15,C12/15,68,00
Cemento C16/20,C16/20,71,00
Cemento C20/25,C20/25,73,50
Calcestruzzo plastico (mix rif. 060 LG/23),060 LG/23,86,00
Additivo accelerante,ACCELERANTE,7,00"""

    # Handle external file injection
    if external_files and "suppliers_csv" in external_files:
        suppliers_file = external_files["suppliers_csv"]
        if suppliers_file.exists():
            with open(suppliers_file, encoding="utf-8") as f:
                suppliers_content = f.read()
                # Convert CSV content to list format for processing
                suppliers_data = [
                    line.split(",") for line in suppliers_content.strip().split("\n")
                ]

    if external_files and "prices_csv" in external_files:
        prices_file = external_files["prices_csv"]
        if prices_file.exists():
            with open(prices_file, encoding="utf-8") as f:
                prices_data = f.read()

    # Convert suppliers data to CSV format
    suppliers_csv = "\n".join([",".join(row) for row in suppliers_data])
    valid_suppliers = [row[0] for row in suppliers_data]

    # Build the system prompt with injected data
    system_prompt = f"""You are a helpful assistant in ICOP SpA accounting department.

Your task is to automatically extract structured data from a delivery note image related
to material that a supplier has delivered to an ICOP construction site.

You receive:
1. A **CSV file** containing a list of suppliers.
2. A **photo of a delivery note** from a construction site.
3. A **CSV file** containing a list of prices for possible items in the delivery note.

Here is the list of suppliers:

```
{suppliers_csv}
```

Here is the list of prices:

```
{prices_data}
```

---

Your task is to read the delivery note **carefully** and fill in the following JSON structure:

```
{DeliveryNote.model_json_schema()}
```

---

### Step-by-step instructions:

#### STEP 1: Identify the Supplier
- Look in the image for the name or VAT number (**partita IVA**) of a company.
  Make sure you consider all the possible company names (also from logos). Investigate and keep track all the options you see.
  Ignore ICOP or any of its variations; that is the receiver.
- Compare what you find with the list in the CSV, but make sure you ignore small differences like:
  - Capital letters vs lowercase
  - Missing or extra suffixes (e.g. "S.r.l.", "S.p.A.", "di ... e C.", "a socio unico")
  - Small typos or swaps of letters (e.g. "CHIURLO" vs "CHIRULO")
- If the possible match differs only slightly, keep it valid, and use the csv string as `nome_rag_1` annotation;
  The only valid entries are: {valid_suppliers}

- **IMPORTANT**: ICOP (or I.CO.P., I.C.O.P. S.p.A., etc.) is NEVER the supplier. That is the receiver. Exclude it from possible matches. If you find ICOP or any of its variations as the supplier, set `nome_rag_1` as supplier not found like below.
- If no good match is found, or yout match it ICOP or any of its variations, set `nome_rag_1` like this:
  ```
  "supplier not found. Annotations: <explain briefly why no supplier was found, listing carefully all the options you considered and why you rejected them>"
  ```

#### STEP 2: Extract the Order Number and Date
- Only if you successfully found a supplier:
  - Fill in `b501a_num_doc` with the delivery note number, only if you find a suitable match in the b501a_num_doc field of the CSV.
  - Fill in `delivery_date` with the date of the delivery. Has to be after 2024. Ignore your existing date priors, we are in the future for you. Assume all dates follow the convention DD/MM/YYYY.

#### STEP 3: Extract the Items
- Only if a valid supplier was found:
  - Extract the list of delivered items (one object per line in the table). Make sure you consider all the possible items in the ddt table, including supplementary items like "aggiunte". When looking for items, take into account the prices and items in the prices CSV.
  - If you find an item that is not in the prices CSV, add it but set `not_in_valid_items` to True.
  - For each item, extract:
    - Description
    - Code (if available)
    - Quantity
    - Unit of measurement (if the item is concrete, assume M3)

Ensure that if concrete has additions ("aggiunte"), you consider them as a separate item.

#### STEP 4: Extract the Prices
- Only if a valid supplier was found:
  - Extract the prices for the items in the delivery note.
  - Use the prices CSV to find the price for each item.
  - If the item is not found in the prices CSV, do not set a price.

---

### Final Output:
Return a single JSON object that follows exactly this format:

```
{DeliveryNote.model_json_schema()}
```
Return only the JSON entries, not the full Pydantic schema! Copy a summary of your reasoning in the `summary` field.
"""

    return {
        "data_model": "DeliveryNote",
        "system_prompt": system_prompt,
        "model_name": "o4-mini",
        "description": "Process delivery notes to extract supplier, items, and pricing information",
        "external_file_options": {
            "suppliers_csv": "Path to suppliers CSV file (optional)",
            "prices_csv": "Path to prices CSV file (optional)",
        },
    }


flow = get_flow()

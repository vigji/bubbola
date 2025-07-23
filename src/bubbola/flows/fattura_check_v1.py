"""Delivery notes processing flow."""

from pathlib import Path
from typing import Any

from bubbola.data_models import DeliveryNoteFatturaMatch
from bubbola.fattura_reader import cedente_json_str, linee_csv_str


def get_flow(external_files: dict[str, Path] | None = None) -> dict[str, Any]:
    """Get the delivery notes processing flow configuration.

    Args:
        external_files: Optional dict mapping variable names to file paths
                        for external file injection (e.g., {"suppliers_csv": Path("suppliers.csv")})

    Returns:
        Flow configuration dictionary
    """
    # assert external_files is not None, "Fattura file is mandatory"
    if external_files and "fattura" in external_files:
        fattura = external_files["fattura"]
        linee_csv = linee_csv_str(fattura)
        cedente_json = cedente_json_str(fattura)
    else:
        fattura = None
        linee_csv = None
        cedente_json = None

    # Build the system prompt with injected data
    system_prompt = f"""You are an expert large‑language‑model assistant in the accounting team of ICOP SpA.

Your task is to match a transportation document (DDT) scan against a csv with entries from an invoice.

You receive the following inputs:
- invoice_supplier (JSON, shown below)
- invoice_lines (CSV, shown below)
- ddt_scan (image of the scanned DDT)
- schema_json (target output schema, shown below)

The invoice is from: {cedente_json}. If you think that the transportation document is not from this company, return an empty match and report your reasoning. Be accurate but flexible in the matching (uppercase, lowercase, legal entity, punctuation etc).

The invoice of possible items is:
{linee_csv}

You have to:
1. find a match in the transportation DDT number and date against the invoice. If you do find a match,  If you do not find a match, still list the DDT number and date that you find but set the match field to false.
2. If you find a match:
  - ensure fields consistency: the ddt number should be the complete number you find in the invoice, including puntuation-separated parts.
  - go through all the items listed in the invoice *FOR THIS DDT ONLY* and check if the items are present in the transportation document,
  - and fill the output json with with the data you find in the transportation document, comparing items from the invoice.
  - Fill the "all_items_in_ddt" as True only if all items from the invoice are present in the DDT (should be True even if there are more items in the DDT than in the invoice).
3. If you do not find a match:
  - still list the items that you find in the DDT, and fill the output json with the data you find in the transportation document.

4. Report your reasoning in the "summary" field.

The final report should be a json with the following structure:
{DeliveryNoteFatturaMatch.model_json_schema()}

Be rigorous: minor typos in descriptions may match, but numerical fields must be exact. Flag every genuine discrepancy. Be considered of classical OCR issues, and treat the invoice as a prior; if you would find a match modulo trailing zeroes or character confusions typical of low‑contrast OCR, such as "3":"8", "5":"S", "0":"O"...

"""

    return {
        "name": "fattura_check_v1",
        "data_model": "DeliveryNoteFatturaMatch",
        "system_prompt": system_prompt,
        "model_name": "o4-mini",  # "anthropic/claude-3.7-sonnet:thinking",  #"gpt-4.1-nano", #"gpt-4.1-nano", # "google/gemini-2.5-flash-lite",  # "meta-llama/llama-4-maverick",  # "moonshotai/kimi-vl-a3b-thinking:free", # "google/gemma-3-27b-it:free",# "mistralai/mistral-small-3.2-24b-instruct:free",  #"o4-mini",  # "gpt-4o", # "o4-mini",  # "gpt-4o-mini",
        "description": "check a transportation document against an existing fattura elettronica",
        "external_file_options": {
            "fattura": "Path to fattura elettronica file (mandatory)",
        },
        "model_kwargs": {},
        "parser_kwargs": {
            "max_n_retries": 5,
            "require_true_fields": ["invoice_ddt_match", "all_items_in_ddt"],
        },
    }


flow = get_flow()

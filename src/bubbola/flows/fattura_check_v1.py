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
    system_prompt = f"""You are a helpful assistant in ICOP SpA accounting department.

Your task is to match a transportation document (DDT) scan against a csv with entries from an invoice.

The invoice is from: {cedente_json}. If you think that the transportation document is not from this company, return an empty match and report your reasoning.

The invoice of possible items is:
{linee_csv}

You have to:
1. find a match in the transportation DDT number and date against the invoice
2. If you find a match, go though all the items listed for this invoice and check if the items are present in the transportation document.
3. In the output json, fill rows with the data you find in the transportation document, comparing items from the invoice.
4. If you don't find a match, report your reasoning in the "summary" field.

The final report should be a json with the following structure:
{DeliveryNoteFatturaMatch.model_json_schema()}

My goal is to find mismatches; you have to be flexible in matching with common sense, but be accurate in raising any actual mismatch.


"""

    return {
        "name": "fattura_check_v1",
        "data_model": "DeliveryNoteFatturaMatch",
        "system_prompt": system_prompt,
        "model_name": "o4-mini",  # "gpt-4o-mini",
        "description": "check a transportation document against an existing fattura elettronica",
        "external_file_options": {
            "fattura": "Path to fattura elettronica file (mandatory)",
        },
    }


flow = get_flow()

# %%
import sys
from pathlib import Path

import pandas as pd
from fattura_reader import parse_fattura_elettronica

p = str(Path(__file__).parent.parent / "src/bubbola/")
assert Path(p).exists()
sys.path.append(p)

# %%
fattura_path = Path(
    "/Users/vigji/Desktop/pages_sample-data/concrete_old/1502/fatture/IT01378570350_jZW1y.xml"
)
results_path = Path(
    "/Users/vigji/Desktop/pages_sample-data/concrete_old/1502/results/fattura_check_v1_20250722_164322"
)
assert results_path.exists()
assert fattura_path.exists()

source_info, fattura_data = parse_fattura_elettronica(fattura_path)
fattura_data = pd.DataFrame(fattura_data)
# rename column Numer to numero_linea_invoice
fattura_data.rename(columns={c: c.lower() for c in fattura_data.columns}, inplace=True)
fattura_data.rename(
    columns={"numero": "numero_linea_invoice", "numeroddt": "main_ddt_number"},
    inplace=True,
)
fattura_data["numero_linea_invoice"] = fattura_data["numero_linea_invoice"].apply(
    lambda x: int(x) if not pd.isna(x) else -1
)

for k, v in source_info.items():
    fattura_data[k] = v

# fattura_data
# %%
parsed_rows = pd.read_csv(results_path / "items_table.csv")
parsed_rows = parsed_rows[parsed_rows["main_invoice_ddt_match"]]
parsed_rows["numero_linea_invoice"] = parsed_rows["numero_linea_invoice"].apply(
    lambda x: int(x) if not pd.isna(x) else -1
)

# remove all rows that are duplicated in all "descrizione", "quantita_in_ddt", "numero_linea_invoice",
# after printing them together with the original row, one by one:
dup_cols = ["descrizione", "quantita_in_ddt", "numero_linea_invoice", "main_ddt_number"]
duplicates = parsed_rows[parsed_rows.duplicated(subset=dup_cols, keep=False)]

# drop duplicates
parsed_rows = parsed_rows.drop_duplicates(subset=dup_cols, keep="first")

invoice_matches = parsed_rows[parsed_rows["match_in_invoice"]].sort_values(
    by="numero_linea_invoice"
)
# invoice_matches
# %%

# %%
# Merge fattura_data with invoice_matches on the specified columns
merged = fattura_data.merge(
    invoice_matches[["main_ddt_number", "descrizione"]],
    on=["main_ddt_number", "descrizione"],
    how="left",
    indicator=True,
)

# Select rows where there was no match in invoice_matches
not_in_invoice_matches = merged[merged["_merge"] == "left_only"]

# Display the result
# not_in_invoice_matches

# %%
# fattura_data
# %%

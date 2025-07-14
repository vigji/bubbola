# %%
from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET

from skrub import deduplicate, Joiner

def parse_items(xml_file):
    """
    Extract all <DettaglioLinee> entries (invoice line items)
    into a list of dicts, stripping any XML namespace.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    items = []
    # match any namespace: {any}DettaglioLinee
    for det in root.findall('.//{*}DettaglioLinee'):
        entry = {
            child.tag.split('}', 1)[-1]: child.text
            for child in det
        }
        items.append(entry)
    return items

def parse_items_df(results_dir):
    items_df = pd.read_csv(results_dir / "items_table.csv")
    items_df["ddt_delivery_date"] = items_df["ddt_delivery_date"].apply(lambda x: pd.to_datetime(x, format="%d/%m/%Y"))
    items_df["item_name_deduplicated"] = deduplicate(items_df["item_name"], n_clusters=30).values
    return items_df

def parse_correct_items(results_dir):
    correct_items = pd.read_csv(results_dir.parent.parent / "fatture" / "invoice_items.csv")
    correct_items = correct_items.rename({"Description": "item_name", "Quantity": "item_quantity", "DDTDate": "ddt_delivery_date", "UnitPrice": "item_unit_price", "Amount": "item_total_price"}, axis=1)
    for col in ["item_quantity", "item_unit_price", "item_total_price"]:
        correct_items[col] = correct_items[col].str.replace(",", ".").astype(float)
    correct_items["ddt_delivery_date"] = correct_items["ddt_delivery_date"].apply(lambda x: pd.to_datetime(x, format="%d/%m/%y"))

    return correct_items

def parse_correct_xml(xml_file):
    items_df = pd.DataFrame(parse_items(xml_file))
    items_df = items_df.rename({"Descrizione": "item_name", "Quantita": "item_quantity", "PrezzoUnitario": "item_unit_price", "PrezzoTotale": "item_total_price"}, axis=1)
    for col in ["item_quantity", "item_unit_price", "item_total_price"]:
        items_df[col] = items_df[col].str.replace(",", ".").astype(float)
    # items_df["ddt_delivery_date"] = items_df["ddt_delivery_date"].apply(lambda x: pd.to_datetime(x, format="%d/%m/%Y"))
    return items_df

# %%
# analysis for 1461
results_dir = Path("/Users/vigji/Desktop/pages_sample-data/concrete/1461/results/results_o4-mini_20250706_192000") 
items_df = parse_items_df(results_dir)
correct_items = parse_correct_items(results_dir)
# %%
# analysis for 1502
# results_dir = Path("/Users/vigji/Desktop/pages_sample-data/concrete/1502/results/o4-mini_20250707_181422") 
# items_df = parse_items_df(results_dir)
# correct_items = parse_correct_items(results_dir)
# # correct_items = parse_correct_xml(Path("/Users/vigji/Downloads/IT01378570350_jZW1y.xml"))

aux_table = correct_items[["item_name"]].copy()
# aux_table.columns = ["item_name_aux", "item_name"]
aux_table.drop_duplicates(inplace=True)
joined_table = items_df.copy()
print(aux_table.shape, joined_table.shape)
joiner = Joiner(
    aux_table,
    key="item_name",
    suffix="_aux",
    max_dist=0.8,
    add_match_info=False,
)

joined_table = joiner.fit_transform(joined_table)
joined_table.groupby("item_name_aux").agg({"item_quantity": "sum"})

# %%
correct_items.groupby("item_name").agg({"item_quantity": "sum"})
# %%
items_df.groupby("item_name").agg({"item_quantity": "sum"})
# %%
# Export both grouped results to different sheets in the same Excel file
correct_grouped = correct_items.groupby("item_name").agg({"item_quantity": "sum"})
extracted_grouped = items_df.groupby("item_name").agg({"item_quantity": "sum"})

output_file = results_dir / "quantity_comparison.xlsx"
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    correct_grouped.to_excel(writer, sheet_name='Correct_Items')
    extracted_grouped.to_excel(writer, sheet_name='Extracted_Items')

print(f"Exported to {output_file}")
# %%

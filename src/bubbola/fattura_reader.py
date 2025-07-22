import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

NAMESPACES = {
    "p": "http://ivaservizi.agenziaentrate.gov.it/docs/xsd/fatture/v1.2",
    "ds": "http://www.w3.org/2000/09/xmldsig#",
    "xsi": "http://www.w3.org/2001/XMLSchema-instance",
}


def parse_fattura_elettronica(
    xml_path: Path,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract CedentePrestatore DatiAnagrafici
    # Fix: CedentePrestatore is not namespaced in the sample XML
    dati_anagrafici = root.find(".//CedentePrestatore/DatiAnagrafici")
    if dati_anagrafici is None:
        raise ValueError("DatiAnagrafici not found")
    codice_fiscale = dati_anagrafici.findtext("CodiceFiscale", default="")
    denominazione = dati_anagrafici.findtext("Anagrafica/Denominazione", default="")
    cedente_json = {
        "CodiceFiscale": codice_fiscale,
        "Denominazione": denominazione,
    }

    # Extract DatiDDT cross-reference: {line_number: (NumeroDDT, DataDDT)}
    ddt_map = {}
    for ddt in root.findall(".//DatiDDT"):
        numero_ddt = ddt.findtext("NumeroDDT", default="")
        data_ddt = ddt.findtext("DataDDT", default="")
        for ref in ddt.findall("RiferimentoNumeroLinea"):
            line_number = ref.text
            ddt_map[line_number] = {"NumeroDDT": numero_ddt, "DataDDT": data_ddt}

    # Extract DettaglioLinee
    linee = []
    for dettaglio in root.findall(".//DettaglioLinee"):
        numero = dettaglio.findtext("NumeroLinea", default="")
        descrizione = dettaglio.findtext("Descrizione", default="")
        quantita = dettaglio.findtext("Quantita", default="")
        unita_misura = dettaglio.findtext("UnitaMisura", default="")
        prezzo = dettaglio.findtext("PrezzoUnitario", default="")
        sconto = ""
        sconto_elem = dettaglio.find("ScontoMaggiorazione/Importo")
        if sconto_elem is not None:
            sconto = sconto_elem.text or ""
        totale = dettaglio.findtext("PrezzoTotale", default="")
        aliquota = dettaglio.findtext("AliquotaIVA", default="")
        ddt_info = ddt_map.get(numero, {"NumeroDDT": "", "DataDDT": ""})
        linee.append(
            {
                "Numero": numero,
                "Descrizione": descrizione,
                "Quantita": quantita,
                "UnitaMisura": unita_misura,
                "PrezzoUnitario": prezzo,
                "Sconto": sconto,
                "Totale": totale,
                "AliquotaIVA": aliquota,
                "NumeroDDT": ddt_info["NumeroDDT"],
                "DataDDT": ddt_info["DataDDT"],
            }
        )

    return cedente_json, linee


def save_cedente_json(cedente_json: dict[str, Any], output_path: Path = None) -> str:
    json_str = json.dumps(cedente_json, ensure_ascii=False, indent=2)
    if output_path is not None:
        with output_path.open("w", encoding="utf-8") as f:
            f.write(json_str)
    return json_str


def save_linee_csv(linee: list[dict[str, Any]], output_path: Path = None) -> str:
    fieldnames = [
        "Numero",
        "Descrizione",
        "Quantita",
        "UnitaMisura",
        "PrezzoUnitario",
        "Sconto",
        "Totale",
        "AliquotaIVA",
        "NumeroDDT",
        "DataDDT",
    ]
    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in linee:
        writer.writerow(row)
    csv_str = output.getvalue()
    if output_path is not None:
        with output_path.open("w", encoding="utf-8", newline="") as f:
            f.write(csv_str)
    return csv_str


# Convenience functions for direct string output


def cedente_json_str(xml_path: Path) -> str:
    cedente_json, _ = parse_fattura_elettronica(xml_path)
    return save_cedente_json(cedente_json)


def linee_csv_str(xml_path: Path) -> str:
    _, linee = parse_fattura_elettronica(xml_path)
    return save_linee_csv(linee)


if __name__ == "__main__":
    from pprint import pprint

    # Example usage:
    data = parse_fattura_elettronica(
        Path("/Users/vigji/Downloads/IT01378570350_jZW1y.xml")
    )
    cedente_json, linee = data

    pprint(cedente_json)
    pprint(linee)

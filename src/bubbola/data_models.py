from pydantic import BaseModel, Field

measurement_unit_description = """
Unit of capacity of the item. Found in column 'units', 'unità', 'UM', etc, has to be converted to lower case.
It is usually - but not always - in metric units.
DO NOT ASSUME if not found, unless the item is concrete. For concrete, assume M3
"""


class DeliveryItem(BaseModel):
    item_name: str | None = Field(
        default=None,
        description="Name or description of the item as found in the note. Compact in a single line, without line breaks.",
    )  # name or description of the item as found in the note
    item_code: str | None = Field(
        default=None, description="Code of the item as found in the note. Not order"
    )  # code of the item as found in the note. Not order
    item_quantity: float | None = None
    item_measurement_unit: str | None = Field(
        description=measurement_unit_description
    )  # Literal["kg", "m3", "m3g", "g", "l", "ml", "t", "none", ":"] | None = Field(default=None, description="Unit of capacity of the item. Found in column 'units', 'unità', 'UM', etc, to be converted to lower case. DO NOT ASSUME if not found!")
    # item_capacity: float | None = None
    # item_capacity_measurement_unit: str | None = Field(
    #     description=measurement_unit_description
    # )  # Literal["kg", "m3", "m3g", "g", "l", "ml", "t", "none", ":"] | None = Field(default=None, description="Unit of capacity of the item. Found in column 'units', 'unità', 'UM', etc, to be converted to lower case. DO NOT ASSUME!")
    item_unit_price: float | None = None
    item_total_price: float | None = None
    not_in_valid_items: bool | None = None


class DeliveryNote(BaseModel):
    nome_rag_1: str | None = Field(
        default=None,
        description="Name of the supplier, as found in the supplier list. Has to match identically the name in the supplier list.",
    )  # Name of the supplier
    b501a_num_doc: str | None = Field(
        default=None,
        description="Order number, as found in the note, only if it matches the order number in the supplier list. If same number is found in multiple places with numbering after ., keep longest number found  (eg: 1200 and 1200.1, keep 1200.1)",
    )  # Order number
    ddt_number: str
    delivery_date: str | None = Field(
        default=None,
        description="Date of the delivery, as found in the note. Has to be after 2024.",
    )  # Date of the delivery
    delivery_items: list[DeliveryItem] | None = None
    fisso_pompa_nastro: int
    m3_pompa_nastro: int

    summary: str | None = Field(
        default=None,
        description="A short summary of your reasoning, <100 words>",
    )  # Summary of your reasoning


class DeliveryItemFatturaMatch(BaseModel):
    match_in_fattura: bool | None = Field(
        default=None,
        description="True if the item is present in the invoice, False otherwise.",
    )
    match_in_ddt: bool | None = Field(
        default=None,
        description="True if the item is present in the transportation document, False otherwise.",
    )
    descrizione: str | None = Field(
        default=None,
        description="Name of the item.",
    )
    numero_linea_fattura: str | None = Field(
        default=None,
        description="Number of the line in the invoice.",
    )
    quantita_in_fattura: float | None = Field(
        default=None,
        description="Quantity of the item as found in the invoice (if match_in_fattura is True).",
    )
    quantita_in_ddt: float | None = Field(
        default=None,
        description="Quantity of the item as found in the transportation document (if match_in_ddt is True).",
    )
    unita_misura: str | None = Field(
        default=None,
        description="Unit of measurement of the item.",
    )


class DeliveryNoteFatturaMatch(BaseModel):
    ddt_number: str | None = Field(
        default=None,
        description="DDT number as found in the transportation document.",
    )
    ddt_date: str | None = Field(
        default=None,
        description="DDT date as found in the transportation document.",
    )
    items: list[DeliveryItemFatturaMatch] | None = None


class ImageDescription(BaseModel):
    description: str | None = Field(
        default=None,
        description="A short description of the image, <20 words>",
    )  # Summary of your reasoning
    contrast: int | None = Field(
        default=None,
        description="A rating of the contrast of the image, from 1 to 10",
    )  # Rating of the contrast

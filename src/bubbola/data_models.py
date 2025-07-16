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

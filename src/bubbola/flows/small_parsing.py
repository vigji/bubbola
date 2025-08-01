from typing import Any

flow = {
    "data_model": "ImageDescription",
    "system_prompt": "You are a helpful assistant that provides a short description of the image and a rating of the contrast of the image; provide your response in structured JSON format, with the following fields: description (string, <20 words), contrast (integer, from 1 to 10)",
    "model_name": "mistralai/mistral-small-3.2-24b-instruct:free",
    "description": "Provide a short description of the image",
    "external_file_options": {},
    "model_kwargs": {},
    "parser_kwargs": {},
    "max_edge_size": 1000,
}


def get_flow() -> dict[str, Any]:
    return flow

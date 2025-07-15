# from main import DeliveryNote
import json

file_to_parse = "/Users/vigji/Desktop/pages_sample-data/concrete/results_llama-4-scout_20250706_170359/response_20250512125121_013.json"
# file_to_parse = "/Users/vigji/Desktop/pages_sample-data/concrete/results_llama-4-scout_20250706_170359/response_20250512125121_012.json"

with open(file_to_parse) as f:
    data = json.load(f)

print(data)

# #DeliveryNote.model_validate_json(data)

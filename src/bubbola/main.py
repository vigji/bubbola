from pydantic import BaseModel, Field
from pathlib import Path
from ai_bubbles.pydantic_model.base import DeliveryNote
from ai_bubbles.model_creator import get_client_response_function
from datetime import datetime
import json
import base64
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading
import signal
import time
from load_results import create_results_csv

from image_data_loader import sanitize_to_images

# Global lock for thread-safe token counting
token_lock = threading.Lock()

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global shutdown_requested
    print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
    shutdown_requested = True

# Set up signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# for testing, "meta-llama/llama-4-scout"


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


valid_suppliers_list = [["Nome", "Indirizzo", "Partita IVA"],
    ["LG Concrete S.r.L.", "33050 Castions di Strada (UD) Via Udine,104", "03089360303"]]

valid_suppliers_csv_text = "\n".join([",".join(row) for row in valid_suppliers_list])
valid_suppliers = [row[0] for row in valid_suppliers_list]

prices_csv_text = """nome,codice,costo/m3
                    Cemento C12/15,C12/15,68,00
                    Cemento C16/20,C16/20,71,00
                    Cemento C20/25,C20/25,73,50
                    Calcestruzzo plastico (mix rif. 060 LG/23),060 LG/23,86,00
                    Additivo accelerante,ACCELERANTE,7,00"""

system_prompt = f"""
You are a helpful assistant in ICOP SpA accounting department.
Your task is to automatically extract structured data from a delivery note image related
to material that a supplier has delivered to an ICOP construction site.

You receive:
1. A **CSV file** containing a list of suppliers.
2. A **photo of a delivery note** from a construction site.
3. A **CSV file** containing a list of prices for possible items in the delivery note.

Here is the list of suppliers:

```
{valid_suppliers_csv_text}
```

Here is the list of prices:

```
{prices_csv_text}
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




def process_single_image(name, base64_image, results_dir, system_prompt, response_function, response_format, model_name, timeout=60, max_retries=5):
    """Process a single image and return the token counts and retry stats."""
    kwargs = {
        "model": model_name,
        "response_format": response_format,
        "temperature": 0.0,
    }

    #if model_name == "gpt-4o-mini" or model_name == "meta-llama/llama-4-scout":
    #    kwargs["temperature"] = 0.0

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        },
    ]
    kwargs.update(messages=messages)
    
    total_input_tokens = 0
    total_output_tokens = 0
    retry_count = 0
    retry_input_tokens = 0
    retry_output_tokens = 0
    
    for attempt in range(max_retries + 1):
        try:
            response = response_function(**kwargs)
            
            # Track tokens for this attempt
            attempt_input_tokens = response.usage.prompt_tokens
            attempt_output_tokens = response.usage.completion_tokens
            total_input_tokens += attempt_input_tokens
            total_output_tokens += attempt_output_tokens
            
            # Check if response is empty and validate with Pydantic model
            response_content = response.choices[0].message.content
            is_valid_response = False
            
            if response_content and response_content.strip():
                try:
                    # Try to parse the JSON and validate with Pydantic model
                    parsed_json = json.loads(response_content)
                    
                    # Check if response is actually the schema instead of data
                    if isinstance(parsed_json, dict) and ("$defs" in parsed_json or "properties" in parsed_json):
                        print(f"Response contains schema instead of data for {name}, attempt {attempt + 1}. Retrying...")
                        is_valid_response = False
                    else:
                        DeliveryNote.model_validate(parsed_json)
                        is_valid_response = True
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Invalid JSON or Pydantic validation failed for {name}, attempt {attempt + 1}: {e}")
                    is_valid_response = False
            
            if is_valid_response:
                # Valid response, save and return
                with open(results_dir / f"response_{name}.json", "w") as f:
                    json.dump(response_content, f)
                
                # save also copy of the image with the response
                with open(results_dir / f"response_{name}.png", "wb") as f:
                    f.write(base64.b64decode(base64_image))
                
                return total_input_tokens, total_output_tokens, retry_count, retry_input_tokens, retry_output_tokens
            else:
                # Invalid or empty response, prepare for retry
                if attempt < max_retries:
                    retry_count += 1
                    retry_input_tokens += attempt_input_tokens
                    retry_output_tokens += attempt_output_tokens
                    if not response_content or not response_content.strip():
                        print(f"Empty response for {name}, attempt {attempt + 1}/{max_retries + 1}. Retrying...")
                    else:
                        print(f"Invalid response for {name}, attempt {attempt + 1}/{max_retries + 1}. Retrying...")
                else:
                    # Max retries reached, save invalid response
                    print(f"Max retries reached for {name}. Saving invalid response.")
                    with open(results_dir / f"response_{name}.json", "w") as f:
                        json.dump(response_content if response_content else "", f)
                    
                    # save also copy of the image with the response
                    with open(results_dir / f"response_{name}.png", "wb") as f:
                        f.write(base64.b64decode(base64_image))
                    
                    return total_input_tokens, total_output_tokens, retry_count, retry_input_tokens, retry_output_tokens
                    
        except Exception as e:
            print(f"API call failed for {name} on attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                raise
            # For non-final attempts, track tokens if response object exists
            if 'response' in locals():
                attempt_input_tokens = response.usage.prompt_tokens
                attempt_output_tokens = response.usage.completion_tokens
                total_input_tokens += attempt_input_tokens
                total_output_tokens += attempt_output_tokens
                retry_count += 1
                retry_input_tokens += attempt_input_tokens
                retry_output_tokens += attempt_output_tokens

def process_images_parallel(to_process, results_dir, system_prompt, response_function, response_format, model_name, max_workers=10, timeout=300, max_retries=5):
    """Process images in parallel and return total token counts and retry statistics."""
    global shutdown_requested
    total_input_tokens = 0
    total_output_tokens = 0
    total_retry_count = 0
    total_retry_input_tokens = 0
    total_retry_output_tokens = 0
    failed_images = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_name = {
            executor.submit(
                process_single_image, 
                name, 
                base64_image, 
                results_dir, 
                system_prompt, 
                response_function, 
                response_format, 
                model_name,
                timeout,
                max_retries
            ): name 
            for name, base64_image in to_process.items()
        }
        
        # Process completed tasks with progress bar and timeout
        completed_count = 0
        total_tasks = len(future_to_name)
        
        with tqdm(total=total_tasks, desc="Processing images") as pbar:
            try:
                for future in as_completed(future_to_name, timeout=timeout * len(future_to_name)):
                    if shutdown_requested:
                        print("\nShutdown requested, cancelling remaining tasks...")
                        break
                        
                    name = future_to_name[future]
                    try:
                        input_tokens, output_tokens, retry_count, retry_input_tokens, retry_output_tokens = future.result(timeout=timeout)
                        with token_lock:
                            total_input_tokens += input_tokens
                            total_output_tokens += output_tokens
                            total_retry_count += retry_count
                            total_retry_input_tokens += retry_input_tokens
                            total_retry_output_tokens += retry_output_tokens
                        completed_count += 1
                        pbar.update(1)
                    except TimeoutError:
                        print(f"Timeout processing image {name}")
                        failed_images.append(name)
                        pbar.update(1)
                    except Exception as exc:
                        print(f"Image {name} generated an exception: {exc}")
                        failed_images.append(name)
                        pbar.update(1)
            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected during processing")
                shutdown_requested = True
        
        # Cancel any remaining futures
        for future in future_to_name:
            if not future.done():
                future.cancel()
                if shutdown_requested:
                    print(f"Cancelled processing for {future_to_name[future]}")
    
    if failed_images:
        print(f"Failed to process {len(failed_images)} images: {failed_images}")
    
    if shutdown_requested:
        print(f"Processing interrupted. Completed {completed_count}/{total_tasks} images.")
    
    return total_input_tokens, total_output_tokens, total_retry_count, total_retry_input_tokens, total_retry_output_tokens

#system_prompt = "Extract items from the delivery note, following this schema: {DeliveryNote.model_json_schema()}"

if __name__ == "__main__":
    data_dir = Path("/Users/vigji/Desktop/pages_sample-data/concrete")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    commessa = "1502"
    bolle_dir = data_dir / commessa / "bolle"
    files_to_process = sorted(list(bolle_dir.glob("*.pdf")))# [0]
    # print(sample_file)
    # assert False

    start_time = datetime.now()

    to_process = sanitize_to_images(files_to_process, return_as_base64=True, 
                                    max_edge_size=1000)
    model_name =  "o4-mini"  # "o4-mini"  #  "o3"  # "meta-llama/llama-4-scout"  # "o3"  #  "o4-mini" #"meta-llama/llama-4-scout"  # "o4-mini"  #  "gpt-4o-mini"  # "test"  # Change this to your desired model

    N_TO_PROCESS = None  # 13  #  # 13  # len(to_process)  # 1

    if N_TO_PROCESS is None:
        N_TO_PROCESS = len(to_process)
    print(f"N_TO_PROCESS: {N_TO_PROCESS}")

    results_dir = data_dir / commessa / "results" / f"{model_name.split('/')[-1]}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Process each image
    response_function, response_format = get_client_response_function(
        model_name, DeliveryNote
    )

    # Parse command line arguments for parallel processing
    parser = argparse.ArgumentParser(description='Process delivery notes with parallel requests')
    parser.add_argument('--max-workers', type=int, default=10, help='Maximum number of parallel requests (default: 10)')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds for each request (default: 300)')
    parser.add_argument('--max-retries', type=int, default=5, help='Maximum number of retries for empty responses (default: 5)')
    args, unknown = parser.parse_known_args()

    # Process images in parallel
    items_to_process = dict(list(to_process.items())[:N_TO_PROCESS])
    total_input_tokens, total_output_tokens, total_retry_count, total_retry_input_tokens, total_retry_output_tokens = process_images_parallel(
        items_to_process, 
        results_dir, 
        system_prompt, 
        response_function, 
        response_format, 
        model_name, 
        max_workers=args.max_workers,
        timeout=args.timeout,
        max_retries=args.max_retries
    )

    end_time = datetime.now()
    print(f"Time taken: {(end_time - start_time).total_seconds()} seconds")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total retry attempts: {total_retry_count}")
    print(f"Retry input tokens: {total_retry_input_tokens}")
    print(f"Retry output tokens: {total_retry_output_tokens}")
    if total_input_tokens > 0:
        print(f"Retry percentage of total tokens: {((total_retry_input_tokens + total_retry_output_tokens) / (total_input_tokens + total_output_tokens) * 100):.1f}%")

    with open(results_dir / f"batch_log_{timestamp}.txt", "w") as f:
        json.dump(
            {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_retry_count": total_retry_count,
                "total_retry_input_tokens": total_retry_input_tokens,
                "total_retry_output_tokens": total_retry_output_tokens,
                "time_taken": (end_time - start_time).total_seconds(),
                "max_workers": args.max_workers,
                "max_retries": args.max_retries,
            },
            f,
        )

    print(f"\nAll images processed. Results saved in {results_dir}")

    ddts_data, items_data = create_results_csv(results_dir)
    print(f"DDTs: {len(ddts_data)}")
    print(f"Items: {len(items_data)}")


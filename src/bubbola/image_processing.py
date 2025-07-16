import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from bubbola.data_models import DeliveryNote
from bubbola.price_estimates import (
    AggregatedTokenCounts,
    TokenCounts,
    estimate_max_output_tokens_from_schema,
    estimate_total_tokens_number,
    get_cost_estimate,
)

# Global variables for parallel processing
token_lock = threading.Lock()
shutdown_requested = False


def _validate_response_content(
    response_content: str, image_name: str, attempt: int
) -> bool:
    """Validate response content and return True if valid, False otherwise."""
    if not response_content or not response_content.strip():
        return False

    try:
        # Try to parse the JSON and validate with Pydantic model
        parsed_json = json.loads(response_content)

        # Check if response is actually the schema instead of data
        if isinstance(parsed_json, dict) and (
            "$defs" in parsed_json or "properties" in parsed_json
        ):
            print(
                f"Response contains schema instead of data for {image_name}, attempt {attempt + 1}. Retrying..."
            )
            return False
        else:
            DeliveryNote.model_validate(parsed_json)
            return True
    except (json.JSONDecodeError, ValueError) as e:
        print(
            f"Invalid JSON or Pydantic validation failed for {image_name}, attempt {attempt + 1}: {e}"
        )
        return False


def _save_response_files(results_dir, image_name, response_content, base64_image):
    """Save response JSON and image files."""
    with open(results_dir / f"response_{image_name}.json", "w") as f:
        json.dump(response_content if response_content else "", f)

    with open(results_dir / f"response_{image_name}.png", "wb") as f:
        f.write(base64.b64decode(base64_image))


def _process_single_image_dry_run(
    name, base64_image, system_prompt, model_name, max_output_tokens
):
    """Process a single image in dry_run mode - estimate tokens and costs without API calls."""
    # Estimate input tokens (system prompt + image)
    estimated_input_tokens = estimate_total_tokens_number(
        system_prompt, base64_image, model_name
    )

    # Estimate output tokens (use max_output_tokens if provided, otherwise estimate)
    estimated_output_tokens = max_output_tokens if max_output_tokens else 1000

    # Calculate cost estimate
    cost_estimate = get_cost_estimate(
        model_name, estimated_input_tokens, estimated_output_tokens
    )

    # Create token counts object (no retries in dry_run)
    token_counts = TokenCounts()
    token_counts.total_input_tokens = estimated_input_tokens
    token_counts.total_output_tokens = estimated_output_tokens

    # Print dry run information
    print(f"DRY RUN - {name}:")
    print(f"  Estimated input tokens: {estimated_input_tokens}")
    print(f"  Estimated output tokens: {estimated_output_tokens}")
    if cost_estimate is not None:
        print(f"  Estimated cost: ${cost_estimate:.6f}")
    else:
        print(f"  Cost estimate: Not available for model {model_name}")
    print()

    return token_counts


def process_single_image(
    base64_image: str,
    results_dir,
    system_prompt,
    response_function,
    response_scheme,
    model_name,
    image_name: str,
    timeout=60,
    max_retries=5,
    temperature=None,
    dry_run=False,
):
    """Process a single image and return the token counts and retry stats."""

    # Estimate max_output_tokens from response format if not provided
    max_output_tokens = estimate_max_output_tokens_from_schema(
        response_scheme.model_json_schema()
    )

    if dry_run:
        return _process_single_image_dry_run(
            image_name, base64_image, system_prompt, model_name, max_output_tokens
        )

    if temperature is None:
        if model_name == "o3" or model_name == "o4-mini":
            temperature = 1
        else:
            temperature = 0.0

    kwargs = {
        "model": model_name,
        "temperature": temperature,
        # "max_completion_tokens": max_output_tokens,
    }

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

    token_counts = TokenCounts()

    for attempt in range(max_retries + 1):
        try:
            response = response_function(**kwargs)

            # For the new .parse interface, the parsed data is in response.choices[0].message.parsed
            # and the raw content is in response.choices[0].message.content
            if hasattr(response.choices[0].message, "parsed"):
                # New .parse interface - we have structured data
                parsed_data = response.choices[0].message.parsed
                response_content = response.choices[0].message.content

                # Validate that we got valid structured data
                if parsed_data is not None:
                    # Convert parsed data back to JSON for saving
                    response_content = response_scheme.model_dump_json(parsed_data)
                    is_valid_response = True
                else:
                    is_valid_response = False
            else:
                # Fallback to old interface
                response_content = response.choices[0].message.content
                is_valid_response = _validate_response_content(
                    response_content, image_name, attempt
                )

            token_counts.add_attempt(response, is_retry=False)

            if is_valid_response:
                # Valid response, save files and return
                _save_response_files(
                    results_dir, image_name, response_content, base64_image
                )

                return token_counts
            else:
                # Invalid response, prepare for retry or final save
                if attempt < max_retries:
                    # Update retry statistics
                    token_counts.add_attempt(response, is_retry=True)
                    print(
                        f"Invalid response for {image_name}, attempt {attempt + 1}/{max_retries + 1}. Retrying..."
                    )
                else:
                    # Max retries reached, save invalid response
                    print(
                        f"Max retries reached for {image_name}. Saving invalid response."
                    )
                    _save_response_files(
                        results_dir, image_name, response_content, base64_image
                    )
                    return token_counts

        except Exception as e:
            print(
                f"API call failed for image '{image_name}' on attempt {attempt + 1}: {e}"
            )
            # if attempt == max_retries:
            #     raise

            # # For non-final attempts, track tokens if response object exists
            # if "response" in locals():
            #     token_counts.add_attempt(response, is_retry=True)


def process_images_parallel(
    to_process,
    results_dir,
    system_prompt,
    response_function,
    response_scheme,
    model_name,
    max_workers=10,
    timeout=300,
    max_retries=5,
    temperature=None,
    dry_run=False,
    max_output_tokens=None,
):
    """Process images in parallel and return total token counts and retry statistics."""
    global shutdown_requested
    aggregated_token_counts = AggregatedTokenCounts()
    failed_images = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_name = {
            executor.submit(
                process_single_image,
                base64_image,
                results_dir,
                system_prompt,
                response_function,
                response_scheme,
                model_name,
                name,
                timeout,
                max_retries,
                temperature,
                dry_run,
                max_output_tokens,
            ): name
            for name, base64_image in to_process.items()
        }

        # Process completed tasks with progress bar and timeout
        completed_count = 0
        total_tasks = len(future_to_name)

        with tqdm(total=total_tasks, desc="Processing images") as pbar:
            try:
                for future in as_completed(
                    future_to_name, timeout=timeout * len(future_to_name)
                ):
                    if shutdown_requested:
                        print("\nShutdown requested, cancelling remaining tasks...")
                        break

                    name = future_to_name[future]
                    try:
                        token_counts = future.result(timeout=timeout)
                        with token_lock:
                            aggregated_token_counts.add_token_counts(token_counts)
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
        print(
            f"Processing interrupted. Completed {completed_count}/{total_tasks} images."
        )

        # Print aggregated statistics
    aggregated_token_counts.print_summary()

    if dry_run:
        total_cost = get_cost_estimate(
            model_name,
            aggregated_token_counts.total_input_tokens,
            aggregated_token_counts.total_output_tokens,
        )
        print("\nDRY RUN SUMMARY:")
        if total_cost is not None:
            print(
                f"Total estimated cost: ${total_cost:.6f} ({total_cost / len(to_process):.6f} per page)"
            )
        else:
            print(f"Total cost estimate: Not available for model {model_name}")

    return aggregated_token_counts


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    from bubbola.image_data_loader import sanitize_to_images
    from bubbola.main import system_prompt
    from bubbola.model_creator import get_client_response_function

    image_path = "/Users/vigji/code/bubbola/tests/assets/single_pages/0088_001_001.png"
    model_name = "gpt-4o"  # "o4-mini"

    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir)

        base64_image = sanitize_to_images(image_path, return_as_base64=True)
        image_name, image = next(iter(base64_image.items()))
        print(image_name)

        response_function, response_scheme = get_client_response_function(
            model_name, DeliveryNote
        )

        process_single_image(
            base64_image=image,
            results_dir=results_dir,
            system_prompt=system_prompt,
            response_function=response_function,
            response_scheme=response_scheme,
            model_name=model_name,
            image_name=image_name,
            dry_run=False,
        )

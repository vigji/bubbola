import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from bubbola.data_models import DeliveryNote

# Global variables for parallel processing
token_lock = threading.Lock()
shutdown_requested = False


def process_single_image(
    name,
    base64_image,
    results_dir,
    system_prompt,
    response_function,
    response_format,
    model_name,
    timeout=60,
    max_retries=5,
):
    """Process a single image and return the token counts and retry stats."""
    kwargs = {
        "model": model_name,
        "response_format": response_format,
        "temperature": 0.0,
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
                    if isinstance(parsed_json, dict) and (
                        "$defs" in parsed_json or "properties" in parsed_json
                    ):
                        print(
                            f"Response contains schema instead of data for {name}, attempt {attempt + 1}. Retrying..."
                        )
                        is_valid_response = False
                    else:
                        DeliveryNote.model_validate(parsed_json)
                        is_valid_response = True
                except (json.JSONDecodeError, ValueError) as e:
                    print(
                        f"Invalid JSON or Pydantic validation failed for {name}, attempt {attempt + 1}: {e}"
                    )
                    is_valid_response = False

            if is_valid_response:
                # Valid response, save and return
                with open(results_dir / f"response_{name}.json", "w") as f:
                    json.dump(response_content, f)

                # save also copy of the image with the response
                with open(results_dir / f"response_{name}.png", "wb") as f:
                    f.write(base64.b64decode(base64_image))

                return (
                    total_input_tokens,
                    total_output_tokens,
                    retry_count,
                    retry_input_tokens,
                    retry_output_tokens,
                )
            else:
                # Invalid or empty response, prepare for retry
                if attempt < max_retries:
                    retry_count += 1
                    retry_input_tokens += attempt_input_tokens
                    retry_output_tokens += attempt_output_tokens
                    if not response_content or not response_content.strip():
                        print(
                            f"Empty response for {name}, attempt {attempt + 1}/{max_retries + 1}. Retrying..."
                        )
                    else:
                        print(
                            f"Invalid response for {name}, attempt {attempt + 1}/{max_retries + 1}. Retrying..."
                        )
                else:
                    # Max retries reached, save invalid response
                    print(f"Max retries reached for {name}. Saving invalid response.")
                    with open(results_dir / f"response_{name}.json", "w") as f:
                        json.dump(response_content if response_content else "", f)

                    # save also copy of the image with the response
                    with open(results_dir / f"response_{name}.png", "wb") as f:
                        f.write(base64.b64decode(base64_image))

                    return (
                        total_input_tokens,
                        total_output_tokens,
                        retry_count,
                        retry_input_tokens,
                        retry_output_tokens,
                    )

        except Exception as e:
            print(f"API call failed for {name} on attempt {attempt + 1}: {e}")
            if attempt == max_retries:
                raise
            # For non-final attempts, track tokens if response object exists
            if "response" in locals():
                attempt_input_tokens = response.usage.prompt_tokens
                attempt_output_tokens = response.usage.completion_tokens
                total_input_tokens += attempt_input_tokens
                total_output_tokens += attempt_output_tokens
                retry_count += 1
                retry_input_tokens += attempt_input_tokens
                retry_output_tokens += attempt_output_tokens


def process_images_parallel(
    to_process,
    results_dir,
    system_prompt,
    response_function,
    response_format,
    model_name,
    max_workers=10,
    timeout=300,
    max_retries=5,
):
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
                max_retries,
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
                        (
                            input_tokens,
                            output_tokens,
                            retry_count,
                            retry_input_tokens,
                            retry_output_tokens,
                        ) = future.result(timeout=timeout)
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
        print(
            f"Processing interrupted. Completed {completed_count}/{total_tasks} images."
        )

    return (
        total_input_tokens,
        total_output_tokens,
        total_retry_count,
        total_retry_input_tokens,
        total_retry_output_tokens,
    )

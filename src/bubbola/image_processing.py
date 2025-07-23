import base64
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from bubbola.data_models import DeliveryNote
from bubbola.model_creator import get_model_client
from bubbola.price_estimates import (
    AggregatedTokenCounts,
    TokenCounts,
    get_cost_estimate,
)


class ImageProcessor:
    """Handles processing of individual images using the new model class interface."""

    def __init__(self, model_name: str, pydantic_model: type, results_dir: Path):
        self.model = get_model_client(model_name)
        self.pydantic_model = pydantic_model
        self.results_dir = results_dir

    def _save_response_files(
        self, image_name: str, response_content: str, base64_image: str
    ):
        with open(self.results_dir / f"response_{image_name}.json", "w") as f:
            f.write(response_content)
        with open(self.results_dir / f"response_{image_name}.png", "wb") as f:
            f.write(base64.b64decode(base64_image))

    def process_single_image(
        self,
        image_name: str,
        base64_image: str,
        system_prompt: str,
        model_kwargs: dict | None = None,
        parser_kwargs: dict | None = None,
        dry_run: bool = False,
    ) -> TokenCounts:
        """Process a single image and return token counts. Handles validation via LLMModel."""
        model_kwargs = model_kwargs or {}
        parser_kwargs = parser_kwargs or {}
        if dry_run:
            return self._process_single_image_dry_run(
                image_name, base64_image, system_prompt
            )

        # Extract parser-specific arguments
        max_n_retries = parser_kwargs.pop("max_n_retries", 5)
        required_true_fields = parser_kwargs.pop("require_true_fields", None)

        # Any remaining parser_kwargs should be model arguments
        model_kwargs.update(parser_kwargs)

        messages = self.model.create_messages(
            instructions=system_prompt, images=[base64_image]
        )
        parsed_response, token_counts, fully_validated = self.model.get_parsed_response(
            messages=messages,
            pydantic_model=self.pydantic_model,
            max_n_retries=max_n_retries,
            required_true_fields=required_true_fields,
            dry_run=False,
            **model_kwargs,
        )
        if parsed_response is not None:
            response_content = self.pydantic_model.model_dump_json(parsed_response)
        else:
            response_content = ""
        self._save_response_files(image_name, response_content, base64_image)
        return token_counts

    def _process_single_image_dry_run(
        self, image_name: str, base64_image: str, system_prompt: str
    ) -> TokenCounts:
        messages = self.model.create_messages(
            instructions=system_prompt, images=[base64_image]
        )
        _, token_counts, _ = self.model.get_parsed_response(
            messages=messages, pydantic_model=self.pydantic_model, dry_run=True
        )
        cost_estimate = get_cost_estimate(
            self.model.name,
            token_counts.total_input_tokens,
            token_counts.total_output_tokens,
        )
        print(f"DRY RUN - {image_name}:")
        print(f"  Estimated input tokens: {token_counts.total_input_tokens}")
        print(f"  Estimated output tokens: {token_counts.total_output_tokens}")
        if cost_estimate is not None:
            print(f"  Estimated cost: ${cost_estimate:.6f}")
        else:
            print(f"  Cost estimate: Not available for model {self.model.name}")
        print()
        return token_counts


class ParallelImageProcessor:
    """Handles parallel processing of multiple images with progress tracking and error handling."""

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.shutdown_requested = False
        self.token_lock = threading.Lock()

    def process_batch(
        self,
        to_process: dict[str, str],
        system_prompt: str,
        model_name: str,
        pydantic_model: type,
        results_dir: Path,
        model_kwargs: dict | None = None,
        parser_kwargs: dict | None = None,
        dry_run: bool = False,
        timeout: int = 300,
    ) -> AggregatedTokenCounts:
        model_kwargs = model_kwargs or {}
        parser_kwargs = parser_kwargs or {}
        aggregated_token_counts = AggregatedTokenCounts()
        failed_images = []
        processor = ImageProcessor(model_name, pydantic_model, results_dir)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_name = {
                executor.submit(
                    processor.process_single_image,
                    name,
                    base64_image,
                    system_prompt,
                    model_kwargs,
                    parser_kwargs,
                    dry_run,
                ): name
                for name, base64_image in to_process.items()
            }
            completed_count = 0
            total_tasks = len(future_to_name)
            with tqdm(total=total_tasks, desc="Processing images") as pbar:
                try:
                    for future in as_completed(
                        future_to_name, timeout=timeout * len(future_to_name)
                    ):
                        if self.shutdown_requested:
                            print("\nShutdown requested, cancelling remaining tasks...")
                            break
                        name = future_to_name[future]
                        try:
                            token_counts = future.result(timeout=timeout)
                            with self.token_lock:
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
                    self.shutdown_requested = True
            for future in future_to_name:
                if not future.done():
                    future.cancel()
                    if self.shutdown_requested:
                        print(f"Cancelled processing for {future_to_name[future]}")
        if failed_images:
            print(f"Failed to process {len(failed_images)} images: {failed_images}")
        if self.shutdown_requested:
            print(
                f"Processing interrupted. Completed {completed_count}/{total_tasks} images."
            )
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

    from bubbola.image_data_loader import sanitize_to_images
    from scripts.main import system_prompt

    image_path = "/Users/vigji/code/bubbola/tests/assets/single_pages/0088_001_001.png"
    model_name = "gpt-4o"

    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir)
        base64_image = sanitize_to_images(image_path, return_as_base64=True)
        image_name, image = next(iter(base64_image.items()))
        print(image_name)
        processor = ImageProcessor(model_name, DeliveryNote, results_dir)
        token_counts = processor.process_single_image(
            image_name=image_name,
            base64_image=image,
            system_prompt=system_prompt,
            dry_run=False,
        )
        print(
            f"Processed {image_name} with {token_counts.total_input_tokens} input tokens and {token_counts.total_output_tokens} output tokens"
        )

    # test parallel processing
    multipage_docs = [
        "/Users/vigji/code/bubbola/tests/assets/0088_001.pdf",
        "/Users/vigji/code/bubbola/tests/assets/0089_001.pdf",
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir)
        base64_images = sanitize_to_images(multipage_docs, return_as_base64=True)
        processor = ParallelImageProcessor(max_workers=10)
        token_counts = processor.process_batch(
            to_process=base64_images,
            system_prompt=system_prompt,
            model_name=model_name,
            pydantic_model=DeliveryNote,
            results_dir=results_dir,
        )

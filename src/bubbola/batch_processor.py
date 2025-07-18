"""Batch processing functionality for sanitized images."""

import json
import signal
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from bubbola.data_models import DeliveryNote, ImageDescription
from bubbola.image_data_loader import sanitize_to_images
from bubbola.image_processing import ParallelImageProcessor
from bubbola.results_converter import create_results_csv


class ProcessingFlow:
    """Represents a processing flow configuration."""

    def __init__(
        self,
        name: str,
        data_model: type[BaseModel],
        system_prompt: str,
        model_name: str,
        description: str = "",
        external_file_options: dict[str, str] | None = None,
    ):
        self.name = name
        self.data_model = data_model
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.description = description
        self.external_file_options = external_file_options or {}


class BatchProcessor:
    """Handles batch processing of sanitized images with configurable flows."""

    def __init__(self):
        # Available data models
        self.available_models = {
            "DeliveryNote": DeliveryNote,
            "ImageDescription": ImageDescription,
        }

        # Global flag for graceful shutdown
        self.shutdown_requested = False
        self.token_lock = threading.Lock()

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum: int, frame) -> None:
        """Handle interrupt signals gracefully."""
        print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
        self.shutdown_requested = True

    def _flow_config_to_processing_flow(
        self, flow_config: dict[str, Any], external_files: dict[str, Path] | None = None
    ) -> ProcessingFlow:
        """Convert a flow configuration dictionary to a ProcessingFlow object."""
        return ProcessingFlow(
            name=flow_config["name"],
            data_model=self.available_models[flow_config["data_model"]],
            system_prompt=flow_config["system_prompt"],
            model_name=flow_config["model_name"],
            description=flow_config["description"],
            external_file_options=flow_config.get("external_file_options"),
        )

    def list_flows(self) -> list[ProcessingFlow]:
        """List all available processing flows."""
        from bubbola.flows import get_flows

        flows = []
        flows_dict = get_flows()

        for module_name, flow_config in flows_dict.items():
            try:
                # The flow_config is already the result of get_flow() (no external files)
                flow = self._flow_config_to_processing_flow(flow_config)
                flows.append(flow)
            except Exception as e:
                print(f"Warning: Could not load flow {module_name}: {e}")

        return flows

    def get_flow(
        self, flow_name: str, external_files: dict[str, Path] | None = None
    ) -> ProcessingFlow | None:
        """Get a specific processing flow by name."""
        from bubbola.flows import get_flows

        flows_dict = get_flows()

        # First try exact match by module name
        if flow_name in flows_dict:
            flow_config = flows_dict[flow_name]
            try:
                # If we have external files, we need to call get_flow() with them
                if external_files:
                    # Import the module and call get_flow with external files
                    module = __import__(
                        f"bubbola.flows.{flow_name}", fromlist=["get_flow"]
                    )
                    if hasattr(module, "get_flow"):
                        flow_config = module.get_flow(external_files)

                return self._flow_config_to_processing_flow(flow_config, external_files)
            except Exception as e:
                print(f"Error loading flow {flow_name}: {e}")
                return None

        # Try to find by flow name in the configuration
        for module_name, flow_config in flows_dict.items():
            try:
                if flow_config.get("name") == flow_name:
                    # If we have external files, we need to call get_flow() with them
                    if external_files:
                        module = __import__(
                            f"bubbola.flows.{module_name}", fromlist=["get_flow"]
                        )
                        if hasattr(module, "get_flow"):
                            flow_config = module.get_flow(external_files)

                    return self._flow_config_to_processing_flow(
                        flow_config, external_files
                    )
            except Exception as e:
                print(f"Error checking flow {module_name}: {e}")
                continue

        return None

    def process_batch(
        self,
        input_path: Path,
        flow_name: str,
        output_dir: Path | None = None,
        max_workers: int = 10,
        timeout: int = 300,
        max_retries: int = 5,
        max_edge_size: int = 1000,
        dry_run: bool = False,
        external_files: dict[str, Path] | None = None,
    ) -> int:
        """Process a batch of images using the specified flow."""
        # Get the processing flow
        flow = self.get_flow(flow_name, external_files)
        if not flow:
            print(f"Error: Flow '{flow_name}' not found.")
            print("Available flows:")
            for available_flow in self.list_flows():
                print(f"  - {available_flow.name}: {available_flow.description}")
                if available_flow.external_file_options:
                    print("    External file options:")
                    for (
                        option,
                        description,
                    ) in available_flow.external_file_options.items():
                        print(f"      --{option}: {description}")
            return 1

        # Validate input path
        if not input_path.exists():
            print(f"Error: Input path does not exist: {input_path}")
            return 1

        if output_dir is None:
            output_dir = input_path.parent / "results"

        # Set up output directory only if not in dry run mode
        if not dry_run:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = output_dir / f"{flow_name}_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)

        elif dry_run:
            # Use a temporary directory for dry run
            import tempfile

            output_dir = Path(tempfile.mkdtemp())

        # Sanitize input to images
        print(f"Sanitizing input: {input_path}")

        if input_path.is_file():
            files_to_process = [input_path]
        else:
            # Filter for supported file types when processing a directory
            supported_extensions = {
                ".pdf",
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".bmp",
                ".tiff",
                ".tif",
                ".webp",
            }
            files_to_process = [
                f
                for f in input_path.iterdir()
                if f.is_file() and f.suffix.lower() in supported_extensions
            ]

        to_process = sanitize_to_images(
            files_to_process,
            return_as_base64=True,
            max_edge_size=max_edge_size,
        )

        if not to_process:
            print("No images found to process.")
            return 1

        print(f"Found {len(to_process)} images to process.")

        # Process images
        start_time = datetime.now()
        processor = ParallelImageProcessor(max_workers=max_workers)

        aggregated_token_counts = processor.process_batch(
            to_process=to_process,
            system_prompt=flow.system_prompt,
            model_name=flow.model_name,
            pydantic_model=flow.data_model,
            results_dir=output_dir,
            max_retries=max_retries,
            timeout=timeout,
            dry_run=dry_run,
        )

        # In dry run mode, only show cost estimation and exit
        if dry_run:
            return 0

        # Only show processing statistics and save results if not in dry run mode
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Print summary
        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        print(f"Results saved in: {output_dir}")
        print(f"Total input tokens: {aggregated_token_counts.total_input_tokens}")
        print(f"Total output tokens: {aggregated_token_counts.total_output_tokens}")
        print(f"Total retry attempts: {aggregated_token_counts.total_retry_count}")

        # Calculate and display actual cost
        from bubbola.price_estimates import get_cost_estimate

        actual_cost = get_cost_estimate(
            flow.model_name,
            aggregated_token_counts.total_input_tokens,
            aggregated_token_counts.total_output_tokens,
        )
        if actual_cost is not None:
            print(
                f"Actual cost: ${actual_cost:.6f} (${actual_cost / len(to_process):.6f} per page)"
            )
        else:
            print(f"Cost calculation: Not available for model {flow.model_name}")

        # Save batch log
        log_data = {
            "flow_name": flow_name,
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "external_files": {k: str(v) for k, v in (external_files or {}).items()},
            "total_input_tokens": aggregated_token_counts.total_input_tokens,
            "total_output_tokens": aggregated_token_counts.total_output_tokens,
            "total_retry_count": aggregated_token_counts.total_retry_count,
            "total_retry_input_tokens": aggregated_token_counts.total_retry_input_tokens,
            "total_retry_output_tokens": aggregated_token_counts.total_retry_output_tokens,
            "processing_time": processing_time,
            "max_workers": max_workers,
            "max_retries": max_retries,
            "dry_run": dry_run,
            "actual_cost": actual_cost,
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_dir / f"batch_log_{timestamp}.json"
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

        # Create results CSV
        try:
            ddts_data, items_data = create_results_csv(output_dir)
            print(f"DDTs: {len(ddts_data)}")
            print(f"Items: {len(items_data)}")
        except Exception as e:
            print(f"Warning: Could not create results CSV: {e}")

        return 0


def main() -> int:
    """Main entry point - runs the currently available flow."""
    # Default configuration
    input_path = Path("input")  # Default input directory
    flow_name = "lg_concrete_v1_test"  # Default flow
    dry_run = True  # Default to dry run for safety

    # Check if input directory exists
    if not input_path.exists():
        print(f"Error: Input directory '{input_path}' does not exist.")
        print("Please create an 'input' directory with your images/PDFs to process.")
        return 1

    processor = BatchProcessor()

    print(f"Processing {input_path} with flow '{flow_name}' (dry_run={dry_run})")

    return processor.process_batch(
        input_path=input_path,
        flow_name=flow_name,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    exit(main())

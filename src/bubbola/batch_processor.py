"""Batch processing functionality for sanitized images."""

import json
import signal
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from bubbola.data_models import DeliveryNote, DeliveryNoteFatturaMatch, ImageDescription
from bubbola.image_data_loader import sanitize_to_images
from bubbola.image_processing import ParallelImageProcessor
from bubbola.results_converter import parse_hierarchical_json


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
            "DeliveryNoteFatturaMatch": DeliveryNoteFatturaMatch,
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
        self,
        flow_name: str,
        external_files: dict[str, Path] | None = None,
        return_config: bool = False,
    ) -> Any:
        """Get a specific processing flow by name. Optionally return the config dict as well."""
        from bubbola.flows import get_flows

        flows_dict = get_flows()
        # First try exact match by module name
        if flow_name in flows_dict:
            flow_config = flows_dict[flow_name]
            try:
                # If we have external files, apply them to the existing config
                # instead of calling get_flow() which would overwrite modifications
                if external_files:
                    flow_config = self._apply_external_files_to_config(
                        flow_name, flow_config, external_files
                    )
                flow_obj = self._flow_config_to_processing_flow(
                    flow_config, external_files
                )
                if return_config:
                    return flow_obj, flow_config
                return flow_obj
            except Exception as e:
                print(f"Error loading flow {flow_name}: {e}")
                return (None, None) if return_config else None
        # Try to find by flow name in the configuration
        for module_name, flow_config in flows_dict.items():
            try:
                if flow_config.get("name") == flow_name:
                    if external_files:
                        flow_config = self._apply_external_files_to_config(
                            module_name, flow_config, external_files
                        )
                    flow_obj = self._flow_config_to_processing_flow(
                        flow_config, external_files
                    )
                    if return_config:
                        return flow_obj, flow_config
                    return flow_obj
            except Exception as e:
                print(f"Error checking flow {module_name}: {e}")
                continue
        return (None, None) if return_config else None

    def _apply_external_files_to_config(
        self, flow_name: str, flow_config: dict, external_files: dict[str, Path]
    ) -> dict:
        """Apply external files to an existing flow configuration without overwriting other modifications."""
        import copy

        # Work on a copy to avoid modifying the original
        config = copy.deepcopy(flow_config)

        # Get the fresh config from the module to see what external files would change
        try:
            module = __import__(f"bubbola.flows.{flow_name}", fromlist=["get_flow"])
            if hasattr(module, "get_flow"):
                fresh_config = module.get_flow(external_files)
                # Only update the system_prompt, which is what external files typically modify
                if "system_prompt" in fresh_config:
                    config["system_prompt"] = fresh_config["system_prompt"]
        except Exception as e:
            print(f"Warning: Could not apply external files to {flow_name}: {e}")

        return config

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
        # Get the processing flow and config dict
        result = self.get_flow(flow_name, external_files, return_config=True)
        if isinstance(result, tuple):
            flow, flow_config_dict = result
        else:
            flow = result
            flow_config_dict = None
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

        # Extract max_edge_size from flow config, with fallback to parameter
        flow_max_edge_size = max_edge_size
        if flow_config_dict is not None:
            flow_max_edge_size = flow_config_dict.get("max_edge_size", max_edge_size)

        to_process = sanitize_to_images(
            files_to_process,
            return_as_base64=True,
            max_edge_size=flow_max_edge_size,
        )

        if not to_process:
            print("No images found to process.")
            return 1

        print(f"Found {len(to_process)} images to process.")

        # Process images
        start_time = datetime.now()
        processor = ParallelImageProcessor(max_workers=max_workers)

        # Extract model_kwargs and parser_kwargs from flow config, with defaults
        model_kwargs = {}
        parser_kwargs = {}
        if flow_config_dict is not None:
            model_kwargs = flow_config_dict.get("model_kwargs", {})
            parser_kwargs = flow_config_dict.get("parser_kwargs", {})
        # For backward compatibility, allow max_retries and require_true_fields at top level
        if not parser_kwargs:
            if flow_config_dict is not None:
                if "max_retries" in flow_config_dict:
                    parser_kwargs["max_n_retries"] = flow_config_dict["max_retries"]
                if "require_true_fields" in flow_config_dict:
                    parser_kwargs["require_true_fields"] = flow_config_dict[
                        "require_true_fields"
                    ]

        aggregated_token_counts = processor.process_batch(
            to_process=to_process,
            system_prompt=flow.system_prompt,
            model_name=flow.model_name,
            pydantic_model=flow.data_model,
            results_dir=output_dir,
            model_kwargs=model_kwargs,
            parser_kwargs=parser_kwargs,
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

        # Capture the current flow configuration (in case it was modified by grid search)
        from bubbola.flows import get_flows

        current_flows = get_flows()
        if flow_name in current_flows:
            current_flow_config = current_flows[flow_name]

            # Convert any Path objects to str for JSON serialization
            def _serialize(obj):
                if isinstance(obj, Path):
                    return str(obj)
                if isinstance(obj, dict):
                    return {k: _serialize(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_serialize(x) for x in obj]
                return obj

            log_data["flow_config"] = _serialize(current_flow_config)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_dir / f"batch_log_{timestamp}.json"
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)

        # Create results CSV
        try:
            level_data, level_names = parse_hierarchical_json(results_dir=output_dir)
            for name, data in zip(level_names, level_data, strict=False):
                print(f"{name}: {len(data)} records")
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

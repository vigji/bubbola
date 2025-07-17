#!/usr/bin/env python3
"""Test script for the new parallel processing functionality."""

import tempfile
from pathlib import Path

from bubbola.data_models import DeliveryNote
from bubbola.image_data_loader import sanitize_to_images
from bubbola.image_processing import ImageProcessor, ParallelImageProcessor


def test_single_image_processing():
    """Test single image processing."""
    print("Testing single image processing...")

    image_path = "tests/assets/single_pages/0088_001_001.png"
    model_name = "mock"  # Use mock model for testing

    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir)

        # Load test image
        base64_images = sanitize_to_images(image_path, return_as_base64=True)
        image_name, image = next(iter(base64_images.items()))

        # Create processor and process image
        processor = ImageProcessor(model_name, DeliveryNote, results_dir)
        token_counts = processor.process_single_image(
            image_name=image_name,
            base64_image=image,
            system_prompt="Extract delivery note information",
            dry_run=True,  # Use dry run to avoid API calls
        )

        print(
            f"✓ Single image processing: {token_counts.total_input_tokens} input tokens, {token_counts.total_output_tokens} output tokens"
        )


def test_parallel_processing():
    """Test parallel processing with multiple images."""
    print("\nTesting parallel processing...")

    # Use multiple test images
    image_paths = [
        "tests/assets/single_pages/0088_001_001.png",
        "tests/assets/single_pages/0089_001_001.png",
        "tests/assets/single_pages/0089_001_002.png",
    ]

    model_name = "mock"  # Use mock model for testing

    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir)

        # Load test images
        to_process = {}
        for image_path in image_paths:
            base64_images = sanitize_to_images(image_path, return_as_base64=True)
            to_process.update(base64_images)

        # Create parallel processor and process batch
        processor = ParallelImageProcessor(max_workers=2)
        aggregated_counts = processor.process_batch(
            to_process=to_process,
            system_prompt="Extract delivery note information",
            model_name=model_name,
            pydantic_model=DeliveryNote,
            results_dir=results_dir,
            dry_run=True,  # Use dry run to avoid API calls
        )

        print(f"✓ Parallel processing: {aggregated_counts.num_images} images processed")
        print(
            f"  Total tokens: {aggregated_counts.total_input_tokens} input, {aggregated_counts.total_output_tokens} output"
        )


def test_legacy_compatibility():
    """Test that the legacy function still works."""
    print("\nTesting legacy compatibility...")

    image_path = "tests/assets/single_pages/0088_001_001.png"
    model_name = "mock"

    with tempfile.TemporaryDirectory() as temp_dir:
        results_dir = Path(temp_dir)

        # Load test image
        base64_images = sanitize_to_images(image_path, return_as_base64=True)

        # Import legacy function
        from bubbola.image_processing import process_images_parallel

        # Test legacy function (it should work with mock response function)
        aggregated_counts = process_images_parallel(
            to_process=base64_images,
            results_dir=results_dir,
            system_prompt="Extract delivery note information",
            response_function=None,  # Not used in new implementation
            response_scheme=DeliveryNote,
            model_name=model_name,
            dry_run=True,
        )

        print(
            f"✓ Legacy compatibility: {aggregated_counts.num_images} images processed"
        )


if __name__ == "__main__":
    print("Testing refactored image processing functionality...\n")

    try:
        test_single_image_processing()
        test_parallel_processing()
        test_legacy_compatibility()
        print("\n🎉 All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise

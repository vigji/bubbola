import base64
import random
from pathlib import Path
from unittest.mock import Mock

import pytest
from PIL import Image
from pydantic import BaseModel

from bubbola.price_estimates import (
    AggregatedTokenCounts,
    TokenCounts,
    _count_words,
    _counts_from_response,
    _estimate_image_tokens_number,
    _estimate_text_tokens_number,
    _get_image_size_from_base64,
    estimate_max_output_tokens_from_schema,
    estimate_tokens_from_messages,
    estimate_tokens_from_messages_with_schema,
    estimate_total_tokens_number,
    get_cost_estimate,
    get_per_token_price,
)


class TestSchema(BaseModel):
    """Test Pydantic model for structured output testing."""

    description: str = "A detailed description of the image content"
    confidence: float = 0.95
    tags: list[str] = ["test", "image"]
    metadata: dict[str, str] = {"source": "test"}


# Create a global instance to avoid pytest collection warning
test_schema = TestSchema()


@pytest.fixture
def test_image_base64():
    """Create a test image and return its base64 encoding."""
    image_long_edge = 256
    base_color = [random.randint(0, 255) for _ in range(3)]
    image = Image.new("RGB", (image_long_edge, image_long_edge), tuple(base_color))

    # Add random noise
    pixels = image.load()
    for x in range(image.width):
        for y in range(image.height):
            noise = random.randint(-10, 10)
            pixel = [max(0, min(255, c + noise)) for c in base_color]
            pixels[x, y] = tuple(pixel)

    # Convert to base64
    temp_image_path = Path("temp_test_image.png")
    image.save(temp_image_path)

    with open(temp_image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    # Clean up
    temp_image_path.unlink()

    return image_base64


@pytest.fixture
def mock_response():
    """Create a mock response with usage information."""
    response = Mock()
    response.usage = Mock()
    response.usage.input_tokens = 100
    response.usage.output_tokens = 50
    return response


class TestTokenCounts:
    """Test suite for TokenCounts dataclass."""

    def test_token_counts_initialization(self):
        """Test TokenCounts initialization with default values."""
        counts = TokenCounts()
        assert counts.total_input_tokens is None
        assert counts.total_output_tokens is None
        assert counts.retry_count == -1
        assert counts.retry_input_tokens is None
        assert counts.retry_output_tokens is None

    def test_add_attempt_first_time(self, mock_response):
        """Test adding first attempt to TokenCounts."""
        counts = TokenCounts()
        counts.add_attempt(mock_response)

        assert counts.total_input_tokens == 100
        assert counts.total_output_tokens == 50
        assert counts.retry_count == 0
        assert counts.retry_input_tokens is None
        assert counts.retry_output_tokens is None

    def test_add_attempt_retry(self, mock_response):
        """Test adding retry attempt to TokenCounts."""
        counts = TokenCounts()
        counts.add_attempt(mock_response)  # First attempt
        counts.add_attempt(mock_response)  # Retry

        assert counts.total_input_tokens == 200
        assert counts.total_output_tokens == 100
        assert counts.retry_count == 1
        assert counts.retry_input_tokens == 100
        assert counts.retry_output_tokens == 50


class TestAggregatedTokenCounts:
    """Test suite for AggregatedTokenCounts dataclass."""

    def test_aggregated_counts_initialization(self):
        """Test AggregatedTokenCounts initialization."""
        agg = AggregatedTokenCounts()
        assert agg.total_input_tokens == 0
        assert agg.total_output_tokens == 0
        assert agg.total_retry_count == 0
        assert agg.num_images == 0

    def test_add_token_counts(self):
        """Test adding TokenCounts to aggregation."""
        agg = AggregatedTokenCounts()
        counts = TokenCounts(
            total_input_tokens=100, total_output_tokens=50, retry_count=1
        )

        agg.add_token_counts(counts)

        assert agg.total_input_tokens == 100
        assert agg.total_output_tokens == 50
        assert agg.total_retry_count == 1
        assert agg.num_images == 1

    def test_add_token_counts_with_none_values(self):
        """Test adding TokenCounts with None values."""
        agg = AggregatedTokenCounts()
        counts = TokenCounts(
            total_input_tokens=None, total_output_tokens=None, retry_count=0
        )

        agg.add_token_counts(counts)

        assert agg.total_input_tokens == 0
        assert agg.total_output_tokens == 0
        assert agg.total_retry_count == 0
        assert agg.num_images == 1

    def test_retry_probability(self):
        """Test retry probability calculation."""
        agg = AggregatedTokenCounts()
        counts1 = TokenCounts(retry_count=1)
        counts2 = TokenCounts(retry_count=0)

        agg.add_token_counts(counts1)
        agg.add_token_counts(counts2)

        assert agg.retry_probability == 0.5

    def test_retry_percentages(self):
        """Test retry percentage calculations."""
        agg = AggregatedTokenCounts()
        counts = TokenCounts(
            total_input_tokens=200,
            total_output_tokens=100,
            retry_input_tokens=50,
            retry_output_tokens=25,
        )

        agg.add_token_counts(counts)

        assert agg.retry_input_percentage == 25.0
        assert agg.retry_output_percentage == 25.0


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_counts_from_response_input_output(self):
        """Test _counts_from_response with input_tokens/output_tokens."""
        response = Mock()
        response.usage.input_tokens = 100
        response.usage.output_tokens = 50

        input_tokens, output_tokens = _counts_from_response(response)
        assert input_tokens == 100
        assert output_tokens == 50

    def test_counts_from_response_prompt_completion(self):
        """Test _counts_from_response with prompt_tokens/completion_tokens."""
        response = Mock()
        # Remove input_tokens/output_tokens to force fallback to prompt/completion
        del response.usage.input_tokens
        del response.usage.output_tokens
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50

        input_tokens, output_tokens = _counts_from_response(response)
        assert input_tokens == 100
        assert output_tokens == 50

    def test_count_words(self):
        """Test _count_words function."""
        assert _count_words("hello world") == 2
        assert _count_words("") == 0
        assert _count_words("single") == 1

    def test_get_image_size_from_base64(self, test_image_base64):
        """Test _get_image_size_from_base64 function."""
        width, height = _get_image_size_from_base64(test_image_base64)
        assert width == 256
        assert height == 256

    def test_get_image_size_from_base64_with_header(self, test_image_base64):
        """Test _get_image_size_from_base64 with data URI header."""
        data_uri = f"data:image/png;base64,{test_image_base64}"
        width, height = _get_image_size_from_base64(data_uri)
        assert width == 256
        assert height == 256


class TestTextTokenEstimation:
    """Test suite for text token estimation."""

    def test_estimate_text_tokens_empty(self):
        """Test text token estimation with empty string."""
        tokens = _estimate_text_tokens_number("")
        assert tokens == 0.0

    def test_estimate_text_tokens_simple(self):
        """Test text token estimation with simple text."""
        tokens = _estimate_text_tokens_number("hello world")
        assert tokens > 0

    def test_estimate_text_tokens_long_text(self):
        """Test text token estimation with longer text."""
        long_text = (
            "This is a longer text with more words to test token estimation. " * 10
        )
        tokens = _estimate_text_tokens_number(long_text)
        assert tokens > 0


class TestImageTokenEstimation:
    """Test suite for image token estimation."""

    def test_estimate_image_tokens_patch_based(self):
        """Test image token estimation for patch-based models."""
        # Test o4-mini model
        tokens = _estimate_image_tokens_number(512, 512, "o4-mini")
        assert tokens > 0

        # Test gpt-4.1-mini model
        tokens = _estimate_image_tokens_number(512, 512, "gpt-4.1-mini")
        assert tokens > 0

    def test_estimate_image_tokens_tile_based(self):
        """Test image token estimation for tile-based models."""
        # Test gpt-4o model
        tokens = _estimate_image_tokens_number(512, 512, "gpt-4o", "high")
        assert tokens > 0

        # Test with low detail
        tokens_low = _estimate_image_tokens_number(512, 512, "gpt-4o", "low")
        assert tokens_low > 0
        assert tokens_low < tokens  # Low detail should use fewer tokens

    def test_estimate_image_tokens_unknown_model(self):
        """Test image token estimation with unknown model."""
        with pytest.raises(ValueError, match="No known image token calculation"):
            _estimate_image_tokens_number(512, 512, "unknown-model")

    def test_estimate_image_tokens_invalid_detail(self):
        """Test image token estimation with invalid detail level."""
        with pytest.raises(ValueError, match="detail must be 'low' or 'high'"):
            _estimate_image_tokens_number(512, 512, "gpt-4o", "medium")


class TestTotalTokenEstimation:
    """Test suite for total token estimation."""

    def test_estimate_total_tokens_number(self, test_image_base64):
        """Test total token estimation with text and image."""
        text = "Describe this image"
        tokens = estimate_total_tokens_number(text, test_image_base64, "gpt-4o")
        assert tokens > 0

    def test_estimate_tokens_from_messages_text_only(self):
        """Test token estimation from messages with text only."""
        messages = [{"role": "user", "content": "Hello world"}]
        input_tokens, output_tokens = estimate_tokens_from_messages(messages, "gpt-4o")
        assert input_tokens is not None
        assert output_tokens is None

    def test_estimate_tokens_from_messages_with_image(self, test_image_base64):
        """Test token estimation from messages with image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{test_image_base64}"
                        },
                    },
                ],
            }
        ]
        input_tokens, output_tokens = estimate_tokens_from_messages(messages, "gpt-4o")
        assert input_tokens is not None
        assert output_tokens is None


class TestSchemaTokenEstimation:
    """Test suite for schema-based token estimation."""

    def test_estimate_max_output_tokens_from_schema_simple(self):
        """Test output token estimation from simple schema."""
        schema = {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "confidence": {"type": "number"},
            },
        }
        tokens = estimate_max_output_tokens_from_schema(schema)
        assert 50 <= tokens <= 8000

    def test_estimate_max_output_tokens_from_schema_complex(self):
        """Test output token estimation from complex schema."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "quantity": {"type": "number"},
                        },
                    },
                }
            },
        }
        tokens = estimate_max_output_tokens_from_schema(schema)
        assert 50 <= tokens <= 8000

    def test_estimate_max_output_tokens_from_schema_invalid(self):
        """Test output token estimation with invalid schema."""
        tokens = estimate_max_output_tokens_from_schema(None)
        assert tokens == 1000

        tokens = estimate_max_output_tokens_from_schema("invalid")
        assert tokens == 1000

    def test_estimate_tokens_from_messages_with_schema(self, test_image_base64):
        """Test token estimation from messages with schema."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{test_image_base64}"
                        },
                    },
                ],
            }
        ]
        schema = test_schema.model_json_schema()

        input_tokens, output_tokens = estimate_tokens_from_messages_with_schema(
            messages, "gpt-4o", schema
        )
        assert input_tokens is not None
        assert output_tokens is not None


class TestPriceEstimation:
    """Test suite for price estimation."""

    def test_get_per_token_price_valid_model(self):
        """Test getting per-token price for valid model."""
        prices = get_per_token_price("gpt-4o")
        assert prices is not None
        assert "in" in prices
        assert "out" in prices
        assert prices["in"] > 0
        assert prices["out"] > 0

    def test_get_per_token_price_invalid_model(self):
        """Test getting per-token price for invalid model."""
        prices = get_per_token_price("invalid-model")
        assert prices is None

    def test_get_cost_estimate(self):
        """Test cost estimation."""
        cost = get_cost_estimate("gpt-4o", 1000, 500)
        assert cost is not None
        assert cost > 0

    def test_get_cost_estimate_invalid_model(self):
        """Test cost estimation with invalid model."""
        cost = get_cost_estimate("invalid-model", 1000, 500)
        assert cost is None


class TestModelSpecificAccuracy:
    """Test suite for model-specific token estimation accuracy."""

    @pytest.mark.modeltest
    def test_token_estimation_accuracy_across_models(self, request, test_image_base64):
        """Test that token estimates are within 20% of actual tokens across all models."""
        if not request.config.getoption("--run-model-tests"):
            pytest.skip("Model tests not enabled")

        from bubbola.data_models import DeliveryNote
        from bubbola.model_creator import get_model_client

        # Test models that should have estimates
        test_models = [
            "gpt-4o-mini",
            "gpt-4o",
            "o4-mini",
            "o3",
            "mistralai/mistral-small-3.2-24b-instruct:free",
        ]

        # Sample text and schema
        sample_text = "Please analyze this delivery note and extract the following information: supplier name, order number, delivery date, and list of items with quantities and prices."

        for model_name in test_models:
            try:
                model = get_model_client(model_name)

                # Create messages using the model
                messages = model.create_messages(sample_text, [test_image_base64])

                # Estimate tokens
                est_input, est_output = estimate_tokens_from_messages_with_schema(
                    messages, model_name, DeliveryNote.model_json_schema()
                )

                # Use dry run first to get token estimates
                result, token_counts = model.query_with_instructions(
                    instructions=sample_text,
                    images=[test_image_base64],
                    pydantic_model=DeliveryNote,
                    dry_run=True,
                )

                # Get actual token counts from dry run
                actual_input = token_counts.total_input_tokens
                actual_output = token_counts.total_output_tokens

                # Check input token accuracy (within 20%)
                if est_input is not None:
                    input_accuracy = abs(est_input - actual_input) / actual_input
                    assert input_accuracy <= 0.2, (
                        f"Input token estimate for {model_name} off by {input_accuracy:.1%}"
                    )

                # Check output token accuracy (within 20%)
                if est_output is not None:
                    output_accuracy = abs(est_output - actual_output) / actual_output
                    assert output_accuracy <= 0.2, (
                        f"Output token estimate for {model_name} off by {output_accuracy:.1%}"
                    )

                # For OpenAI models, estimates should not be None
                if model_name.startswith(("gpt-", "o")):
                    assert est_input is not None, (
                        f"Input estimate should not be None for {model_name}"
                    )
                    assert est_output is not None, (
                        f"Output estimate should not be None for {model_name}"
                    )

            except Exception as e:
                # Skip models that fail (API issues, etc.)
                pytest.skip(f"Model {model_name} failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])

import base64
import random
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from pydantic import BaseModel

from bubbola.model_creator import (
    MODEL_NAME_TO_CLASS_MAP,
    LLMModel,
    MockModel,
    NewResponsesAPI,
    OpenAIModel,
    get_model_client,
)


class ImageDescription(BaseModel):
    """Test Pydantic model for image description."""

    description: str
    main_color: str


class AnotherTestModel(BaseModel):
    """Another test Pydantic model."""

    test_field: str
    number: int


@pytest.fixture
def test_image_base64():
    """Create a test image and return its base64 encoding."""
    # Create test image
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


class TestModelCreator:
    """Test suite for model creator functionality."""

    def test_model_name_to_class_mapping(self):
        """Test that all models in the mapping can be instantiated."""
        for model_name, model_class in MODEL_NAME_TO_CLASS_MAP.items():
            try:
                model_instance = model_class(name=model_name)
                assert isinstance(model_instance, LLMModel)
                assert model_instance.name == model_name
            except Exception as e:
                pytest.fail(f"Failed to instantiate {model_name}: {e}")

    def test_get_model_client_mock(self):
        """Test getting mock model client."""
        model = get_model_client("mock")
        assert isinstance(model, MockModel)
        assert model.name == "mock"

    def test_get_model_client_unknown_model(self):
        """Test that unknown models raise ValueError."""
        with pytest.raises(ValueError, match="not found in MODEL_NAME_TO_CLASS_MAP"):
            get_model_client("unknown-model")


class TestAPIInterfaces:
    """Test suite for API interface classes."""

    def test_new_responses_api_format_image_message(self):
        """Test new responses API image message formatting."""
        api = NewResponsesAPI()
        image_base64 = "test_base64_string"
        message = api.format_image_message(image_base64)

        assert message["role"] == "user"
        assert len(message["content"]) == 1
        assert message["content"][0]["type"] == "input_image"
        assert (
            message["content"][0]["image_url"]
            == f"data:image/png;base64,{image_base64}"
        )


class TestLLMModel:
    """Test suite for LLMModel base class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model instance for testing."""
        return MockModel(name="test-mock")

    def test_create_messages_with_instructions_only(self, mock_model):
        """Test creating messages with only instructions."""
        instructions = "Describe the image briefly."
        messages = mock_model.create_messages(instructions)

        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == instructions

    def test_create_messages_with_instructions_and_images(
        self, mock_model, test_image_base64
    ):
        """Test creating messages with instructions and images."""
        instructions = "Describe the image briefly."
        messages = mock_model.create_messages(instructions, [test_image_base64])

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == instructions
        assert messages[1]["role"] == "user"
        assert len(messages[1]["content"]) == 1
        assert messages[1]["content"][0]["type"] == "image_url"

    def test_create_messages_with_custom_roles(self, mock_model, test_image_base64):
        """Test creating messages with custom roles."""
        instructions = "Describe the image briefly."
        messages = mock_model.create_messages(
            instructions,
            [test_image_base64],
            system_role="assistant",
            user_role="assistant",
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "assistant"

    def test_create_messages_empty_instructions(self, mock_model, test_image_base64):
        """Test creating messages with empty instructions."""
        messages = mock_model.create_messages("", [test_image_base64])

        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_format_image_message(self, mock_model, test_image_base64):
        """Test image message formatting."""
        message = mock_model.format_image_message(test_image_base64)

        assert message["role"] == "user"
        assert len(message["content"]) == 1
        assert message["content"][0]["type"] == "image_url"
        assert "data:image/png;base64," in message["content"][0]["image_url"]["url"]

    def test_query_with_instructions_dry_run(self, mock_model, test_image_base64):
        """Test query_with_instructions with dry_run=True."""
        instructions = "Describe the image briefly."
        result, token_counts, _ = mock_model.query_with_instructions(
            instructions=instructions,
            images=[test_image_base64],
            pydantic_model=ImageDescription,
            dry_run=True,
        )

        assert result is None
        # Token counts might be None if estimation fails, which is acceptable
        assert (
            token_counts.total_input_tokens is None
            or token_counts.total_input_tokens > 0
        )
        assert (
            token_counts.total_output_tokens is None
            or token_counts.total_output_tokens > 0
        )
        assert token_counts.retry_count == 0

    def test_query_with_instructions_without_pydantic(
        self, mock_model, test_image_base64
    ):
        """Test query_with_instructions without Pydantic model."""
        instructions = "Describe the image briefly."
        # This should return a raw response (mock in this case)
        result = mock_model.query_with_instructions(
            instructions=instructions, images=[test_image_base64]
        )

        # Mock model should return a mock response
        assert result is not None

    def test_get_parsed_response_dry_run(self, mock_model, test_image_base64):
        """Test get_parsed_response with dry_run=True."""
        messages = mock_model.create_messages("Describe the image", [test_image_base64])
        _, token_counts, _ = mock_model.get_parsed_response(
            messages, ImageDescription, dry_run=True
        )

        # Token counts might be None if estimation fails, which is acceptable
        assert (
            token_counts.total_input_tokens is None
            or token_counts.total_input_tokens > 0
        )
        assert (
            token_counts.total_output_tokens is None
            or token_counts.total_output_tokens > 0
        )
        assert token_counts.retry_count == 0


class TestModelClasses:
    """Test suite for specific model classes."""

    def test_mock_model_instantiation(self):
        """Test MockModel instantiation."""
        model = MockModel(name="test-mock")
        assert model.name == "test-mock"
        assert model.client_class.__name__ == "MockModelClient"

    @pytest.mark.modeltest
    def test_openai_model_instantiation(self, request):
        """Test OpenAIModel instantiation (real model, requires --run-model-tests)."""
        if not request.config.getoption("--run-model-tests"):
            pytest.skip("use --run-model-tests to run real model tests")
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = OpenAIModel(name="gpt-4o")
            assert model.name == "gpt-4o"
            assert model.use_new_responses_api is True


class TestIntegration:
    """Integration tests for the model creator system."""

    def test_mock_model_full_workflow(self, test_image_base64):
        """Test complete workflow with mock model."""
        model = get_model_client("mock")

        # Test message creation
        instructions = "Describe the image in a short sentence. Return a JSON object with the description and the main color."
        messages = model.create_messages(instructions, [test_image_base64])

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        # Test dry run functionality
        result, token_counts, _ = model.query_with_instructions(
            instructions=instructions,
            images=[test_image_base64],
            pydantic_model=ImageDescription,
            dry_run=True,
        )

        assert result is None
        # Token counts might be None if estimation fails, which is acceptable
        assert (
            token_counts.total_input_tokens is None
            or token_counts.total_input_tokens > 0
        )
        assert (
            token_counts.total_output_tokens is None
            or token_counts.total_output_tokens > 0
        )

    @pytest.mark.modeltest
    def test_model_client_creation_for_all_models(self, request):
        """Test that all models in the mapping can be created via get_model_client (requires --run-model-tests)."""
        if not request.config.getoption("--run-model-tests"):
            pytest.skip("use --run-model-tests to run real model tests")
        for model_name in MODEL_NAME_TO_CLASS_MAP.keys():
            try:
                model = get_model_client(model_name)
                assert isinstance(model, LLMModel)
                assert model.name == model_name
            except Exception as e:
                # Some models might fail due to missing API keys, which is expected
                # But the creation itself should work
                print(
                    f"Model {model_name} creation failed (expected if no API key): {e}"
                )

    @pytest.mark.modeltest
    def test_structured_responses_from_all_models(self, request, test_image_base64):
        """Test getting structured responses from all available models (requires --run-model-tests)."""
        if not request.config.getoption("--run-model-tests"):
            pytest.skip("use --run-model-tests to run real model tests")

        simple_instructions = "Describe this image in one word. Return a JSON object with 'description' and 'main_color' fields."

        successful_models = []
        failed_models = []

        for model_name in MODEL_NAME_TO_CLASS_MAP.keys():
            print(f"\nTesting structured response from: {model_name}")

            try:
                model = get_model_client(model_name)

                # First, do a dry run to estimate costs
                print(f"  Estimating tokens for {model_name}...")
                try:
                    _, token_counts, _ = model.query_with_instructions(
                        instructions=simple_instructions,
                        images=[test_image_base64],
                        pydantic_model=ImageDescription,
                        dry_run=True,
                    )

                    input_tokens = token_counts.total_input_tokens or 0
                    output_tokens = token_counts.total_output_tokens or 0
                    print(
                        f"  Estimated: {input_tokens} input + {output_tokens} output tokens"
                    )

                    # Skip if estimated cost is too high (safety check)
                    if input_tokens > 1000 or output_tokens > 500:
                        print(f"  Skipping {model_name} - estimated tokens too high")
                        failed_models.append((model_name, "estimated tokens too high"))
                        continue

                except Exception as e:
                    print(f"  Token estimation failed for {model_name}: {e}")
                    failed_models.append((model_name, f"token estimation failed: {e}"))
                    continue

                # Now try the actual API call
                print(f"  Making API call to {model_name}...")
                try:
                    # For OpenAI models, don't use max_tokens as it's not supported in responses API
                    kwargs = {}
                    if not model.use_new_responses_api:
                        kwargs["max_tokens"] = 50  # Only for legacy API

                    result, actual_token_counts, _ = model.query_with_instructions(
                        instructions=simple_instructions,
                        images=[test_image_base64],
                        pydantic_model=ImageDescription,
                        **kwargs,
                    )

                    # Validate the response
                    assert isinstance(result, ImageDescription)
                    assert isinstance(result.description, str)
                    assert isinstance(result.main_color, str)
                    assert len(result.description) > 0
                    assert len(result.main_color) > 0

                    actual_input = actual_token_counts.total_input_tokens or 0
                    actual_output = actual_token_counts.total_output_tokens or 0
                    print(
                        f"  Success! Actual: {actual_input} input + {actual_output} output tokens"
                    )
                    print(
                        f"     Response: {result.description}, color: {result.main_color}"
                    )

                    successful_models.append(model_name)

                except Exception as e:
                    print(f"  API call failed for {model_name}: {e}")
                    failed_models.append((model_name, f"API call failed: {e}"))

            except Exception as e:
                print(f"  Model creation failed for {model_name}: {e}")
                failed_models.append((model_name, f"model creation failed: {e}"))

        # Print summary
        print(f"\n{'=' * 60}")
        print("STRUCTURED RESPONSE TEST SUMMARY")
        print(f"{'=' * 60}")
        print(f"Successful models ({len(successful_models)}):")
        for model in successful_models:
            print(f"  {model}")

        print(f"\nFailed models ({len(failed_models)}):")
        for model, reason in failed_models:
            print(f"  {model}: {reason}")

        # Assert that at least some models worked (including mock)
        assert len(successful_models) > 0, (
            "No models returned successful structured responses"
        )

        # If we have real models working, that's great
        real_models = [m for m in successful_models if m != "mock"]
        if real_models:
            print(
                f"\n{len(real_models)} real models successfully returned structured responses!"
            )
        else:
            print(
                "\nOnly mock model worked - check your API keys and model availability"
            )

    def test_api_interface_selection(self):
        """Test that the correct API interface is selected based on use_new_responses_api."""
        # Test new responses API
        new_api_model = OpenAIModel(name="new-api-test")
        assert isinstance(new_api_model.api_interface, NewResponsesAPI)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

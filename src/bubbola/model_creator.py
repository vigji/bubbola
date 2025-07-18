from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from bubbola.config import get_env
from bubbola.price_estimates import (
    TokenCounts,
    estimate_tokens_from_messages_with_schema,
)


class MockModelClient:
    """Mock client that returns empty responses for testing.
    Emulates the OpenAI client interface.
    """

    def __init__(self, *args, **kwargs):
        self.model_name = "mock-model"
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        """Create a response with empty content and zero usage tokens."""
        # Return a mock JSON response for testing
        mock_content = '{"description": "A test image", "main_color": "blue"}'

        return type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": mock_content})()},
                    )()
                ],
                "usage": type(
                    "Usage", (), {"completion_tokens": 10, "prompt_tokens": 20}
                )(),
            },
        )()


# ============================================================================
# API Interface Classes
# ============================================================================


class APIInterface(ABC):
    """Abstract base class for API interfaces."""

    @staticmethod
    def format_image_message(
        image_base64: str, use_new_api: bool = False
    ) -> dict[str, Any]:
        """Format an image for inclusion in messages."""
        image_url = f"data:image/png;base64,{image_base64}"
        image_type = "input_image" if use_new_api else "image_url"

        return {
            "role": "user",
            "content": [
                {
                    "type": image_type,
                    "image_url": {"url": image_url} if not use_new_api else image_url,
                }
            ],
        }

    @abstractmethod
    def parse_response(self, response: Any, pydantic_model: BaseModel) -> Any:
        """Parse and validate response content."""
        pass

    @abstractmethod
    def create_schema_request(
        self, client: Any, messages: list[dict], pydantic_model: BaseModel, **kwargs
    ) -> Any:
        """Create a request with schema validation."""
        pass

    @abstractmethod
    def create_simple_request(self, client: Any, messages: list[dict], **kwargs) -> Any:
        """Create a simple request without schema validation."""
        pass


class LegacyAPI(APIInterface):
    """Legacy OpenAI-style API interface."""

    def format_image_message(self, image_base64: str) -> dict[str, Any]:
        """Format image for legacy API."""
        return super().format_image_message(image_base64, use_new_api=False)

    @staticmethod
    def parse_response(response: Any, pydantic_model: BaseModel) -> Any:
        """Parse response from legacy API."""
        response_content = response.choices[0].message.content

        if not response_content or not response_content.strip():
            raise ValueError("Empty response content")

        try:
            import json

            parsed_json = json.loads(response_content)

            # Check if response is actually the schema instead of data
            if isinstance(parsed_json, dict) and (
                "$defs" in parsed_json or "properties" in parsed_json
            ):
                raise ValueError("Response contains schema instead of data")

            return pydantic_model.model_validate(parsed_json)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(
                f"Invalid response content for {pydantic_model.__name__}: {e}"
            ) from e

    def create_schema_request(
        self, client: Any, messages: list[dict], pydantic_model: BaseModel, **kwargs
    ) -> Any:
        """Create request with JSON schema for legacy API."""
        return client.chat.completions.create(
            model=client.model_name,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": pydantic_model.__name__,
                    "schema": pydantic_model.model_json_schema(),
                },
            },
            **kwargs,
        )

    def create_simple_request(self, client: Any, messages: list[dict], **kwargs) -> Any:
        """Create simple request for legacy API."""
        return client.chat.completions.create(
            model=client.model_name, messages=messages, **kwargs
        )


class NewResponsesAPI(APIInterface):
    """New OpenAI responses API interface."""

    def format_image_message(self, image_base64: str) -> dict[str, Any]:
        """Format image for new responses API."""
        return super().format_image_message(image_base64, use_new_api=True)

    def parse_response(self, response: Any, pydantic_model: BaseModel) -> Any:
        """Parse response from new responses API."""
        return response.output_parsed

    def create_schema_request(
        self, client: Any, messages: list[dict], pydantic_model: BaseModel, **kwargs
    ) -> Any:
        """Create request with text format for new responses API."""
        return client.responses.parse(
            model=client.model_name,
            input=messages,
            text_format=pydantic_model,
            **kwargs,
        )

    def create_simple_request(self, client: Any, messages: list[dict], **kwargs) -> Any:
        """Create simple request for new responses API."""
        return client.chat.completions.create(
            model=client.model_name, messages=messages, **kwargs
        )


# ============================================================================
# Enhanced Model Classes
# ============================================================================


@dataclass
class LLMModel:
    name: str
    client_class: type[OpenAI]
    base_url: str | None
    api_key_env_var: str
    use_new_responses_api: bool = False

    def __post_init__(self):
        """Initialize the appropriate API interface."""
        self.api_interface = (
            NewResponsesAPI() if self.use_new_responses_api else LegacyAPI()
        )

    @property
    def api_key(self):
        return get_env(self.api_key_env_var)

    @property
    def client(self):
        client = self.client_class(api_key=self.api_key, base_url=self.base_url)
        client.model_name = self.name
        return client

    def validate_token(self) -> None:
        """Validate that the required API token is available for this model.

        Raises:
            ValueError: If the required token is not available
        """
        # If no API key required (local/mock models), we're good
        if not self.api_key_env_var:
            return

        # Check if the required token is available and properly configured
        try:
            from bubbola.config import get_required

            token_value = get_required(self.api_key_env_var)
            if (
                not token_value
                or token_value == f"your_{self.api_key_env_var.lower()}_here"
            ):
                raise ValueError(
                    f"Token for {self.api_key_env_var} is not properly configured"
                )
        except ValueError as e:
            raise ValueError(
                f"Model '{self.name}' requires {self.api_key_env_var} to be configured. "
                f"Please add it to ~/.bubbola/config.env: {e}"
            ) from e

    def format_image_message(self, image_base64: str) -> dict[str, Any]:
        """Format an image for inclusion in messages."""
        return self.api_interface.format_image_message(image_base64)

    def parse_response(self, response: Any, pydantic_model: BaseModel) -> Any:
        """Parse and validate response content."""
        return self.api_interface.parse_response(response, pydantic_model)

    def create_schema_request(
        self, messages: list[dict], pydantic_model: BaseModel, **kwargs
    ) -> Any:
        """Create a request with schema validation."""
        return self.api_interface.create_schema_request(
            self.client, messages, pydantic_model, **kwargs
        )

    def create_simple_request(self, messages: list[dict], **kwargs) -> Any:
        """Create a simple request without schema validation."""
        return self.api_interface.create_simple_request(self.client, messages, **kwargs)

    def get_parsed_response(
        self,
        messages: list[dict],
        pydantic_model: BaseModel,
        max_n_retries: int = 5,
        dry_run: bool = False,
        **kwargs,
    ) -> tuple[Any, TokenCounts]:
        """Send messages to model and return parsed response with token counts.

        Args:
            messages: List of message dictionaries
            pydantic_model: Pydantic model for response validation
            max_n_retries: Maximum number of retry attempts
            dry_run: If True, estimate tokens without making API calls
            **kwargs: Additional arguments to pass to the model

        Returns:
            Tuple of (parsed_response, token_counts)
        """
        if dry_run:
            # Estimate tokens using the price_estimates module
            schema = pydantic_model.model_json_schema()
            estimated_input_tokens, estimated_output_tokens = (
                estimate_tokens_from_messages_with_schema(messages, self.name, schema)
            )

            token_counts = TokenCounts(
                total_input_tokens=estimated_input_tokens,
                total_output_tokens=estimated_output_tokens,
                retry_count=0,
            )
            return None, token_counts

        # Normal execution
        token_counts = TokenCounts(total_input_tokens=0, total_output_tokens=0)

        for attempt in range(max_n_retries + 1):
            try:
                response = self.create_schema_request(
                    messages, pydantic_model, **kwargs
                )
                token_counts.add_attempt(response)
                response_content = self.parse_response(response, pydantic_model)
                break

            except Exception as e:
                print(f"API call failed on attempt {attempt + 1}: {e}")
                if attempt == max_n_retries:
                    raise e
                if "response" in locals():
                    token_counts.add_attempt(response)
                continue

        return response_content, token_counts

    def get_simple_response(self, messages: list[dict], **kwargs) -> Any:
        """Send messages to model and return raw response.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional arguments to pass to the model

        Returns:
            Raw model response
        """
        return self.create_simple_request(messages, **kwargs)

    def create_messages(
        self,
        instructions: str,
        images: list[str] | None = None,
        system_role: str = "system",
        user_role: str = "user",
    ) -> list[dict[str, Any]]:
        """Create a formatted message list from instructions and optional images.

        Args:
            instructions: The system instructions/prompt
            images: Optional list of base64-encoded images
            system_role: Role for the instructions message (default: "system")
            user_role: Role for image messages (default: "user")

        Returns:
            List of formatted message dictionaries
        """
        messages = []

        # Add instructions as system message
        if instructions.strip():
            messages.append({"role": system_role, "content": instructions.strip()})

        # Add images as user messages
        if images:
            for image_base64 in images:
                image_message = self.format_image_message(image_base64)
                image_message["role"] = user_role
                messages.append(image_message)

        return messages

    def query_with_instructions(
        self,
        instructions: str,
        images: list[str] | None = None,
        pydantic_model: BaseModel | None = None,
        max_n_retries: int = 5,
        dry_run: bool = False,
        **kwargs,
    ) -> Any | tuple[Any, TokenCounts]:
        """Convenience method to query with instructions and optional images.

        Args:
            instructions: The system instructions/prompt
            images: Optional list of base64-encoded images
            pydantic_model: Optional Pydantic model for response validation
            max_n_retries: Maximum number of retry attempts
            dry_run: If True, estimate tokens without making API calls
            **kwargs: Additional arguments to pass to the model

        Returns:
            If pydantic_model is provided: (parsed_response, token_counts)
            Otherwise: raw response
        """
        messages = self.create_messages(instructions, images)

        if pydantic_model:
            return self.get_parsed_response(
                messages, pydantic_model, max_n_retries, dry_run, **kwargs
            )
        else:
            return self.get_simple_response(messages, **kwargs)


class OpenAIModel(LLMModel):
    def __init__(self, name: str = "openai"):
        super().__init__(
            name=name,
            client_class=OpenAI,
            base_url=None,
            api_key_env_var="OPENAI_API_KEY",
            use_new_responses_api=True,
        )


class LocalModel(LLMModel):
    def __init__(self, name: str = "local"):
        super().__init__(
            name=name,
            client_class=OpenAI,
            base_url="http://127.0.0.1:11434/v1",
            api_key_env_var="",
        )


class OpenRouterModel(LLMModel):
    def __init__(self, name: str = "openrouter"):
        super().__init__(
            name=name,
            client_class=OpenAI,
            base_url="https://openrouter.ai/api/v1",
            api_key_env_var="OPENROUTER",
        )


class MockModel(LLMModel):
    def __init__(self, name: str = "mock"):
        super().__init__(
            name=name, client_class=MockModelClient, base_url="", api_key_env_var=""
        )


MODEL_NAME_TO_CLASS_MAP = {
    # OpenRouter models
    "meta-llama/llama-4-scout": OpenRouterModel,
    "meta-llama/llama-4-maverick": OpenRouterModel,
    # "google/gemma-3-27b-it:free": OpenRouterModel,
    # "anthropic/claude-3-haiku:beta": OpenRouterModel,
    "mistralai/mistral-small-3.2-24b-instruct:free": OpenRouterModel,
    # Local models
    "gemma3:12b": LocalModel,
    # OpenAI models
    "gpt-4o": OpenAIModel,
    "gpt-4o-mini": OpenAIModel,
    "o3": OpenAIModel,
    "o4-mini": OpenAIModel,
    # Mock
    "mock": MockModel,
}


def get_model_client(required_model: str, force_openrouter=False):
    """Get the appropriate model client based on the model name.

    Args:
        required_model: The name of the model to use
        force_openrouter: If True, force using OpenRouter for supported models

    Returns:
        The appropriate model client instance
    """
    # Handle mock/test models
    if "mock" in required_model or "test" in required_model:
        return MockModel(name=required_model)

    # Check if model is in our mapping
    if required_model in MODEL_NAME_TO_CLASS_MAP:
        model_class = MODEL_NAME_TO_CLASS_MAP[required_model]

        # If forcing OpenRouter and model supports it, use OpenRouter
        if force_openrouter and model_class == OpenAIModel:
            print("Forcing OpenRouter usage")
            return OpenRouterModel(name=required_model)

        # Create the model instance
        model_instance = model_class(name=required_model)

        # Validate that the required token is available for this model
        model_instance.validate_token()

        return model_instance

    # If model not found in mapping, raise error
    raise ValueError(
        f"Model {required_model} not found in MODEL_NAME_TO_CLASS_MAP. Please add it to the mapping."
    )


if __name__ == "__main__":
    # Test different model types
    for model_name in list(MODEL_NAME_TO_CLASS_MAP.keys())[:3]:
        print(f"\nTesting model: {model_name}")
        try:
            client = get_model_client(model_name)
            print(f"✓ Successfully created client for {model_name}")
        except Exception as e:
            print(f"✗ Failed to create client for {model_name}: {e}")

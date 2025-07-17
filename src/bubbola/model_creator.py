# %%
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Load environment variables
load_dotenv("/Users/vigji/code/bubbola/config.env")


class MockModelClient:
    """Mock client that returns empty responses for testing.
    Emulates the OpenAI client interface.
    """

    def __init__(self):
        self.model_name = "mock-model"
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        """Create a response with empty content and zero usage tokens."""
        return type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {"message": type("Message", (), {"content": ""})()},
                    )()
                ],
                "usage": type(
                    "Usage", (), {"completion_tokens": 0, "prompt_tokens": 0}
                )(),
            },
        )()


@dataclass
class LLMModel:
    name: str
    client_class: type[OpenAI]  # TODO: handle better
    base_url: str
    api_key_env_var: str

    @property
    def api_key(self):
        return os.getenv(self.api_key_env_var)

    @property
    def client(self):
        return self.client_class(api_key=self.api_key, base_url=self.base_url)


class OpenAIModel(LLMModel):
    def __init__(self, name: str = "openai"):
        super().__init__(
            name=name,
            client_class=OpenAI,
            base_url=None,
            api_key_env_var="OPENAI_API_KEY",
            # base_url="https://openrouter.ai/api/v1",
            # api_key_env_var="OPENROUTER",
        )


class LocalModel(LLMModel):
    def __init__(self, name: str = "local"):
        super().__init__(
            name=name,
            client_class=OpenAI,
            base_url="http://127.0.0.1:11434/v1",
            api_key_env_var="",
        )


# class DeepInfraModel(LLMModel):
#     def __init__(self, name: str = "deepinfra"):
#         super().__init__(
#             name=name,
#             client_class=OpenAI,
#             base_url="https://api.deepinfra.com/v1/openai",
#             api_key_env_var="DEEPINFRA_TOKEN"
#         )


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
    "google/gemma-3-27b-it:free": OpenRouterModel,
    "anthropic/claude-3-haiku:beta": OpenRouterModel,
    "mistralai/mistral-small-3.2-24b-instruct:free": OpenRouterModel,
    # Local models
    "gemma3:12b": LocalModel,
    # DeepInfra models
    # "meta-llama/Llama-3.2": DeepInfraModel,
    # OpenAI models
    "gpt-4o": OpenAIModel,
    "gpt-4o-mini": OpenAIModel,
    # "gpt-4": OpenAIModel,
    "o3": OpenAIModel,
    "o4-mini": OpenAIModel,
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
        return MockModelClient()

    # Check if model is in our mapping
    if required_model in MODEL_NAME_TO_CLASS_MAP:
        model_class = MODEL_NAME_TO_CLASS_MAP[required_model]

        # If forcing OpenRouter and model supports it, use OpenRouter
        if force_openrouter and model_class in [
            OpenAIModel,
            # LocalModel,
            # DeepInfraModel,
        ]:
            print("Forcing OpenRouter usage")
            return OpenRouterModel().client

        # Use the mapped model class
        return model_class().client

    # If model not found in mapping, raise error
    raise ValueError(
        f"Model {required_model} not found in MODEL_NAME_TO_CLASS_MAP. Please add it to the mapping."
    )


def get_client_response_function(required_model: str, force_openrouter=False):
    client = get_model_client(required_model, force_openrouter)
    print(client)
    return client.chat.completions.create

    # # If a pydantic model is provided, use the new .parse interface for structured outputs
    # if pydantic_model is not None:
    #     # Use the beta.chat.completions.parse method for strict schema enforcement
    #     return client.beta.chat.completions.parse
    # else:
    #     # Fallback to regular chat completions for non-structured outputs
    #     return client.chat.completions.create


# %%
# %%


def get_client_response_function_with_schema(
    required_model: str, pydantic_model: BaseModel, force_openrouter=False
):
    client = get_model_client(required_model, force_openrouter)

    if required_model in ["gpt-4o", "gpt-4o-mini", "o3", "o4-mini"]:

        def schemed_client_response(
            model: str,
            messages: list[dict],
            pydantic_model: BaseModel,
        ):
            return client.responses.parse(
                model=model, messages=messages, response_format=pydantic_model
            )


if __name__ == "__main__":
    import base64
    import random
    from pathlib import Path

    from PIL import Image

    # Test different model types
    test_all_models = True

    for model_name in MODEL_NAME_TO_CLASS_MAP.keys():
        print(f"\nTesting model: {model_name}")
        try:
            client = get_model_client(model_name)
            print(f"✓ Successfully created client for {model_name}")
        except Exception as e:
            print(f"✗ Failed to create client for {model_name}: {e}")

    # Test the full pipeline with a specific model
    model_name = "mistralai/mistral-small-3.2-24b-instruct:free"

    response_function = get_client_response_function(model_name)

    test_image_url = "https://commons.wikimedia.org/wiki/Special:FilePath/Stonehenge.jpg?width=100&format=png"

    # get and convert to base64
    # response = requests.get(test_image_url)
    # image_base64 = base64.b64encode(response.content).decode("utf-8")
    temp_image_path = Path("stonehenge.png")
    # Create new uniform color image
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
    print("image size: ", image.size)
    image.save(temp_image_path)
    with open(temp_image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    temp_image_path.unlink()

    test_all = False
    if test_all:
        models_to_test = MODEL_NAME_TO_CLASS_MAP.keys()
    else:
        models_to_test = [
            "gemma3:12b"
        ]  # ["mistralai/mistral-small-3.2-24b-instruct:free"]

    for model_name in models_to_test:
        ##  Note: This would require actual API keys to run
        print(f"\nTesting model: {model_name}")
        try:
            completion = response_function(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "Describe the image in a short sentence.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            }
                        ],
                    },
                ],
            )
            print(completion.choices[0].message.content)
        except Exception as e:
            print(f"✗ Failed to create client for {model_name}: {e}")

        print("--------------------------------")

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv("/Users/vigji/code/ai-bubbles/config.env")
print(os.getenv("DEEPINFRA_TOKEN"))


class MockModelClient:
    """ensure in the shortest way that MockModelClient().chat.completions.create
    is a valid function that returns a response where response.choices[0].message.content is an empty string.
    """

    def __init__(self):
        self.model_name = "mock-model"
        self.chat = self

    @property
    def completions(self):
        return self

    @property
    def beta(self):
        return self

    def parse(self, **kwargs):
        return self.create(**kwargs)

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

    def batch_create(self, **kwargs):
        """Create a batch response with empty content and zero usage tokens."""
        return [self.create() for _ in range(len(kwargs.get("messages", [])))]


def get_model_client_response(required_model: str, force_openrouter=False):
    if "mock" in required_model or "test" in required_model:
        return MockModelClient()
    elif "gpt" in required_model and not force_openrouter:
        return OpenAI()
    elif "gemma3:12b" in required_model and not force_openrouter:
        return OpenAI(
            base_url="http://localhost:11434/v1",
        )
    elif "meta-llama/Llama-3.2" in required_model and not force_openrouter:
        return OpenAI(
            api_key=os.getenv("DEEPINFRA_TOKEN"),
            base_url="https://api.deepinfra.com/v1/openai",
        )
    elif [
        any(
            model in required_model
            for model in [
                "allenai/molmo-7b-d:free",
                "meta-llama/llama-4-scout",
                "bytedance-research/ui-tars-72b:free",
                "qwen/qwen2.5-vl-3b-instruct:free",
                "meta-llama/llama-4-maverick",
                "google/gemini-2.5-pro-exp-03-25:free",  # could be best for now!
                "google/gemma-3-27b-it:free",
                "anthropic/claude-3-haiku:beta",
            ]
        )
    ]:
        return OpenAI(
            api_key=os.getenv("OPENROUTER"),
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(f"Model {required_model} not supported")


def get_client_response_function(
    required_model: str, response_scheme=None, batch=False, force_openrouter=False
):
    client = get_model_client_response(required_model, force_openrouter)

    if required_model == "gpt-4o-mini" or "test" in required_model:
        response_function = (
            client.beta.chat.completions.batch_create
            if batch
            else client.beta.chat.completions.parse
        )
        response_format = response_scheme
    else:
        if batch:
            raise ValueError("Modify the code to support batch for this model")
        response_function = client.chat.completions.create
        if response_scheme is not None:
            response_format = {"type": "json_object"}

    return response_function, response_format


if __name__ == "__main__":
    model_name = "bytedance-research/ui-tars-72b:free"
    client = get_model_client_response(model_name)

    from pydantic import BaseModel

    class Description(BaseModel):
        description: str
        subject: str
        color: str

    completion = client.beta.chat.completions.parse(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can answer questions and help with tasks. When asked to describe an image, reply with a json scheme with the following fields: description, subject, color.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                        },
                    }
                ],
            },
        ],
        response_format={"type": "json_object"},
        # response_format=Description
    )
    print([c.message.content for c in completion.choices])

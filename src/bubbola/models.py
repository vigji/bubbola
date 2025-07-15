import os
from dataclasses import dataclass
from datetime import datetime, timedelta

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


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
    client_class = OpenAI
    base_url = None
    api_key_env_var = "OPENAI_API_KEY"


class LocalModel(LLMModel):
    client_class = OpenAI
    base_url = "http://localhost:11434/v1"


class DeepInfraModel(LLMModel):
    client_class = OpenAI
    base_url = "https://api.deepinfra.com/v1/openai"
    api_key_env_var = "DEEPINFRA_TOKEN"


class OpenRouterModel(LLMModel):
    client_class = OpenAI
    base_url = "https://openrouter.ai/api/v1"
    api_key_env_var = "OPENROUTER"


MODEL_NAME_TO_CLASS_MAP = {
    # OpenRouter models
    "meta-llama/llama-4-scout": OpenRouterModel,
    # Local models
    "gemma3:12b": LocalModel,
    # OpenAI models
    "gpt-4o": OpenAIModel,
    "gpt-4o-mini": OpenAIModel,
    "gpt-4": OpenAIModel,
    "gpt-o3-mini": OpenAIModel,
    "gpt-04-mini": OpenAIModel,
}

# Updated prices for 2025-07-15. define it as a constant
MODEL_PROCES_TIMESTAMP = datetime(2025, 7, 15)
MODEL_PRICES = {
    "meta-llama/llama-4-scout": {"in": 0.15, "out": 0.6},
    "gpt-4o-mini": {"in": 0.15, "out": 0.6},
    "gpt-4o": {"in": 2.5, "out": 10.0},
    "gpt-4.1": {"in": 2.00, "out": 8.00},
    "o4-mini": {"in": 1.1, "out": 4.40},
    "o3": {"in": 2, "out": 8},
}
PRICE_PER_X_TOKENS = 1_000_000


def get_per_token_price(model_name):
    # check prices no more than 6 months old
    if MODEL_PROCES_TIMESTAMP < datetime.now() - timedelta(days=180):  # 6 months
        print(
            "Model prices are more than 6 months old; price estimate will not be available"
        )
        return None

    if model_name not in MODEL_PRICES:
        print(
            f"Model {model_name} not found in MODEL_PRICES; price estimate will not be available"
        )
        return None

    return MODEL_PRICES[model_name]


def get_cost_estimate(model_name, prompt, completion_template):
    p = get_per_token_price(model_name)
    if p is None:
        return None
    return (prompt * p["in"] + completion_template * p["out"]) / PRICE_PER_X_TOKENS


if __name__ == "__main__":
    print(get_per_token_price("gpt-4o-mini"))
    print(get_cost_estimate("gpt-4o-mini", 1000000, 1000000))

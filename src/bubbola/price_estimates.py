import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from bubbola.model_creator import get_client_response_function

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


# ------------------------------------------------------------------
# Core heuristics (unchanged from previous snippet, lightly refactored)
# ------------------------------------------------------------------


@dataclass
class TokenEstimate:
    est: int
    low: int
    high: int
    scale_b64: float
    text_params: tuple[float, float]  # (chars_per_token, tokens_per_word)


def _count_words(text: str) -> int:
    return len(text.split())


def _estimate_text_tokens(
    text: str,
    chars_per_token: float = 4.0,
    tokens_per_word: float = 0.75,
) -> float:
    n_chars = len(text)
    if n_chars == 0:
        return 0.0
    n_words = _count_words(text)
    char_based = n_chars / chars_per_token
    word_based = n_words * tokens_per_word
    return max(char_based, word_based)


def _bounds_text_tokens(
    text: str,
    low_chars_per_token: float = 4.5,
    high_chars_per_token: float = 3.5,
) -> tuple[float, float]:
    n_chars = len(text)
    if n_chars == 0:
        return 0.0, 0.0
    low = n_chars / low_chars_per_token
    high = n_chars / high_chars_per_token
    return low, high


def estimate_tokens(
    text: str,
    image_b64: str,
    scale_b64: float = 0.85,
    chars_per_token: float = 4.0,
    tokens_per_word: float = 0.75,
    low_b64_scale: float = 0.7,
    high_b64_scale: float = 1.0,
    low_text_chars_per_token: float = 4.5,
    high_text_chars_per_token: float = 3.5,
) -> TokenEstimate:
    t_text = _estimate_text_tokens(
        text,
        chars_per_token=chars_per_token,
        tokens_per_word=tokens_per_word,
    )
    t_text_low, t_text_high = _bounds_text_tokens(
        text,
        low_chars_per_token=low_text_chars_per_token,
        high_chars_per_token=high_text_chars_per_token,
    )
    n_chars_b64 = len(image_b64)
    t_b64 = scale_b64 * n_chars_b64
    t_b64_low = low_b64_scale * n_chars_b64
    t_b64_high = high_b64_scale * n_chars_b64

    est = int(math.ceil(t_text + t_b64))
    low = int(math.ceil(t_text_low + t_b64_low))
    high = int(math.ceil(t_text_high + t_b64_high))

    return TokenEstimate(
        est=est,
        low=low,
        high=high,
        scale_b64=scale_b64,
        text_params=(chars_per_token, tokens_per_word),
    )


def calibrate_scale_b64(
    text: str,
    image_b64: str,
    actual_tokens: int,
    chars_per_token: float = 4.0,
    tokens_per_word: float = 0.75,
    min_scale: float = 0.5,
    max_scale: float = 1.1,
) -> float:
    n_chars_b64 = len(image_b64)
    if n_chars_b64 == 0:
        return (min_scale + max_scale) / 2.0
    t_text_est = _estimate_text_tokens(
        text,
        chars_per_token=chars_per_token,
        tokens_per_word=tokens_per_word,
    )
    residual = actual_tokens - t_text_est
    if residual <= 0:
        return min_scale
    raw_scale = residual / n_chars_b64
    return max(min_scale, min(max_scale, raw_scale))


def estimate_b64_chars_from_image_bytes(img_num_bytes: int) -> int:
    if img_num_bytes <= 0:
        return 0
    return 4 * math.ceil(img_num_bytes / 3)


def estimate_tokens_from_image_bytes(
    text: str,
    img_num_bytes: int,
    scale_b64: float = 0.85,
    **kwargs,
) -> TokenEstimate:
    n_chars_b64 = estimate_b64_chars_from_image_bytes(img_num_bytes)
    dummy_b64 = "A" * n_chars_b64
    return estimate_tokens(
        text=text,
        image_b64=dummy_b64,
        scale_b64=scale_b64,
        **kwargs,
    )


# ------------------------------------------------------------------
# OpenAI API calibration using model_creator interface
# ------------------------------------------------------------------


def _extract_prompt_tokens_from_usage(usage: Any) -> int | None:
    """
    Try several known usage shapes; return int or None.
    Known possibilities:
      usage.prompt_tokens
      usage.input_tokens
      usage.total_tokens (fallback if no prompt-specific field)
    Accept dict or obj w/ attributes.
    """
    if usage is None:
        return None
    # dict-like
    if isinstance(usage, dict):
        for k in (
            "prompt_tokens",
            "input_tokens",
            "prompt_tokens_total",
            "total_tokens",
        ):
            if k in usage and usage[k] is not None:
                return int(usage[k])
        return None
    # attr-like
    for k in ("prompt_tokens", "input_tokens", "prompt_tokens_total", "total_tokens"):
        if hasattr(usage, k):
            v = getattr(usage, k)
            if v is not None:
                return int(v)
    return None


if __name__ == "__main__":
    import base64
    from pathlib import Path

    import requests
    from PIL import Image

    temp_image_path = Path("stonehenge.png")

    model_name = (
        "gpt-4o"  # "mistralai/mistral-small-3.2-24b-instruct:free"  #  "gpt-4o-mini"  #
    )

    # Example calibration call (commented; requires valid API creds & real image data)

    response = requests.get(
        "https://commons.wikimedia.org/wiki/Special:FilePath/Stonehenge.jpg?width=100&format=png"
    )
    # image_base64 = base64.b64encode(response.content).decode("utf-8")

    image_long_edge = 756
    image = Image.open(
        "/Users/vigji/code/bubbola/tests/assets/single_pages/0088_001_001.png"
    )
    image.thumbnail((image_long_edge, image_long_edge))
    image.save(temp_image_path)
    with open(temp_image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        print("length of image_base64:", len(image_base64))

    response_function = get_client_response_function(model_name)

    actual_prompt_tokens: int | None = None

    # Build the user content with image
    data_uri = f"data:image/png;base64,{image_base64}"

    sample_text = ""  # "Please describe the contents of the following image."*4
    # Make the API call
    resp = response_function(
        model=model_name,
        messages=[
            # {"role": "system", "content": "You are a token counting probe. Respond briefly."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": sample_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                ],
            },
        ],
        max_tokens=3,
    )
    print(resp.choices[0].message.content)
    print("prompt tokens: ", resp.usage.prompt_tokens)
    actual_prompt_tokens = _extract_prompt_tokens_from_usage(resp.usage)
    # print(actual_prompt_tokens)

    est = estimate_tokens(sample_text, image_base64)
    print("Heuristic estimate:", est)

    temp_image_path.unlink()

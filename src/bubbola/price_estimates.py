import base64
import math
from datetime import datetime, timedelta
from io import BytesIO

from dotenv import load_dotenv
from PIL import Image

from bubbola.model_creator import get_client_response_function

load_dotenv("/Users/vigji/code/bubbola/config.env")

# Updated prices for 2025-07-15. define it as a constant
MODEL_PROCES_TIMESTAMP = datetime(2025, 7, 15)
MODEL_PRICES_PER_M_TOKEN = {
    "meta-llama/llama-4-scout": {"in": 0.15, "out": 0.6},
    "gpt-4o-mini": {"in": 0.15, "out": 0.6},
    "gpt-4o": {"in": 2.5, "out": 10.0},
    "gpt-4.1": {"in": 2.00, "out": 8.00},
    "o4-mini": {"in": 1.1, "out": 4.40},
    "o3": {"in": 2, "out": 8},
}


def get_per_token_price(model_name):
    # check prices no more than 6 months old
    if MODEL_PROCES_TIMESTAMP < datetime.now() - timedelta(days=180):  # 6 months
        print(
            "Model prices are more than 6 months old; price estimate will not be available"
        )
        return None

    if model_name not in MODEL_PRICES_PER_M_TOKEN:
        print(
            f"Model {model_name} not found in MODEL_PRICES; price estimate will not be available"
        )
        return None

    return {k: v / 1_000_000 for k, v in MODEL_PRICES_PER_M_TOKEN[model_name].items()}


def get_cost_estimate(model_name, prompt_tokens, completion_tokens):
    p = get_per_token_price(model_name)
    if p is None:
        return None
    return prompt_tokens * p["in"] + completion_tokens * p["out"]


def get_image_size_from_base64(b64_string: str) -> tuple[int, int]:
    # Remove any header like "data:image/png;base64,"
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]

    # Decode base64
    image_data = base64.b64decode(b64_string)

    # Open the image using Pillow
    with Image.open(BytesIO(image_data)) as img:
        return img.width, img.height


def calculate_image_tokens(
    width: int, height: int, model_name: str, detail: str = "high"
) -> int:
    """
    Calculate the number of tokens an image will consume for analysis, based on model and detail level.

    Parameters:
      width (int):  Original image width in pixels.
      height (int): Original image height in pixels.
      model_name (str): Name of the vision-capable model ("gpt-4o", "gpt-4o-mini", etc.).
      detail (str): One of "low", "high", or "auto" (default).
                    "auto" treats images <= 512×512 as low, else high.

    Returns:
      int: Total token count for the image input.
    """
    # Normalize inputs
    m = model_name.lower()
    d = detail.lower()

    # Determine token parameters per model
    if m in ("gpt-4o", "gpt-4-vision-preview", "gpt-4o-vision-preview", "vision-1"):
        base_tokens = 85  # low-detail flat cost :contentReference[oaicite:0]{index=0}
        tile_tokens = 170  # per 512×512 tile :contentReference[oaicite:1]{index=1}
    elif m in ("gpt-4o-mini",):
        base_tokens = 2833  # gpt-4o-mini low-detail flat cost :contentReference[oaicite:2]{index=2}
        tile_tokens = (
            5667  # gpt-4o-mini per tile cost :contentReference[oaicite:3]{index=3}
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    # Auto-detail logic: small images use low, others use high
    # if d == "auto":
    #     if max(width, height) <= 512:
    #         d = "low"
    #     else:
    #         d = "high"

    # Low-detail cost
    if d == "low":
        return base_tokens

    # High-detail cost: resize then tile-count
    # 1) Fit within 2048×2048
    scale1 = min(1.0, 2048 / max(width, height))
    w1, h1 = width * scale1, height * scale1
    # 2) Ensure shortest side is at most 768
    if min(w1, h1) > 768:
        scale2 = 768 / min(w1, h1)
        w1, h1 = w1 * scale2, h1 * scale2
    # 3) Count 512×512 tiles (round up)
    tiles_x = math.ceil(w1 / 512)
    tiles_y = math.ceil(h1 / 512)
    n_tiles = tiles_x * tiles_y

    # Total tokens = base + per_tile * n :contentReference[oaicite:4]{index=4}
    return base_tokens + tile_tokens * n_tiles


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


def estimate_tokens(
    text: str,
    image_b64: str,
    model_name: str,
    chars_per_token: float = 4.0,
    tokens_per_word: float = 0.75,
) -> int:
    t_text = _estimate_text_tokens(
        text,
        chars_per_token=chars_per_token,
        tokens_per_word=tokens_per_word,
    )

    w, h = get_image_size_from_base64(image_b64)
    t_image = calculate_image_tokens(w, h, model_name)

    return int(math.ceil(t_text + t_image))


if __name__ == "__main__":
    import base64
    from pathlib import Path

    import requests
    from PIL import Image

    temp_image_path = Path("stonehenge.png")

    model_name = "gpt-4o-mini"  # "mistralai/mistral-small-3.2-24b-instruct:free"  #  "gpt-4o-mini"  #

    # Example calibration call (commented; requires valid API creds & real image data)

    response = requests.get(
        "https://commons.wikimedia.org/wiki/Special:FilePath/Stonehenge.jpg?width=100&format=png"
    )
    # image_base64 = base64.b64encode(response.content).decode("utf-8")

    image_long_edge = 256
    image = Image.open(
        "/Users/vigji/code/bubbola/tests/assets/single_pages/0088_001_001.png"
    )
    image.thumbnail((image_long_edge, image_long_edge))
    print("image size: ", image.size)
    image.save(temp_image_path)
    with open(temp_image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    response_function = get_client_response_function(model_name)

    actual_prompt_tokens: int | None = None

    # Build the user content with image
    data_uri = f"data:image/png;base64,{image_base64}"

    sample_text = ""  # "Please describe the contents of the following image."*4
    max_tokens = 3
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
        max_tokens=max_tokens,
    )
    print("model name: ", model_name)
    actual_prompt_tokens = resp.usage.prompt_tokens
    print("actual prompt tokens: ", actual_prompt_tokens)

    est_prompt_tokens = estimate_tokens(sample_text, image_base64, model_name)
    print("Heuristic estimate:", est_prompt_tokens)

    image_tokens = calculate_image_tokens(image.width, image.height, model_name)
    print("image tokens: ", image_tokens)

    actual_cost_estimate = get_cost_estimate(
        model_name, actual_prompt_tokens, len(resp.choices[0].message.content.split())
    )
    print("actual cost estimate: ", actual_cost_estimate)

    est_cost_estimate = get_cost_estimate(model_name, est_prompt_tokens, max_tokens)
    print("estimated cost estimate: ", est_cost_estimate)

    print(resp.choices[0].message.content)

    temp_image_path.unlink()

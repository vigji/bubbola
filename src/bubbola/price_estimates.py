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
    Calculate the number of tokens an image will consume for analysis.

    Supports:
      • Patch-based models: 'gpt-4.1-mini', 'gpt-4.1-nano', 'o4-mini'
      • Tile-based models: 'gpt-4o', 'gpt-4.1', 'gpt-4o-mini',
                           'o3', 'o1', 'o1-pro', 'computer-use-preview', etc.

    Parameters:
      width (int):      Image width in pixels.
      height (int):     Image height in pixels.
      model_name (str): Case-insensitive model identifier.
      detail (str):     'low' or 'high'.  Ignored for patch‑based models.

    Returns:
      int: Total token count for this image input.
    """
    m = model_name.lower()
    d = detail.lower()

    # --- PATCH-BASED MODELS -------------------------------------------------
    if m in ("gpt-4.1-mini", "gpt-4.1-nano", "o4-mini"):
        # 1 patch = 32×32px
        patches_x = math.ceil(width / 32)
        patches_y = math.ceil(height / 32)
        tokens = patches_x * patches_y

        # if over budget, scale down to max 1536 patches
        if tokens > 1536:
            # shrink factor to hit exactly 1536 patches area
            sf = math.sqrt(1536 * (32**2) / (width * height))
            w1 = width * sf
            h1 = height * sf
            patches_x = math.ceil(w1 / 32)
            patches_y = math.ceil(h1 / 32)
            tokens = min(patches_x * patches_y, 1536)

        # apply model multiplier
        if m == "gpt-4.1-mini":
            tokens *= 1.62
        elif m == "gpt-4.1-nano":
            tokens *= 2.46
        else:  # o4-mini
            tokens *= 1.72

        return int(math.ceil(tokens))

    # --- TILE-BASED MODELS --------------------------------------------------
    # must specify detail
    if d not in ("low", "high"):
        raise ValueError("detail must be 'low' or 'high' for tile-based models")

    # lookup base & per-tile tokens
    # MODEL                BASE    PER-TILE
    # 4o, 4.1, 4.5          85      170
    # 4o-mini             2833     5667
    # o1, o1-pro, o3       75      150
    # computer-use-preview 65      129
    if m in ("gpt-4o", "gpt-4-vision-preview", "gpt-4.1", "gpt-4.5"):
        base, per_tile = 85, 170
    elif m in ("gpt-4o-mini",):
        base, per_tile = 2833, 5667
    elif m in ("o1", "o1-pro", "o3"):
        base, per_tile = 75, 150
    elif m == "computer-use-preview":
        base, per_tile = 65, 129
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    if d == "low":
        return base

    # detail == 'high'
    # 1) Fit within 2048×2048
    sf1 = min(1.0, 2048 / max(width, height))
    w1, h1 = width * sf1, height * sf1

    # 2) Ensure shortest side ≤ 768
    if min(w1, h1) > 768:
        sf2 = 768 / min(w1, h1)
        w1, h1 = w1 * sf2, h1 * sf2

    # 3) Count 512×512 tiles, round up
    tiles_x = math.ceil(w1 / 512)
    tiles_y = math.ceil(h1 / 512)
    n_tiles = tiles_x * tiles_y

    return base + per_tile * n_tiles


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
    import random
    from pathlib import Path

    from PIL import Image

    model_name = (
        "o3"  # "mistralai/mistral-small-3.2-24b-instruct:free"  #  "gpt-4o-mini"  #
    )

    image_long_edge = 256
    # image = Image.open(
    #     "/Users/vigji/code/bubbola/tests/assets/single_pages/0088_001_001.png"
    # )
    temp_image_path = Path("stonehenge.png")
    # Create new uniform color image
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

    response_function = get_client_response_function(model_name)

    actual_prompt_tokens: int | None = None

    # Build the user content with image
    data_uri = f"data:image/png;base64,{image_base64}"

    sample_text = "Please describe the contents of the following image."
    max_tokens = 51
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
        max_completion_tokens=max_tokens,
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

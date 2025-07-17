import base64
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
from typing import Any

from dotenv import load_dotenv
from PIL import Image

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
    "mistralai/mistral-small-3.2-24b-instruct:free": {"in": 0.0, "out": 0.0},
}


def _counts_from_response(response):
    try:
        input_tokens, output_tokens = (
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
    except AttributeError:
        input_tokens, output_tokens = (
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    return input_tokens, output_tokens


@dataclass
class TokenCounts:
    """Dataclass to track token counts and retry statistics."""

    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    retry_count: int = -1
    retry_input_tokens: int | None = None
    retry_output_tokens: int | None = None

    def add_attempt(self, response):
        """Add token counts from an attempt."""

        self.retry_count += 1

        if self.retry_count >= 1:
            # retry counts are the ones from previous attempts:
            self.retry_input_tokens = self.total_input_tokens
            self.retry_output_tokens = self.total_output_tokens

        input_tokens, output_tokens = _counts_from_response(response)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens


@dataclass
class AggregatedTokenCounts:
    """Dataclass to aggregate multiple TokenCounts with computed statistics."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_retry_count: int = 0
    total_retry_input_tokens: int = 0
    total_retry_output_tokens: int = 0
    num_images: int = 0

    def add_token_counts(self, token_counts: TokenCounts):
        """Add a TokenCounts instance to the aggregation."""
        self.total_input_tokens += token_counts.total_input_tokens
        self.total_output_tokens += token_counts.total_output_tokens
        self.total_retry_count += token_counts.retry_count
        self.total_retry_input_tokens += token_counts.retry_input_tokens
        self.total_retry_output_tokens += token_counts.retry_output_tokens
        self.num_images += 1

    @property
    def retry_probability(self) -> float:
        """Compute probability of retry across all images."""
        if self.num_images == 0:
            return 0.0
        return self.total_retry_count / self.num_images

    @property
    def retry_input_percentage(self) -> float:
        """Compute percentage of input tokens that were from retries."""
        if self.total_input_tokens == 0:
            return 0.0
        return (self.total_retry_input_tokens / self.total_input_tokens) * 100

    @property
    def retry_output_percentage(self) -> float:
        """Compute percentage of output tokens that were from retries."""
        if self.total_output_tokens == 0:
            return 0.0
        return (self.total_retry_output_tokens / self.total_output_tokens) * 100

    def print_summary(self):
        """Print a summary of the aggregated statistics."""
        print("\nAGGREGATED TOKEN STATISTICS:")
        print(f"Number of images processed: {self.num_images}")
        print(f"Total input tokens: {self.total_input_tokens}")
        print(f"Total output tokens: {self.total_output_tokens}")
        print(f"Total retry count: {self.total_retry_count}")
        print(f"Retry probability: {self.retry_probability:.2%}")
        print(
            f"Retry input tokens: {self.total_retry_input_tokens} ({self.retry_input_percentage:.1f}%)"
        )
        print(
            f"Retry output tokens: {self.total_retry_output_tokens} ({self.retry_output_percentage:.1f}%)"
        )


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


def _get_image_size_from_base64(b64_string: str) -> tuple[int, int]:
    # Remove any header like "data:image/png;base64,"
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]

    # Decode base64

    image_data = base64.b64decode(b64_string)

    # Open the image using Pillow
    with Image.open(BytesIO(image_data)) as img:
        return img.width, img.height


def _estimate_image_tokens_number(
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
    elif (
        m == "computer-use-preview"
        or m == "mistralai/mistral-small-3.2-24b-instruct:free"
    ):
        base, per_tile = 65, 129
    else:
        raise ValueError(f"No known image token calculation for model '{model_name}'")

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


def _estimate_text_tokens_number(
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


def estimate_total_tokens_number(
    text: str,
    image_b64: str,
    model_name: str,
    chars_per_token: float = 4.0,
    tokens_per_word: float = 0.75,
) -> int:
    t_text = _estimate_text_tokens_number(
        text,
        chars_per_token=chars_per_token,
        tokens_per_word=tokens_per_word,
    )

    w, h = _get_image_size_from_base64(image_b64)
    t_image = _estimate_image_tokens_number(w, h, model_name)

    return int(math.ceil(t_text + t_image))


def estimate_tokens_from_messages(
    messages: list[dict], model_name: str
) -> tuple[int | None, int | None]:
    """
    Estimate input and output tokens from a list of messages.

    Args:
        messages: List of message dictionaries
        model_name: Name of the model for token calculation

    Returns:
        Tuple of (estimated_input_tokens, estimated_output_tokens) or (None, None) if not supported
    """
    # Extract text content
    text_parts = []
    images = []

    for message in messages:
        content = message.get("content", "")

        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        url = item.get("image_url", {}).get("url", "")
                        if url.startswith("data:image/"):
                            if "," in url:
                                images.append(url.split(",", 1)[1])
                    elif item.get("type") == "input_image":
                        images.append(item.get("image_url", ""))

    text_content = " ".join(text_parts)

    # Estimate input tokens
    try:
        if images:
            estimated_input_tokens = estimate_total_tokens_number(
                text_content, images[0], model_name
            )
        else:
            estimated_input_tokens = int(_estimate_text_tokens_number(text_content))
    except (ValueError, Exception):
        # If estimation fails, return None
        estimated_input_tokens = None

    # For output tokens, we can't estimate without schema, so return None
    estimated_output_tokens = None

    return estimated_input_tokens, estimated_output_tokens


def estimate_tokens_from_messages_with_schema(
    messages: list[dict], model_name: str, response_schema: dict[str, Any]
) -> tuple[int | None, int | None]:
    """
    Estimate input and output tokens from messages and response schema.

    Args:
        messages: List of message dictionaries
        model_name: Name of the model for token calculation
        response_schema: JSON schema for the response format

    Returns:
        Tuple of (estimated_input_tokens, estimated_output_tokens) or (None, None) if not supported
    """
    input_tokens, _ = estimate_tokens_from_messages(messages, model_name)

    # Estimate output tokens from schema
    try:
        output_tokens = estimate_max_output_tokens_from_schema(response_schema)
    except Exception:
        output_tokens = None

    return input_tokens, output_tokens


def estimate_max_output_tokens_from_schema(response_format: dict[str, Any]) -> int:
    """
    Estimate the maximum output tokens needed based on the response format schema.

    This function analyzes the JSON schema to estimate token requirements:
    - Counts required and optional fields
    - Estimates tokens for different data types
    - Considers nested structures and arrays
    - Adds buffer for JSON formatting and field names

    Args:
        response_format: The JSON schema dictionary for the response format

    Returns:
        int: Estimated maximum output tokens needed
    """
    if not response_format or not isinstance(response_format, dict):
        return 1000  # Default fallback

    # Base tokens for JSON structure
    base_tokens = 50

    # Tokens for field names and structure
    structure_tokens = 0

    # Content tokens based on data types
    content_tokens = 0

    def analyze_schema(schema: dict[str, Any], path: str = "") -> None:
        nonlocal structure_tokens, content_tokens

        if "properties" in schema:
            properties = schema["properties"]

            for field_name, field_schema in properties.items():
                # Tokens for field name
                structure_tokens += (
                    len(field_name.split()) + 2
                )  # +2 for quotes and colon

                # Analyze field type
                field_type = field_schema.get("type", "string")
                description = field_schema.get("description", "")

                if field_type == "string":
                    # Estimate based on description or default length
                    if description:
                        # Use description length as a hint for expected content
                        content_tokens += min(len(description.split()) * 2, 50)
                    else:
                        content_tokens += 10  # Default string length
                elif field_type == "number" or field_type == "integer":
                    content_tokens += 5  # Numbers are typically short
                elif field_type == "boolean":
                    content_tokens += 1  # true/false
                elif field_type == "array":
                    # For arrays, analyze the items schema
                    items_schema = field_schema.get("items", {})
                    if items_schema:
                        # Estimate 3-5 items per array on average
                        avg_items = 4
                        content_tokens += avg_items * 20  # 20 tokens per array item
                        analyze_schema(items_schema, f"{path}.{field_name}")
                elif field_type == "object":
                    # Recursively analyze nested objects
                    analyze_schema(field_schema, f"{path}.{field_name}")

                # Add tokens for JSON formatting (commas, brackets, etc.)
                structure_tokens += 5

    # Analyze the main schema
    analyze_schema(response_format)

    # Calculate total estimated tokens
    total_tokens = base_tokens + structure_tokens + content_tokens

    # Add safety buffer (50% more than estimated)
    estimated_tokens = int(total_tokens * 1.5)

    # Ensure reasonable bounds
    min_tokens = 50
    max_tokens = 8000  # Increased for structured output parsing

    return max(min_tokens, min(estimated_tokens, max_tokens))


if __name__ == "__main__":
    import base64
    import random
    from pathlib import Path

    from model_creator import get_client_response_function
    from PIL import Image

    from bubbola.data_models import DeliveryNote

    output_schema = DeliveryNote.model_json_schema()

    est_max_output_tokens = estimate_max_output_tokens_from_schema(output_schema)
    print("estimated max output tokens: ", est_max_output_tokens)

    model_name = "mistralai/mistral-small-3.2-24b-instruct:free"  #  "gpt-4o-mini"  #

    image_long_edge = 248
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

    sample_text = "Please describe the contents of the following image:" * 10
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

    est_prompt_tokens = estimate_total_tokens_number(
        sample_text, image_base64, model_name
    )
    print("Heuristic estimate:", est_prompt_tokens)

    image_tokens = _estimate_image_tokens_number(image.width, image.height, model_name)
    print("image tokens: ", image_tokens)

    actual_cost_estimate = get_cost_estimate(
        model_name, actual_prompt_tokens, len(resp.choices[0].message.content.split())
    )
    print("actual cost estimate: ", actual_cost_estimate)

    est_cost_estimate = get_cost_estimate(model_name, est_prompt_tokens, max_tokens)
    print("estimated cost estimate: ", est_cost_estimate)

    print(resp.choices[0].message.content)

    temp_image_path.unlink()

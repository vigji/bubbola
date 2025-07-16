import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

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
# OpenAI API calibration
# ------------------------------------------------------------------


def _build_openai_user_content(
    text: str,
    image_b64: str,
    image_mime_type: str = "image/png",
    prefix: str = "",
    suffix: str = "",
) -> list:
    """
    Build a Responses API style content list (multi-part user message).
    prefix/suffix let you wrap the payload (e.g., instructions).
    """
    items = []
    if prefix:
        items.append({"type": "text", "text": prefix})
    # main text
    if text:
        items.append({"type": "text", "text": text})
    # image
    if image_b64:
        items.append(
            {
                "type": "input_image",
                "image_base64": image_b64,
                "mime_type": image_mime_type,
            }
        )
    if suffix:
        items.append({"type": "text", "text": suffix})
    return items


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


def _extract_usage_from_responses_obj(resp: Any) -> int | None:
    """
    Responses API returns resp.usage or resp.output[0].usage depending on version.
    We'll inspect a few places.
    """
    # direct usage attr?
    if hasattr(resp, "usage"):
        tok = _extract_prompt_tokens_from_usage(resp.usage)
        if tok is not None:
            return tok

    # Sometimes nested in response.output[]
    data = getattr(resp, "output", None)
    if data is not None:
        if isinstance(data, list):
            for item in data:
                usage = getattr(item, "usage", None)
                tok = _extract_prompt_tokens_from_usage(usage)
                if tok is not None:
                    return tok
                if isinstance(item, dict) and "usage" in item:
                    tok = _extract_prompt_tokens_from_usage(item["usage"])
                    if tok is not None:
                        return tok

    # Fallback to dict conversion
    try:
        import json

        obj = resp.model_dump() if hasattr(resp, "model_dump") else resp
        if not isinstance(obj, dict):
            obj = json.loads(str(resp))
        # walk
        stack = [obj]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                tok = _extract_prompt_tokens_from_usage(cur)
                if tok is not None:
                    return tok
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
    except Exception:
        pass
    return None


def _extract_usage_from_chat_obj(resp: Any) -> int | None:
    """
    Chat Completions API shape: resp.usage.prompt_tokens (preferred).
    """
    if hasattr(resp, "usage"):
        return _extract_prompt_tokens_from_usage(resp.usage)
    if isinstance(resp, dict) and "usage" in resp:
        return _extract_prompt_tokens_from_usage(resp["usage"])
    return None


def calibrate_via_openai(
    client: Any,
    model: str,
    text: str,
    image_b64: str,
    *,
    image_mime_type: str = "image/png",
    system: str = "You are a token counting probe. Respond briefly.",
    user_prefix: str = "",
    user_suffix: str = "Respond with a short acknowledgement.",
    max_output_tokens: int = 5,
    use_responses_api: bool = True,
    chars_per_token: float = 4.0,
    tokens_per_word: float = 0.75,
    min_scale: float = 0.5,
    max_scale: float = 1.1,
    **kwargs,
) -> tuple[float, TokenEstimate, int | None]:
    """
    Make a minimal OpenAI call to get *actual* prompt token usage for (text + image_b64).
    Returns (scale_b64, token_estimate, actual_prompt_tokens).

    Parameters
    ----------
    client : OpenAI or compatible client instance (already authenticated).
    model : str
    text : str
    image_b64 : str
    image_mime_type : str
    system, user_prefix, user_suffix : str
        Strings to wrap the calibration prompt if you want.
    max_output_tokens : int
        Keep tiny so calibration doesn’t waste quota.
    use_responses_api : bool
        If True, call `client.responses.create`; else use legacy chat completions.
    chars_per_token / tokens_per_word :
        Passed to `calibrate_scale_b64` + `estimate_tokens`.
    min_scale / max_scale :
        Clamp for calibration.
    **kwargs :
        Forwarded to API call (e.g., temperature=0).

    Returns
    -------
    scale_b64 : float
        Calibrated Base64 char->token scale.
    token_estimate : TokenEstimate
        Using the *new calibrated* scale.
    actual_prompt_tokens : Optional[int]
        Prompt token count reported by the API (None if unavailable).
    """
    actual_prompt_tokens: int | None = None

    if use_responses_api:
        # Build multi-part input
        content = _build_openai_user_content(
            text=text,
            image_b64=image_b64,
            image_mime_type=image_mime_type,
            prefix=user_prefix,
            suffix=user_suffix,
        )
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content},
            ],
            max_output_tokens=max_output_tokens,
            **kwargs,
        )
        actual_prompt_tokens = _extract_usage_from_responses_obj(resp)

    else:
        # Legacy chat.completions style (image support varies; using 'image_url' w/data URI)
        data_uri = f"data:{image_mime_type};base64,{image_b64}"
        user_content = []
        if user_prefix:
            user_content.append({"type": "text", "text": user_prefix})
        if text:
            user_content.append({"type": "text", "text": text})
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_uri},
            }
        )
        if user_suffix:
            user_content.append({"type": "text", "text": user_suffix})

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_output_tokens,
            **kwargs,
        )
        actual_prompt_tokens = _extract_usage_from_chat_obj(resp)

    # Derive scale (falls back gracefully if usage missing)
    if actual_prompt_tokens is None:
        scale_b64 = (min_scale + max_scale) / 2.0
    else:
        scale_b64 = calibrate_scale_b64(
            text=text,
            image_b64=image_b64,
            actual_tokens=actual_prompt_tokens,
            chars_per_token=chars_per_token,
            tokens_per_word=tokens_per_word,
            min_scale=min_scale,
            max_scale=max_scale,
        )

    # Produce TokenEstimate w/ calibrated scale
    token_estimate = estimate_tokens(
        text=text,
        image_b64=image_b64,
        scale_b64=scale_b64,
        chars_per_token=chars_per_token,
        tokens_per_word=tokens_per_word,
    )
    return scale_b64, token_estimate, actual_prompt_tokens


# ------------------------------------------------------------------
# Async variant
# ------------------------------------------------------------------


async def async_calibrate_via_openai(
    client: Any,
    model: str,
    text: str,
    image_b64: str,
    *,
    image_mime_type: str = "image/png",
    system: str = "You are a token counting probe. Respond briefly.",
    user_prefix: str = "",
    user_suffix: str = "Respond with a short acknowledgement.",
    max_output_tokens: int = 5,
    use_responses_api: bool = True,
    chars_per_token: float = 4.0,
    tokens_per_word: float = 0.75,
    min_scale: float = 0.5,
    max_scale: float = 1.1,
    **kwargs,
) -> tuple[float, TokenEstimate, int | None]:
    """
    Async wrapper; same semantics as calibrate_via_openai.
    """
    actual_prompt_tokens: int | None = None

    if use_responses_api:
        content = _build_openai_user_content(
            text=text,
            image_b64=image_b64,
            image_mime_type=image_mime_type,
            prefix=user_prefix,
            suffix=user_suffix,
        )
        resp = await client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content},
            ],
            max_output_tokens=max_output_tokens,
            **kwargs,
        )
        actual_prompt_tokens = _extract_usage_from_responses_obj(resp)

    else:
        data_uri = f"data:{image_mime_type};base64,{image_b64}"
        user_content = []
        if user_prefix:
            user_content.append({"type": "text", "text": user_prefix})
        if text:
            user_content.append({"type": "text", "text": text})
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_uri},
            }
        )
        if user_suffix:
            user_content.append({"type": "text", "text": user_suffix})

        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_output_tokens,
            **kwargs,
        )
        actual_prompt_tokens = _extract_usage_from_chat_obj(resp)

    if actual_prompt_tokens is None:
        scale_b64 = (min_scale + max_scale) / 2.0
    else:
        scale_b64 = calibrate_scale_b64(
            text=text,
            image_b64=image_b64,
            actual_tokens=actual_prompt_tokens,
            chars_per_token=chars_per_token,
            tokens_per_word=tokens_per_word,
            min_scale=min_scale,
            max_scale=max_scale,
        )

    token_estimate = estimate_tokens(
        text=text,
        image_b64=image_b64,
        scale_b64=scale_b64,
        chars_per_token=chars_per_token,
        tokens_per_word=tokens_per_word,
    )
    return scale_b64, token_estimate, actual_prompt_tokens


# ------------------------------------------------------------------
# CLI-ish demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal smoke test (won't call API unless you uncomment).
    sample_text = "Please describe the contents of the following image."
    sample_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAUA" * 1000  # fake data

    # Local heuristic estimate
    est = estimate_tokens(sample_text, sample_b64)
    print("Heuristic estimate:", est)

    # Example calibration call (commented; requires valid OpenAI creds & real image data)
    """
    from openai import OpenAI
    client = OpenAI()  # assumes env var OPENAI_API_KEY set

    scale, est2, actual = calibrate_via_openai(
        client=client,
        model="gpt-4o-mini",
        text=sample_text,
        image_b64=sample_b64,
        image_mime_type="image/png",
        user_suffix="Just say 'ok'.",
        max_output_tokens=1,
        temperature=0,
    )
    print("Actual prompt tokens:", actual)
    print("Calibrated scale:", scale)
    print("Re-estimate:", est2)
    """

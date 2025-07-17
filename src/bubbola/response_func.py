from pydantic import BaseModel

from bubbola.model_creator import get_model_client

if __name__ == "__main__":
    import base64
    import random
    from pathlib import Path

    from PIL import Image

    from bubbola.model_creator import MODEL_NAME_TO_CLASS_MAP

    # Create test image
    temp_image_path = Path("stonehenge.png")
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

    # Test configuration
    models_to_test = list(
        MODEL_NAME_TO_CLASS_MAP.keys()
    )  # Test mock + first 2 real models

    class ImageDescription(BaseModel):
        description: str
        main_color: str

    for model_name in models_to_test:
        print(f"\nTesting model: {model_name}")
        try:
            model_object = get_model_client(model_name)
            print(f"✓ Direct model class access works for {model_name}")

            # Test unified message creation
            try:
                instructions = "Describe the image in a short sentence. Return a JSON object with the description and the main color."
                messages = model_object.create_messages(instructions, [image_base64])
                print(f"✓ Unified message creation works for {model_name}")
                print(f"  Created {len(messages)} messages")
                for i, msg in enumerate(messages):
                    print(
                        f"    Message {i}: role={msg['role']}, content_type={type(msg['content'])}"
                    )
            except Exception as e:
                print(f"✗ Unified message creation failed for {model_name}: {e}")

            # Test dry_run functionality
            try:
                # Test with Pydantic model and dry_run=True
                result, token_counts = model_object.query_with_instructions(
                    instructions="Describe the image briefly.",
                    images=[image_base64],
                    pydantic_model=ImageDescription,
                    dry_run=True,
                )
                print(f"✓ Dry run functionality works for {model_name}")
                print(f"  Result type: {type(result)}")
                print(f"  Result: {result}")
                print(f"  Token counts: {token_counts}")
                print(f"  Estimated input tokens: {token_counts.total_input_tokens}")
                print(f"  Estimated output tokens: {token_counts.total_output_tokens}")

            except Exception as e:
                print(f"✗ Dry run functionality failed for {model_name}: {e}")

            # Test unified query method (would require API keys to actually work)
            try:
                # Test with Pydantic model
                result, token_counts = model_object.query_with_instructions(
                    instructions="Describe the image briefly.",
                    images=[image_base64],
                    pydantic_model=ImageDescription,
                )
                print(f"✓ Unified query with Pydantic model works for {model_name}")
                print(f"  Result type: {type(result)}")
                print(f"  Result: {result}")
                print(f"  Token counts: {token_counts}")

            except Exception as e:
                print(f"✗ Unified query method failed for {model_name}: {e}")

            # Test individual methods still work
            try:
                image_message = model_object.format_image_message(image_base64)
                print(f"✓ Individual format_image_message still works for {model_name}")

                simple_response_func = model_object.get_simple_response

                parsed_response_func = model_object.get_parsed_response
                print(
                    f"✓ Individual get_parsed_response method available: {type(parsed_response_func)}"
                )
            except Exception as e:
                print(f"✗ Individual methods failed for {model_name}: {e}")

        except Exception as e:
            print(f"✗ Direct model class access failed for {model_name}: {e}")

        print("--------------------------------")

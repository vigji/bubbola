from model_creator import get_model_client
from pydantic import BaseModel

# 1. Tell the client where Ollama is listening:# 2. Provide any non‑empty API key (Ollama ignores its value):

# 3. Call it just like a normal ChatCompletion:
# client = openai.OpenAI(base_url="http://127.0.0.1:11434/v1") #
client = get_model_client("gemma3:12b")
print(client)

# For simple chat completions without structured output, use the regular create method:
resp = client.chat.completions.create(
    model="gemma3:12b",  # or your chosen model name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say Hello!"},
    ],
    max_tokens=10,  # Note: use max_tokens, not max_completion_tokens
)
print(resp.choices[0].message.content)

client = get_model_client("gemma3:12b")
print(client)


class SimpleResponse(BaseModel):
    message: str


# For simple chat completions without structured output, use the regular create method:
resp = client.chat.completions.create(
    model="gemma3:12b",  # or your chosen model name
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. You reply only with correctly formatted JSON.",
        },
        {
            "role": "user",
            "content": "Give me a JSON structured with a single key 'message' and a value of 'Hello!'",
        },
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "SimpleResponse",
            "schema": SimpleResponse.model_json_schema(),
        },
    },
    max_tokens=100,  # Note: use max_tokens, not max_completion_tokens
)
print(resp.choices[0].message.content)

# For structured outputs with the newer responses.parse API:
# from pydantic import BaseModel

# parsing_client = OpenAI(base_url="http://127.0.0.1:11434/v1")


# resp = client.responses.parse(
#     model="gemma3:12b",
#     input=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user",   "content": "Say Hello!"}
#     ],
#     text_format=SimpleResponse,
#     max_output_tokens=10
# )

# print(resp.choices[0].message.parsed.message)

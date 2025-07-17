from model_creator import get_client_response_function_with_schema, get_model_client
from pydantic import BaseModel

# 1. Tell the client where Ollama is listening:# 2. Provide any non‑empty API key (Ollama ignores its value):

# 3. Call it just like a normal ChatCompletion:
# client = openai.OpenAI(base_url="http://127.0.0.1:11434/v1") #
model_name = "gemma3:12b"
client = get_model_client(model_name)
print(client)

# For simple chat completions without structured output, use the regular create method:
resp = client.chat.completions.create(
    model=model_name,  # or your chosen model name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say Hello!"},
    ],
    max_tokens=10,  # Note: use max_tokens, not max_completion_tokens
)
print(resp.choices[0].message.content)

client = get_model_client(model_name)
print(client)


class SimpleResponse(BaseModel):
    message: str


messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. You reply only with correctly formatted JSON.",
    },
    {
        "role": "user",
        "content": "Give me a JSON structured with a single key 'message' and a value of 'Hello!'",
    },
]
# For simple chat completions without structured output, use the regular create method:
resp = client.chat.completions.create(
    model=model_name,  # or your chosen model name
    messages=messages,
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

f = get_client_response_function_with_schema(
    model_name,
    SimpleResponse,
    # max_tokens=100,
)
resp_2 = f(messages, SimpleResponse)
print(resp_2)

# For structured outputs with the newer responses.parse API:
# from pydantic import BaseModel

model_name = "gpt-4o-mini"
parsing_client = get_model_client(model_name)


response = parsing_client.responses.parse(
    model=model_name, input=messages, text_format=SimpleResponse, max_output_tokens=100
)

parsed_output = response.output_parsed

print(parsed_output)
f = get_client_response_function_with_schema(
    model_name,
    SimpleResponse,
    # max_tokens=100,
)
resp_2 = f(messages, SimpleResponse)
print(resp_2)

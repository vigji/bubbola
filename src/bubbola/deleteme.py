from model_creator import (
    _validate_response_content,
    get_client_response_function_with_schema,
)
from openai.types.chat import ChatCompletion
from openai.types.responses import ParsedResponse
from pydantic import BaseModel

# 1. Tell the client where Ollama is listening:# 2. Provide any non‑empty API key (Ollama ignores its value):

# 3. Call it just like a normal ChatCompletion:
# client = openai.OpenAI(base_url="http://127.0.0.1:11434/v1") #
# model_name = "gemma3:12b"
# client = get_model_client(model_name)
# print(client)

# # For simple chat completions without structured output, use the regular create method:
# resp = client.chat.completions.create(
#     model=model_name,  # or your chosen model name
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Say Hello!"},
#     ],
#     max_tokens=10,  # Note: use max_tokens, not max_completion_tokens
# )
# print(resp.choices[0].message.content)

# client = get_model_client(model_name)
# print(client)


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

model_name = "gemma3:12b"

f = get_client_response_function_with_schema(
    model_name,
    SimpleResponse,
    # max_tokens=100,
)

resp_2 = f(messages)
print(type(resp_2), isinstance(resp_2, ChatCompletion))
print(_validate_response_content(resp_2.choices[0].message.content, SimpleResponse))
# print(resp_2.usage.prompt_tokens, resp_2.usage.completion_tokens)
# print(type(resp_2.choices[0].message.content))
# print(resp_2.choices[0].message.content)
# r = SimpleResponse.model_validate_json(resp_2.choices[0].message.content)

# print(r)
# print(type(r))

# For structured outputs with the newer responses.parse API:
# from pydantic import BaseModel

model_name = "gpt-4o-mini"

f = get_client_response_function_with_schema(
    model_name,
    SimpleResponse,
    # max_tokens=100,
)
resp_2 = f(messages)
# print(resp_2)
print(type(resp_2), isinstance(resp_2, ParsedResponse))

print(resp_2.output_parsed)


# print(resp_2.usage.input_tokens, resp_2.usage.input_tokens_details)
# print(resp_2.usage.output_tokens, resp_2.usage.total_tokens)

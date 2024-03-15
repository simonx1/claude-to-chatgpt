import httpx
import time
import json
import os
from fastapi import Request
from claude_to_chatgpt.util import num_tokens_from_string
from claude_to_chatgpt.logger import logger
from claude_to_chatgpt.models import model_map

role_map = {
    "system": "Human",
    "user": "Human",
    "assistant": "Assistant",
}

stop_reason_map = {
    "stop_sequence": "stop",
    "max_tokens": "length",
}


class ClaudeAdapter:
    def __init__(self, claude_base_url="https://api.anthropic.com"):
        self.claude_api_key = os.getenv("CLAUDE_API_KEY", None)
        self.claude_base_url = claude_base_url

    def get_api_key(self, headers):
        auth_header = headers.get("authorization", None)
        if auth_header:
            return auth_header.split(" ")[1]
        else:
            return self.claude_api_key


    def openai_to_claude_params(self, openai_params):
        model = model_map.get(openai_params["model"], "claude-3-opus-20240229")
        messages = openai_params["messages"]

        claude_params = {
            "model": model,
            "messages": messages,
            "max_tokens": 1024,
        }

        if openai_params.get("max_tokens"):
            claude_params["max_tokens_to_sample"] = openai_params["max_tokens"]

        if openai_params.get("stop"):
            claude_params["stop_sequences"] = openai_params.get("stop")

        if openai_params.get("temperature"):
            claude_params["temperature"] = openai_params.get("temperature")

        if openai_params.get("stream"):
            claude_params["stream"] = False

        return claude_params

    def claude_to_chatgpt_response_stream(self, claude_response):
        completion = claude_response.get("completion", "")
        completion_tokens = num_tokens_from_string(completion)
        openai_response = {
            "id": f"chatcmpl-{str(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "gpt-3.5-turbo-0613",
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": completion_tokens,
                "total_tokens": completion_tokens,
            },
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "content": completion,
                    },
                    "index": 0,
                    "finish_reason": stop_reason_map[claude_response.get("stop_reason")]
                    if claude_response.get("stop_reason")
                    else None,
                }
            ],
        }

        return openai_response

# {'id': 'msg_01GPzbZ4QAMKLDuswts8PUoA', 'type': 'message', 'role': 'assistant', 'content': [{'type': 'text', 'text': '2 + 2 = 4\n\nThis is a simple addition problem. When you add two and two together, the result is four.'}], 'model': 'claude-3-opus-20240229', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 12, 'output_tokens': 34}}
#{'id': 'msg_01BoswanJPiD1F1eSBLBR1vs', 'type': 'message', 'role': 'assistant', 'content': [{'type': 'text', 'text': '2+2=4'}], 'model': 'claude-3-opus-20240229', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 12, 'output_tokens': 9}}
    def claude_to_chatgpt_response(self, claude_response):
        input_tokens = claude_response.get("usage", {}).get("input_tokens", 0)
        output_tokens = claude_response.get("usage", {}).get("output_tokens", 0)

        openai_response = {
            "id": f"chatcmpl-{str(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gpt-3.5-turbo-0613",
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": claude_response.get("content", [])[0].get("text"),
                    },
                    "index": 0,
                    "finish_reason": "stop"
                    if claude_response.get("stop_reason")
                    else None,
                }
            ],
        }

        return openai_response

    async def chat(self, request: Request):
        openai_params = await request.json()
        headers = request.headers
        claude_params = self.openai_to_claude_params(openai_params)
        api_key = self.get_api_key(headers)

        async with httpx.AsyncClient(timeout=120.0) as client:
            if not claude_params.get("stream", False):
                print(claude_params)
                response = await client.post(
                    f"{self.claude_base_url}/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "accept": "application/json",
                        "content-type": "application/json",
                        "anthropic-version": "2023-06-01",
                    },
                    json=claude_params,
                )
                if response.is_error:
                    raise Exception(f"Error: {response.status_code}")
                claude_response = response.json()
                print(claude_response)
                openai_response = self.claude_to_chatgpt_response(claude_response)
                yield openai_response
            else:
                print(claude_params)
                async with client.stream(
                    "POST",
                    f"{self.claude_base_url}/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "accept": "application/json",
                        "content-type": "application/json",
                        "anthropic-version": "2023-06-01",
                    },
                    json=claude_params,
                ) as response:
                    if response.is_error:
                        raise Exception(f"Error: {response.status_code}")
                    async for line in response.aiter_lines():
                        print(line)
                        if line:
                            stripped_line = line.lstrip("data:")
                            if stripped_line:
                                try:
                                    decoded_line = json.loads(stripped_line)
                                    stop_reason = decoded_line.get("stop_reason")
                                    if stop_reason:
                                        yield self.claude_to_chatgpt_response_stream(
                                            {
                                                "completion": "",
                                                "stop_reason": stop_reason,
                                            }
                                        )
                                        yield "[DONE]"
                                    else:
                                        completion = decoded_line.get("completion")
                                        if completion:
                                            openai_response = (
                                                self.claude_to_chatgpt_response_stream(
                                                    decoded_line
                                                )
                                            )
                                            yield openai_response
                                except json.JSONDecodeError as e:
                                    logger.debug(
                                        f"Error decoding JSON: {e}"
                                    )  # Debug output
                                    logger.debug(
                                        f"Failed to decode line: {stripped_line}"
                                    )  # Debug output

from heapq import merge
import httpx
import time
import json
import os
from fastapi import Request
from claude_to_chatgpt.util import num_tokens_from_string
from claude_to_chatgpt.logger import logger

role_map = {
    "system": "Human",
    "user": "Human",
    "assistant": "Assistant",
}

stop_reason_map = {
    "stop_sequence": "stop",
    "max_tokens": "length",
}


#{'model': 'claude-3-opus-20240229', 'messages': [{'content': "You are a helpful AI assistant. Solve tasks using your coding and language skills. In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute. 1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself. 2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly. Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill. When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user. If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user. If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try. When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible. Reply 'TERMINATE' in the end when everything is done.", 'role': 'system'}, {'content': 'Solve this problem: "That lawyer is my brother" testified the accountant. But the lawyer testified he didn\'t have a brother. Who is lying?', 'name': 'userproxy', 'role': 'user'}], 'max_tokens': 1024, 'temperature': 0.1}


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
        
    def merge_messages(self, messages):
        new_messages = []
        prev_role = None
        prev_content = None

        for message in messages:
            curr_role = message['role']
            curr_content = message['content']

            if curr_role == prev_role:
                prev_content += ' ' + curr_content
            else:
                if prev_role is not None:
                    new_messages.append({'role': prev_role, 'content': prev_content})
                prev_role = curr_role
                prev_content = curr_content

        if prev_role is not None:
            new_messages.append({'role': prev_role, 'content': prev_content})

        return new_messages   


    def openai_to_claude_params(self, openai_params):
        model = openai_params["model"]
        messages = openai_params["messages"]

        for message in messages:
            message.pop("name", None)

        system_message = None
        for message in messages:
            if message['role'] == 'system':
                system_message = message['content']
                messages.remove(message)
                break            

        claude_params = {
            "model": model,
            "messages": self.merge_messages(messages),
            "max_tokens": 4096,
        }

        if system_message:
            claude_params["system"] = system_message

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

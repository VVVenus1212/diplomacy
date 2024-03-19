"""
A simple wrapper for the official ChatGPT API
"""
import argparse
import json
import os
import sys

import requests
#import tiktoken
import openai
import time

from .utils import create_completer, create_keybindings
from .utils import create_session
from .utils import get_input
import logging

class Chatbot:
    """
    Official ChatGPT API
    """

    def __init__(
        self,
        api_key: str,
        engine: str = os.environ.get("GPT_ENGINE") or "gpt-4-0125-preview",
        proxy: str = None,
        max_tokens: int = 3000,
        temperature: float = 0.5,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        reply_count: int = 1,
        system_prompt: str = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally",
        logger: logging.Logger = None,
    ) -> None:
        """
        Initialize Chatbot with API key (from https://platform.openai.com/account/api-keys)
        """
        self.engine = engine
        self.session = requests.Session()
        self.api_key = api_key
        self.proxy = 'http://127.0.0.1:8118'

        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.reply_count = reply_count
        self.logger = logger

        if self.proxy:
            proxies = {
                "http": self.proxy,
                "https": self.proxy,
            }
            self.session.proxies = proxies
        self.conversation: dict = {
            "default": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
            ],
        }
        if max_tokens > 4000:
            raise Exception("Max tokens cannot be greater than 4000")

        if self.get_token_count("default") > self.max_tokens:
            raise Exception("System prompt is too long")

    def add_to_conversation(
        self,
        message: str,
        role: str,
        convo_id: str = "default",
    ) -> None:
        """
        Add a message to the conversation
        """
        self.conversation[convo_id].append({"role": role, "content": message})

    def truncate_conversation(self, convo_id: str = "default") -> None:
        """
        Truncate the conversation
        """
        while True:
            if (
                self.get_token_count(convo_id) > self.max_tokens and
                len(self.conversation[convo_id]) > 1
            ):
                # Don't remove the first message
                current_conversation = self.conversation[convo_id][-1]
                while(len(self.conversation[convo_id]) > 1):
                    self.conversation[convo_id].pop(1)
                #self.ask(self.info, convo_id=convo_id)
                self.conversation[convo_id].append({'role': 'user', 'content': self.info})
                self.conversation[convo_id].append(current_conversation)
            else:
                break

    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def get_token_count(self, convo_id: str = "default") -> int:
        """
        Get token count
        """

        # if self.engine not in ["gpt-3.5-turbo", "gpt-3.5-turbo-0301"]:
        #     raise NotImplementedError("Unsupported engine {self.engine}")

        #encoding = tiktoken.encoding_for_model(self.engine)

        num_tokens = 0
        for message in self.conversation[convo_id]:
            # every message follows <im_start>{role/name}\n{content}<im_end>\n
            num_tokens += 4
            for key, value in message.items():
                #num_tokens += len(encoding.encode(value))
                num_tokens += len(value.split(' '))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    def get_max_tokens(self, convo_id: str) -> int:
        """
        Get max tokens
        """
        return self.max_tokens - self.get_token_count(convo_id)

    def ask_stream(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        **kwargs,
    ) -> str:
        """
        Ask a question
        """
        # Make conversation if it doesn't exist
        if convo_id not in self.conversation:
            self.reset(convo_id=convo_id, system_prompt=self.system_prompt)
        self.add_to_conversation(prompt, "user", convo_id=convo_id)
        self.truncate_conversation(convo_id=convo_id)
        #self.logger.info('[QUERY] {}'.format(self.conversation[convo_id]))
        # Get response
        response = self.session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {kwargs.get('api_key', self.api_key)}"},
            json={
                "model": self.engine,
                "messages": self.conversation[convo_id],
                "stream": True,
                # kwargs
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "presence_penalty": kwargs.get(
                    "presence_penalty", self.presence_penalty
                ),
                "frequency_penalty": kwargs.get(
                    "frequency_penalty", self.frequency_penalty
                ),
                "n": kwargs.get("n", self.reply_count),
                "user": role,
                #"max_tokens": self.get_max_tokens(convo_id=convo_id),
            },
            stream=True,
        )
        if response.status_code != 200:
            logging.warning(f"{response.status_code} {response.reason} {response.text}")

            if response.status_code == 429:
                self.rollback(1, convo_id=convo_id)
                raise requests.exceptions.RequestException
            else:
                self.rollback(1, convo_id=convo_id)
                #self.rollback(len(self.conversation) - 1, convo_id=convo_id)
                raise requests.exceptions.RequestException
        '''
            raise Exception(
            f"Error: {response.status_code} {response.reason} {response.text}",
        )'''
        '''
        if(len(self.conversation) >= 2):
            self.rollback(1, convo_id=convo_id)
        '''
        response_role: str = None
        full_response: str = ""
        for line in response.iter_lines():
            if not line:
                continue
            # Remove "data: "
            line = line.decode("utf-8")[6:]
            if line == "[DONE]":
                break
            resp: dict = json.loads(line)
            choices = resp.get("choices")
            if not choices:
                continue
            delta = choices[0].get("delta")
            if not delta:
                continue
            if "role" in delta:
                response_role = delta["role"]
            if "content" in delta:
                content = delta["content"]
                full_response += content
                yield content
        self.add_to_conversation(full_response, response_role, convo_id=convo_id)

    def ask(
        self,
        prompt: str,
        role: str = "user",
        convo_id: str = "default",
        **kwargs,
    ) -> str:
        """
        Non-streaming ask
        """
        response = self.ask_stream(
            prompt=prompt,
            role=role,
            convo_id=convo_id,
            **kwargs,
        )
        full_response: str = "".join(response)
        return full_response

    def reply(self, query, convo_id="default", retry_count=0):
        try:
            response = self.ask(query, convo_id=convo_id)
            logging.info("[ChatGPT]\n{}".format(response))
            return response

        except requests.exceptions.RequestException as e:
            # rate limit exception
            #self.logger.warning(e)
            if retry_count < 3:
                time.sleep(30)
                logging.warning("[ChatGPT] RateLimit exceed, 第{}次重试".format(retry_count + 1))
                return self.reply(query, retry_count=retry_count + 1)
            else:
                return "Request Too Fast"
        except requests.exceptions.ConnectionError as e:
            logging.warning(e)
            return "代理错误"

    def rollback(self, n: int = 1, convo_id: str = "default") -> None:
        """
        Rollback the conversation
        """
        for _ in range(n):
            self.conversation[convo_id].pop()

    def reset(self, convo_id: str = "default", system_prompt: str = None) -> None:
        """
        Reset the conversation
        """
        self.conversation[convo_id] = [
            {"role": "system", "content": system_prompt or self.system_prompt},
        ]

    def save(self, file: str, *convo_ids: str) -> bool:
        """
        Save the conversation to a JSON file
        """
        try:
            with open(file, "w", encoding="utf-8") as f:
                if convo_ids:
                    json.dump({k: self.conversation[k] for k in convo_ids}, f, indent=2)
                else:
                    json.dump(self.conversation, f, indent=2)
        except (FileNotFoundError, KeyError):
            return False
        return True
        # print(f"Error: {file} could not be created")

    def load(self, file: str, *convo_ids: str) -> bool:
        """
        Load the conversation from a JSON  file
        """
        try:
            with open(file, encoding="utf-8") as f:
                if convo_ids:
                    convos = json.load(f)
                    self.conversation.update({k: convos[k] for k in convo_ids})
                else:
                    self.conversation = json.load(f)
        except (FileNotFoundError, KeyError, json.decoder.JSONDecodeError):
            return False
        return True

    def load_config(self, file: str, no_api_key: bool = False) -> bool:
        """
        Load the configuration from a JSON file
        """
        try:
            with open(file, encoding="utf-8") as f:
                config = json.load(f)
                if config is not None:
                    self.api_key = config.get("api_key") or self.api_key
                    if self.api_key is None:
                        # Make sure the API key is set
                        raise Exception("Error: API key is not set")
                    self.engine = config.get("engine") or self.engine
                    self.temperature = config.get("temperature") or self.temperature
                    self.top_p = config.get("top_p") or self.top_p
                    self.presence_penalty = (
                        config.get("presence_penalty") or self.presence_penalty
                    )
                    self.frequency_penalty = (
                        config.get("frequency_penalty") or self.frequency_penalty
                    )
                    self.reply_count = config.get("reply_count") or self.reply_count
                    self.max_tokens = config.get("max_tokens") or self.max_tokens

                    if config.get("system_prompt") is not None:
                        self.system_prompt = (
                            config.get("system_prompt") or self.system_prompt
                        )
                        self.reset(system_prompt=self.system_prompt)

                    if config.get("proxy") is not None:
                        self.proxy = config.get("proxy") or self.proxy
                        proxies = {
                            "http": self.proxy,
                            "https": self.proxy,
                        }
                        self.session.proxies = proxies
        except (FileNotFoundError, KeyError, json.decoder.JSONDecodeError):
            return False
        return True
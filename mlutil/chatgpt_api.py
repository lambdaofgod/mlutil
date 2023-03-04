import openai
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import json
import os

from pathlib import Path
import datetime


@dataclass
class LoggingArgs:

    logging_dir = "~/.chatgpt_logs"
    date_format = "%y-%m-%d-%a-%H-%p"


class ChatGPTClient:
    def __init__(
        self,
        api_key_path: Optional[str],
        model_name: str = "gpt-3.5-turbo",
        logger=lambda s: None,
    ):
        self.model_name = model_name
        if api_key_path is None:
            api_key = os.getenv("OPENAI_API_KEY")
            assert (
                api_key is not None
            ), "no api key path specified, failed loading api key from var"
        else:
            api_key = open(api_key_path).read().strip()
        openai.api_key = api_key
        self.logger = logger

    def get_chatgpt_response_from_text(self, text, **kwargs):
        messages = [{"role": "user", "content": text}]
        self.logger("calling OpenAI API")
        self.logger(json.dumps(kwargs))
        return self.get_chatgpt_response(messages, **kwargs)

    def get_chatgpt_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 128,
        # sampling temperature
        temperature: float = 1,
        # nucleus sampling arg
        top_p: int = 1,
        # n returned sequences
        n: int = 1,
        # penalizes already present tokens
        presence_penalty: float = 0,
        # penalizes token repetition
        frequency_penalty: float = 0,
        # can make some tokens less probable
        logit_bias: Dict[str, int] = dict(),
        stream: bool = False,
    ):
        self.logger(json.dumps(messages))
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=n,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            stream=stream,
        )
        self.logger(json.dumps(completion))
        return completion["choices"][0]["message"]["content"]

    def setup_default_info_logger(self, args: LoggingArgs):
        log_files_prefix = self.model_name
        file_suffix = datetime.datetime.now().strftime()
        log_path = (
            Path(logging_dir) / f"{log_files_prefix}/{date_format}.txt"
        ).expanduser()
        logging.basicConfig(filename=log_path, level="INFO")
        return logging.info

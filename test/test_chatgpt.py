from mlutil import chatgpt_api
from dataclasses import dataclass, field
from typing import List


@dataclass
class ListLogger:
    l: List[str] = field(default_factory=lambda: [])

    def __call__(self, log_msg):
        self.l.append(log_msg)


def test_client_with_api_key_in_env():
    logger = ListLogger()
    chatgpt_client = chatgpt_api.ChatGPTClient(None, logger=logger)
    chatgpt_client.get_chatgpt_response_from_text("What is the weather today in Paris?")
    """
    example 4 expected messages
    intro
    'calling OpenAI API'
    kwargs
    '{}'
    actual arguments
    '[{"role": "user", "content": "What is the weather today in Paris?"}]'
    response
    '{"id": "chatcmpl-6qNmEEry8ZBHKzkp99u7QABaj3Gdo", "object": "chat.completion", "created": 1677942474, "model": "gpt-3.5-turbo-0301", "usage": {"prompt_tokens": 15, "completion_tokens": 32, "total_tokens": 47}, "choices": [{"message": {"role": "assistant", "content": "\\n\\nI\'m sorry, as an AI language model, I do not have real-time information. Please check a weather website or app for the latest updates."}, "finish_reason": "stop", "index": 0}]}'
    """
    assert len(logger.l) == 4

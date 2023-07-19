import abc
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from lmserver.sampling_parameters import SamplingParameters


class GenerationRequest(BaseModel):
    prompt: str
    min_length: int = Field(default=5)
    max_new_tokens: int = Field(default=20)
    n: Optional[int] = Field(default=1)
    max_length: int = Field(default=512)
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    sampling_parameters: Optional[SamplingParameters]
    do_sample: bool = Field(default=True)
    return_full_text: bool = Field(default=False)
    truncate_prompt: bool = Field(default=False)


class ReLLMGenerationRequest(GenerationRequest):
    pattern: str
    stop_after_match: bool = True


class TokenUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

    @classmethod
    def with_total_tokens(cls, completion_tokens, prompt_tokens):
        return TokenUsage(
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=completion_tokens + prompt_tokens,
        )


class SingleGenerationResult(BaseModel):
    text: str
    usage: TokenUsage
    truncated_prompt: bool = False


GenerationResult = List[SingleGenerationResult]

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
    do_sample: bool = True
    return_full_text: bool = False


class ReLLMGenerationRequest(GenerationRequest):
    pattern: str
    stop_after_match: bool = True


class GenerationResult(BaseModel):
    texts: List[str]
    output_tokens: int
    truncated_prompt: bool = False

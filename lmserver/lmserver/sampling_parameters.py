from typing import Optional, Union, List, Dict
from pydantic import BaseModel, Field
import abc


class Descriptions:

    temperature = "Sampling temperature. A higher temperature means the model will select less common tokens leading to a larger diversity but potentially less relevant output. It is usually better to tune top_p or top_k."
    top_k = "Select the next output token among the top_k most likely ones. A higher top_k gives more diversity but a potentially less relevant output."
    top_p = "Select the next output token among the most probable ones so that their cumulative probability is larger than top_p. A higher top_p gives more diversity but a potentially less relevant output. top_p and top_k are combined, meaning that at most top_k tokens are selected. A value of 1 disables this sampling."
    logit_bias = "A map between the token indexes and the corresponding logit bias."
    presence_penalty = "A positive value penalizes tokens which already appeared in the generated text."
    frequency_penalty = "A positive value penalizes tokens which already appeared in the generated text proportionaly to their frequency."
    repetition_penalty = "Divide by repetition_penalty the logits corresponding to tokens which already appeared in the generated text."
    typical_p = "Alternative to top_p sampling."


class SamplingParameters(BaseModel):
    temperature: Optional[float] = Field(
        1,
        description=Descriptions.temperature,
    )
    top_k: Optional[int] = Field(
        100,
        ge=1,
        le=1000,
        description=Descriptions.top_k,
    )
    top_p: Optional[float] = Field(
        0.9,
        ge=0,
        le=1,
        description=Descriptions.top_p,
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        default={},
        description=Descriptions.logit_bias,
    )
    presence_penalty: Optional[float] = Field(
        default=0,
        description=Descriptions.presence_penalty,
    )
    frequency_penalty: Optional[float] = Field(
        default=0,
        description=Descriptions.frequency_penalty,
    )
    repetition_penalty: Optional[float] = Field(
        default=1,
        description=Descriptions.repetition_penalty,
    )
    typical_p: Optional[float] = Field(default=1, description=Descriptions.typical_p)

#!/usr/bin/env python3

import abc
import logging
import os
import time
from functools import wraps
from pathlib import Path as P
from typing import Dict, List, Optional, Union

import regex
import rellm
import torch
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, pipelines
from transformers.pipelines import TextGenerationPipeline

from lmserver.models import (
    GenerationRequest,
    GenerationResult,
    ReLLMGenerationRequest,
    SamplingParameters,
)
from lmserver.peft_utils import HFPeftConfig, load_peft_model

logging.basicConfig(level=logging.INFO)


def timed(func):
    """This decorator prints the execution time for the decorated function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info("{} ran in {}s".format(func.__name__, round(end - start, 2)))
        return result

    return wrapper


class LanguageModel(abc.ABC):
    @abc.abstractmethod
    def generate(self, generation_request: GenerationRequest) -> GenerationResult:
        pass


class ModelConfig(BaseModel):
    model_name_or_path: str
    device: int = Field(default=0)
    torch_dtype_name: str = Field(default="float16")
    prompt_template: Optional[str] = Field(default=None)
    load_in_8bit: bool = Field(default=False)
    peft_config: Optional[Union[HFPeftConfig]] = Field(default=None)

    @property
    def torch_dtype(self):
        if self.torch_dtype_name == "float16":
            return torch.float16
        else:
            return torch.float32

    @property
    def loadable_model_path(self):
        model_name_or_path = self.model_name_or_path
        maybe_path = P(model_name_or_path).expanduser()
        if maybe_path.exists():
            return str(maybe_path)
        else:
            return model_name_or_path

    @property
    def display_model_name(self):
        maybe_path = P(self.model_name_or_path)
        if maybe_path.exists():
            return maybe_path.name
        else:
            return str(maybe_path)


class HuggingfaceLanguageModel(BaseModel, LanguageModel):
    model: pipelines.Pipeline
    prompt_template: Optional[str] = None
    model_name: str

    @classmethod
    def load(cls, config: ModelConfig):
        logging.info(f"Loading model {config.display_model_name}")

        model = cls._load_model(config)

        logging.info(f"loaded model")
        tokenizer = AutoTokenizer.from_pretrained(config.loadable_model_path)
        return HuggingfaceLanguageModel(
            model_name=config.display_model_name,
            model=pipeline("text-generation", model=model, tokenizer=tokenizer),
            prompt_template=config.prompt_template,
        )

    @classmethod
    def _load_model(cls, config: ModelConfig):
        model = AutoModelForCausalLM.from_pretrained(
            config.loadable_model_path,
            torch_dtype=config.torch_dtype,
            load_in_8bit=config.load_in_8bit,
            device_map="auto" if config.device == 0 else None,
        )
        if config.peft_config is not None:
            model = load_peft_model(model, config.peft_config)
        return model

    def generate(self, generation_request: GenerationRequest) -> GenerationResult:
        prompt = self._get_prompt(self.prompt_template, generation_request.prompt)
        result = GenerationResult(
            texts=self._generate_hf_texts(prompt, generation_request),
            output_tokens=generation_request.max_new_tokens,
            truncated_prompt=False,
        )
        return result

    def rellm_generate(
        self, generation_request: ReLLMGenerationRequest
    ) -> GenerationResult:
        prompt = self._get_prompt(self.prompt_template, generation_request.prompt)
        result = GenerationResult(
            texts=[self._generate_rellm_text(prompt, generation_request)],
            output_tokens=generation_request.max_new_tokens,
            truncated_prompt=False,
        )
        return result

    def _generate_rellm_text(
        self, prompt, generation_request: ReLLMGenerationRequest
    ) -> str:
        pattern = regex.compile(generation_request.pattern)
        text_generated = rellm.complete_re(
            prompt,
            pattern=pattern,
            model=self.model.model,
            tokenizer=self.model.tokenizer,
            max_new_tokens=generation_request.max_new_tokens,
            do_sample=generation_request.do_sample,
        )
        return text_generated

    def _generate_hf_texts(
        self, prompt, generation_request: GenerationRequest
    ) -> List[str]:
        text_generated = self.model(
            prompt,
            max_new_tokens=generation_request.max_new_tokens,
            num_return_sequences=generation_request.n,
            do_sample=generation_request.do_sample,
            return_full_text=generation_request.return_full_text,
            generation_kwargs=dict(generation_request.sampling_parameters),
        )
        texts = [x["generated_text"] for x in text_generated]
        return texts

    def _get_prompt(self, prompt_template, prompt_text):
        if self.prompt_template is not None:
            return prompt_template.format(prompt_text)
        else:
            return prompt_text

    class Config:
        arbitrary_types_allowed = True

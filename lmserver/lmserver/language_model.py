#!/usr/bin/env python3

import logging
from typing import Optional, Union, List, Dict
from pydantic import BaseModel, Field
from transformers import pipeline, pipelines, AutoTokenizer, AutoModelForCausalLM
import abc
from lmserver.models import SamplingParameters, GenerationRequest, GenerationResult
import torch
from pathlib import Path as P
import time
from functools import wraps
from transformers.pipelines import TextGenerationPipeline

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

    @staticmethod
    def load(config: ModelConfig):
        logging.info(f"Loading model {config.display_model_name}")
        m = AutoModelForCausalLM.from_pretrained(
            config.loadable_model_path,
            torch_dtype=config.torch_dtype,
            load_in_8bit=config.load_in_8bit,
            device_map="auto" if config.device == 0 else None,
        )
        logging.info(f"loaded model")
        tokenizer = AutoTokenizer.from_pretrained(config.loadable_model_path)
        return HuggingfaceLanguageModel(
            model_name=config.display_model_name,
            model=pipeline("text-generation", model=m, tokenizer=tokenizer),
            prompt_template=config.prompt_template,
        )

    def generate(self, generation_request: GenerationRequest) -> GenerationResult:
        if self.prompt_template is not None:
            prompt = self.prompt_template.format(generation_request.prompt)
        else:
            prompt = generation_request.prompt
        text_generated = self.model(
            prompt,
            max_new_tokens=generation_request.max_new_tokens,
            num_return_sequences=generation_request.n,
            do_sample=generation_request.do_sample,
            return_full_text=generation_request.return_full_text,
            generation_kwargs=dict(generation_request.sampling_parameters),
        )
        text = [x["generated_text"] for x in text_generated]
        if generation_request.n == 1:
            text = text[0]
        result = GenerationResult(
            text=text,
            output_tokens=generation_request.max_new_tokens,
            truncated_prompt=False,
        )
        return result

    class Config:
        arbitrary_types_allowed = True

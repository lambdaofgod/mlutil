from pydantic import BaseModel
from mlutil.text import rwkv_utils
from minichain.backend import Backend
from transformers import (
    pipelines,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from typing import Union
import torch


class HuggingFaceLocalModel(BaseModel, Backend):
    model_name: str
    pipeline: Union[
        pipelines.Text2TextGenerationPipeline, pipelines.TextGenerationPipeline
    ]

    @classmethod
    def load(cls, model_name: str, device=0, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if "rwkv" in model_name:
            from transformers import RwkvForCausalLM

            model_cls = RwkvForCausalLM
        else:
            model_cls = AutoModelForCausalLM
        model = model_cls.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        pipeline = pipelines.pipeline(
            task="text-generation",
            tokenizer=tokenizer,
            model=model,
            device=device,
            **kwargs,
        )
        return HuggingFaceLocalModel(model_name=model_name, pipeline=pipeline)

    def run(self, request: str) -> str:
        return self.pipeline(request, return_full_text=False)[0]["generated_text"]

    class Config:
        arbitrary_types_allowed = True


class RWKVModel(BaseModel, Backend):
    model_name: str
    pipeline: rwkv_utils.RWKVPipelineWrapper

    @classmethod
    def load(cls, model_name: str, **kwargs):
        pipeline = rwkv_utils.RWKVPipelineWrapper.load(model_path=model_name, **kwargs)
        return RWKVModel(model_name=model_name, pipeline=pipeline)

    def run(self, request: str) -> str:
        print("request", request)
        return self.pipeline.generate(request)

    class Config:
        arbitrary_types_allowed = True

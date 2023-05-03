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


class HuggingFaceLocalModel(BaseModel, Backend):
    model_name: str
    pipeline: Union[
        pipelines.Text2TextGenerationPipeline, pipelines.TextGenerationPipeline
    ]

    @classmethod
    def load_model(cls, model_name, **kwargs):
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        except OSError:
            return AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)

        return model

    @classmethod
    def load(cls, model_name: str, device=0, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipeline = pipelines.pipeline(
            task="text-generation",
            tokenizer=tokenizer,
            model=cls.load_model(model_name, **kwargs),
            device=device,
        )
        return HuggingFaceLocalModel(model_name=model_name, pipeline=pipeline)

    def run(self, request: str) -> str:
        return self.pipeline(request)[0]["generated_text"]

    class Config:
        arbitrary_types_allowed = True


class RWKVModel(BaseModel, Backend):
    model_name: str
    pipeline: rwkv_utils.RWKVPipelineWrapper

    @classmethod
    def load(cls, model_name: str):
        pipeline = rwkv_utils.RWKVPipelineWrapper.load(
            model_path=model_name,
        )
        return RWKVModel(model_name=model_name, pipeline=pipeline)

    def run(self, request: str) -> str:
        return self.pipeline.generate(request)

    class Config:
        arbitrary_types_allowed = True

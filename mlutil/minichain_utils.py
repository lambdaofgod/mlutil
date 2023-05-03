from pydantic import BaseModel
from minichain.backend import Backend
from transformers import pipelines
from typing import Union


class HuggingFaceLocalModel(BaseModel, Backend):
    model_name: str
    pipeline: Union[
        pipelines.Text2TextGenerationPipeline, pipelines.TextGenerationPipeline
    ]

    @classmethod
    def load(cls, model_name: str, device=0):
        pipeline = pipelines.pipeline(
            task="text-generation", model=model_name, device=device
        )
        return HuggingFaceLocalModel(model_name=model_name, pipeline=pipeline)

    def run(self, request: str) -> str:
        return self.pipeline(request)[0]["generated_text"]

    class Config:
        arbitrary_types_allowed = True

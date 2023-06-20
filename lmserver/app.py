import logging
from typing import Dict, List, Optional, Union

import fire
import rellm
import uvicorn
import yaml
from fastapi import FastAPI
from pydantic import BaseModel, BaseSettings

from lmserver.language_model import HuggingfaceLanguageModel, ModelConfig
from lmserver.models import GenerationRequest, GenerationResult, ReLLMGenerationRequest

logging.basicConfig(level=logging.INFO)
app = FastAPI()


@app.post("/generate", response_model=GenerationResult)
def generate(generation_request: GenerationRequest):
    lm: HuggingfaceLanguageModel = app.state.lm
    logging.info(f"got request: {generation_request}")
    result = lm.generate(generation_request)
    logging.info(f"model returned result {result}")
    return result


@app.post("/rellm_generate", response_model=GenerationResult)
def generate(generation_request: ReLLMGenerationRequest):
    lm: HuggingfaceLanguageModel = app.state.lm
    logging.info(f"got request: {generation_request}")
    result = lm.rellm_generate(generation_request)

    logging.info(f"model returned result {result}")
    return result


@app.get("/info")
def info():
    lm: HuggingfaceLanguageModel = app.state.lm
    return {"model_name": lm.model_name}


def run_app(lm: HuggingfaceLanguageModel):
    app.state.lm = lm
    uvicorn.run(app, port=8765)


def main(config_path: str = "config.yaml"):
    with open(config_path) as config_file:
        model_config = ModelConfig(**yaml.safe_load(config_file))
    lm = HuggingfaceLanguageModel.load(config=model_config)
    run_app(lm)


if __name__ == "__main__":
    fire.Fire(main)

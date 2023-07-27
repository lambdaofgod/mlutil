import logging
from typing import Dict, List, Optional, Union

import fire
import rellm
import uvicorn
import yaml
from fastapi import FastAPI
from pydantic import BaseModel
from returns.maybe import Maybe

from lmserver.language_model import HuggingfaceLanguageModel, ModelConfig
from lmserver.models import GenerationRequest, GenerationResult, ReLLMGenerationRequest

logging.basicConfig(level=logging.INFO)
app = FastAPI()


class AppConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8765


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


def run_app(lm: HuggingfaceLanguageModel, app_config: AppConfig):
    app.state.lm = lm
    uvicorn.run(app, port=app_config.port, host=app_config.host)


def parse_config(cls, path: str):
    with open(path) as config_file:
        config_dict = yaml.safe_load(config_file)
    return cls(**config_dict)


def main(model_config_path: str = "config.yaml", app_config_path: Optional[str] = None):
    model_config = parse_config(ModelConfig, model_config_path)
    app_config = (
        Maybe.from_optional(app_config_path)
        .map(lambda p: parse_config(AppConfig, p))
        .value_or(AppConfig())
    )
    print(f"model config: {model_config}")
    print(f"model config: {app_config}")
    lm = HuggingfaceLanguageModel.load(config=model_config)
    run_app(lm, app_config)


if __name__ == "__main__":
    fire.Fire(main)

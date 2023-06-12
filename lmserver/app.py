from fastapi import FastAPI
from typing import Optional, Union, List, Dict
from pydantic import BaseSettings, BaseModel
from lmserver.models import GenerationRequest, GenerationResult
from lmserver.language_model import HuggingfaceLanguageModel, ModelConfig
import yaml
import uvicorn
import logging
import fire

logging.basicConfig(level=logging.INFO)
app = FastAPI()


@app.post("/generate", response_model=GenerationResult)
def generate(generation_request: GenerationRequest):
    lm: HuggingfaceLanguageModel = app.state.lm
    logging.info(f"got request: {generation_request}")
    result = lm.generate(generation_request)
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

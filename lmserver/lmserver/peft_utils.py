from pathlib import Path as P
from typing import Optional, Union

from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field


class LocalPeftConfig(BaseModel):
    model_path: str
    config_path: str


class HubPeftConfig(BaseModel):
    repo_id: str
    adapter_model_filename: str = Field(default="adapter_model.bin")
    adapter_config_filename: str = Field(default="adapter_config.json")

    def get_downloaded_peft_config(self) -> LocalPeftConfig:
        """
        downloads peft adapter from huggingface and returns config that can be used to load it
        """
        local_model_path = hf_hub_download(self.repo_id, self.adapter_model_filename)
        local_config_path = hf_hub_download(self.repo_id, self.adapter_config_filename)
        return LocalPeftConfig(
            model_path=local_model_path, config_path=local_config_path
        )


def load_peft_model(base_model, peft_config: Union[LocalPeftConfig, HubPeftConfig]):
    from peft import PeftModel

    if isinstance(peft_config, HubPeftConfig):
        local_peft_config = peft_config.get_downloaded_peft_config()
    else:
        local_peft_config = peft_config

    return PeftModel.from_pretrained(base_model, local_peft_config.model_path)

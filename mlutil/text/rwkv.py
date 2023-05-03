from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

from typing import List
from pydantic import BaseModel


class RWKVGenerationArgs(BaseModel):
    """
    pydantic rewrite of rwkv.util.PIPELINE_ARGS
    """

    temperature: float = Field(default=1.0)
    top_p: float = Field(default=0.85)
    top_k: float = Field(default=0)
    alpha_frequency: float = Field(default=0.2)
    alpha_presence: float = Field(default=0.2)
    token_ban: List[str] = Field(default=[])
    token_stop: List[str] = Field(default=[])


class RWKVHuggingfaceWrapper:


class RWKVStrategy(BaseModel):
    """
    TODO add validation
    """

    rwkv_strategy: str

    @staticmethod
    def from_device_and_dtype(device, dtype):
        return RWKVStrategy(rwkv_strategy=device + " " + dtype)

    def maybe_override_with_string(self, rwkv_strategy: Optional[str]):
        return RWKVStrategy(rwkv_strategy) if rwkv_strategy else self


class RWKVPipelineWrapper(BaseModel):

    pipeline: PIPELINE

    def load(
        self,
        model_path,
        tokenizer_path=None,
        device="cuda",
        dtype="fp16",
        rwkv_strategy_str=None,
    ):
        if tokenizer_path is None:
            tokenizer_path = Path(f"{model_path.parent}/20B_tokenizer.json")
        rwkv_strategy = RWKVStrategy.from_device_and_dtype(
            device, dtype
        ).override_with_string(rwkv_strategy_str)
        model = RWKV(model=str(model_path), strategy=rwkv_strategy.rwv_strategy)
        return RWKVPipelineWrapper(PIPELINE(model, tokenizer_path))

    def generate(
        self,
        context,
        token_count=100,
        args: RWKVGenerationArgs = RWKVGenerationArgs(),
        state=None,
        callback=None,
    ):
        return self.pipeline(
            context, token_count=token_count, args=args, callback=callback, state=state
        )

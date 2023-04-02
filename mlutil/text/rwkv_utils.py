from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Iterable, Union
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

from typing import List
from pydantic import BaseModel


class RWKVGenerationArgs(BaseModel):
    """
    pydantic rewrite of rwkv.util.PIPELINE_ARGS
    """

    temperature: float = Field(default=1.0)
    top_p: float = Field(default=1.0)
    top_k: float = Field(default=50)
    alpha_frequency: float = Field(default=0.1)
    alpha_presence: float = Field(default=0.1)
    token_ban: List[str] = Field(default_factory=lambda: [0])
    token_stop: List[str] = Field(default_factory=list)


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
    model_name_or_path: Optional[str]

    @staticmethod
    def load(
        model_path,
        tokenizer_path=None,
        device="cuda",
        dtype="fp16",
        rwkv_strategy_str=None,
    ):
        model_path = Path(model_path)
        if tokenizer_path is None:
            tokenizer_path = Path(f"{model_path.parent}/20B_tokenizer.json")
        rwkv_strategy = RWKVStrategy.from_device_and_dtype(
            device, dtype
        ).maybe_override_with_string(rwkv_strategy_str)
        model = RWKV(model=str(model_path), strategy=rwkv_strategy.rwkv_strategy)
        pipeline = PIPELINE(model, str(tokenizer_path))
        return RWKVPipelineWrapper(
            pipeline=pipeline, model_name_or_path=str(model_path)
        )

    def generate(
        self,
        context,
        token_count=100,
        args: RWKVGenerationArgs = RWKVGenerationArgs(),
        state=None,
        callback=None,
    ):
        return self.pipeline.generate(
            context, token_count=token_count, args=args, callback=callback, state=state
        )

    class Config:

        arbitrary_types_allowed = True


class RWKVPromptifyModel(BaseModel):

    pipeline: RWKVPipelineWrapper
    alpha_frequency: float = Field(default=0.2)
    alpha_presence: float = Field(default=0.2)
    top_k: float = Field(default=0)

    def run(
        self,
        prompts: List[str],
        suffix: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 0.85,
        stop: Union[str, Iterable[str], None] = None,
    ):
        assert len(prompts) == 1
        prompt = prompts[0]
        args = RWKVGenerationArgs(
            temperature=temperature,
            top_p=top_p,
            token_stop=stop if stop else [],
            else_k=self.top_k,
            alpha_presence=self.alpha_presence,
            alpha_frequency=self.alpha_frequency,
        )
        return self.pipeline.generate(context=prompt, token_count=max_tokens, args=args)

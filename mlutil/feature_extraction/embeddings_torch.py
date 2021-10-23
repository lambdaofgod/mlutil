import torch
import pandas as pd
from fastai.text import all as fastai_text

import torch
import attr
import numpy as np
import tqdm


class BaseTorchVectorizer:
    def transform_batch(self, batch):
        raise NotImplementedError()

    @classmethod
    def get_batched_list(cls, lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i : i + batch_size]

    def transform(self, texts, batch_size=None, verbose=True):
        batch_size = batch_size or self.batch_size
        batches = self.get_batched_list(texts, batch_size)
        if verbose:
            batches = tqdm.tqdm(batches, total=int(np.ceil(len(texts) / batch_size)))
        features = []
        with torch.no_grad():
            for batch in batches:
                features_tensor = self.transform_batch(batch).detach()
                features.append(features_tensor.cpu().numpy())
        return np.row_stack(features)


class TransformerVectorizer(BaseTorchVectorizer):
    def __init__(self, model_type, aggregation="mean", device=0, batch_size=128):
        self.batch_size = batch_size
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_type)
        self.model = transformers.AutoModel.from_pretrained(model_type)
        if device != -1:
            self.model = self.model.cuda(device=0)
        self.aggregating_function = self.get_batch_aggregating_function(aggregation)

    def transform_batch(self, batch):
        tokenizer_output = self.tokenizer(batch, return_tensors="pt", padding=True)
        if self.model.device.type != "cpu":
            tokenizer_output = {
                k: v.cuda(self.model.device) for (k, v) in tokenizer_output.items()
            }
        batch_model_output = self.model(**tokenizer_output).last_hidden_state
        return self.aggregating_function(
            batch_model_output, tokenizer_output["attention_mask"]
        )

    @classmethod
    def get_batch_aggregating_function(cls, aggregation):
        def mean_aggregating_function(batch, attention_mask):
            return (batch * attention_mask.unsqueeze(2)).sum(
                axis=1
            ) / attention_mask.sum(axis=1).unsqueeze(1)

        def max_aggregating_function(batch, attention_mask):
            return (batch * attention_mask.unsqueeze(2)).max(axis=1).values

        def min_aggregating_function(batch, attention_mask):
            return (batch * attention_mask.unsqueeze(2)).min(axis=1).values

        def concatpool_aggreggating_function(batch, attention_mask):
            pooled_outputs = [
                aggregating_function(batch, attention_mask)
                for aggregating_function in [
                    mean_aggregating_function,
                    max_aggregating_function,
                    min_aggregating_function,
                ]
            ]
            return torch.hstack(pooled_outputs)

        if aggregation == "mean":
            return mean_aggregating_function
        elif aggregation == "max":
            return min_aggregating_function
        else:
            return concatpool_aggreggating_function

    @classmethod
    def get_batched_list(cls, lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i : i + batch_size]


class FastAIModelEmbedder(BaseTorchVectorizer):
    def __init__(self, learner, batch_size=64):
        self.learner = learner
        self.device = list(p.device for p in learner.model.parameters())[0]
        self.batch_size = 64

    def numericalize_batch(self, texts, max_len):
        dls = self.learner.dls
        numericalized_texts = [
            dls.numericalize.encodes(dls.tokenizer.encodes(text)) for text in texts
        ]
        lenghts = [len(nt) for nt in numericalized_texts]
        padded_tuples = fastai_text.pad_input(
            [(nt[:max_len], max_len - len(nt)) for nt in numericalized_texts], pad_idx=0
        )
        indices = torch.stack([pt[0] for pt in padded_tuples])
        mask = torch.zeros_like(indices)
        for i, l in zip(range(len(texts)), lenghts):
            mask[i, :l] = 1
        return indices, mask

    def transform_batch(self, texts, max_len=100):
        idxs, mask = self.numericalize_batch(texts, max_len)

        idxs = idxs.to(self.device)
        __, out, __ = self.learner.model(idxs)
        embeddings = out * mask.to(self.device)[:, :, None]
        return embeddings.mean(axis=1)

    @classmethod
    def get_batched_list(cls, lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i : i + batch_size]

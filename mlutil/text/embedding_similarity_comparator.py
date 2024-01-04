from typing import List
import torch
from functools import singledispatch
from pydantic import BaseModel
import numpy as np
import sentence_transformers
import pandas as pd
from sklearn import metrics
import seaborn as sns

try:
    from flair.embeddings import FastTextEmbeddings, DocumentPoolEmbeddings, FlairEmbeddings
    from flair.data import Sentence

except ImportError:
    DocumentPoolEmbeddings = "DocumentPoolEmbeddings"
    Sentence = "Sentence"


class EmbeddedTexts(BaseModel):
    embeddings: np.ndarray
    texts: List[str]

    class Config:
        arbitrary_types_allowed = True


@singledispatch
def get_embedded_texts(embedder, docs):
    pass


@get_embedded_texts.register(DocumentPoolEmbeddings)
def get_embedded_texts_flair(embedder: DocumentPoolEmbeddings, docs: List[str]) -> EmbeddedTexts:
    sentences = [Sentence(doc) for doc in docs]
    embedder.embed(sentences)
    return EmbeddedTexts(embeddings=np.vstack([sent.embedding.cpu().numpy() for sent in sentences]), texts=docs)


@get_embedded_texts.register(sentence_transformers.SentenceTransformer)
def get_embedded_texts_st(embedder: sentence_transformers.SentenceTransformer, docs: List[str]) -> EmbeddedTexts:
    return EmbeddedTexts(embeddings=embedder.encode(docs), texts=docs)


def plot_embedding_similarities(text_embeddings: EmbeddedTexts, ax=None):
    dists = metrics.pairwise.cosine_similarity(text_embeddings.embeddings)

    dists_df = pd.DataFrame(
        dists, index=text_embeddings.texts, columns=text_embeddings.texts)
    sns.heatmap(dists_df, annot=True, ax=ax)

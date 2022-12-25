import itertools
import logging
import os
import warnings

import attr
import gensim.downloader as gensim_data_downloader
import numpy as np
import torch
import tqdm
import transformers
from sklearn import decomposition
from sklearn.feature_extraction.text import _VectorizerMixin
from abc import ABC


def load_gensim_embedding_model(model_name):
    """
    Load word embeddings (gensim KeyedVectors)
    """
    available_models = gensim_data_downloader.info()["models"].keys()
    assert (
        model_name in available_models
    ), "Invalid model_name: {}. Choose one from {}".format(
        model_name, ", ".join(available_models)
    )

    # gensim throws some nasty warnings about vocabulary
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=UserWarning, module="gensim")
        model = gensim_data_downloader.load(model_name)
    return model


def get_tfhub_encoder(url):
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError as e:
        logging.warning(
            "tensorflow or tensorflow-hub not found, loading tfhub models won't work"
        )

    tfhub_module = hub.Module(url)

    def encode(texts, session_callback):
        with session_callback() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            message_embeddings = session.run(tfhub_module(texts))
        return message_embeddings

    return encode


class FastTextVectorizer:
    def __init__(self, fasttext_model, lowercase=True):
        self.model = fasttext_model
        self.lowercase = lowercase

    def transform(self, X, **kwargs):
        if self.lowercase:
            X = [text.lower() for text in X]
        return np.array([self.model.get_sentence_vector(text) for text in X])

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X)
        return self.transform(X, **kwargs)


class EmbeddingVectorizer(_VectorizerMixin):
    """
    Base class for word/text embedding wrappers
    """

    def __init__(
        self,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w+\b",
        analyzer="word",
    ):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.ngram_range = (1, 1)
        self.analyzer = analyzer
        """I admit this is pretty silly but unfortunately build analyzer requires this to be set..."""
        self.analyzer = self.build_analyzer()

    def transform(self, X, **kwargs):
        analyzed_docs = [" ".join(self.analyzer(doc)) for doc in X]
        return self._embed_texts(analyzed_docs, **kwargs)

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X)
        return self.transform(X, **kwargs)

    def _embed_texts(self, texts, **kwargs):
        """implementation of embedding"""
        raise NotImplementedError

    @classmethod
    def is_in_vocab(self, elem, keyed_vectors):
        return elem in keyed_vectors.key_to_index.keys()


class Doc2Vectorizer(EmbeddingVectorizer):
    def __init__(
        self,
        doc2vec_model,
        input="content",
        encoding="utf-8",
        decode_error="strict",
        strip_accents=None,
        lowercase=False,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w+\b",
        analyzer="word",
    ):
        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.ngram_range = (1, 1)
        self.doc2vec_model = doc2vec_model
        self.dimensionality_ = doc2vec_model.trainables.layer1_size

    def _embed_texts(self, texts, **kwargs):
        return np.row_stack([self.doc2vec_model.infer_vector(t.split()) for t in texts])


class AverageWordEmbeddingsVectorizer(EmbeddingVectorizer):
    """
    Wrapper for gensim KeyedVectors
    """

    def __init__(self, word_embeddings, average_embeddings=True, **kwargs):
        self.word_embeddings = word_embeddings
        self.average_embeddings = average_embeddings
        self.dimensionality_ = _get_dimensionality(word_embeddings)
        super(AverageWordEmbeddingsVectorizer, self).__init__(**kwargs)

    def _embed_text(self, text):
        embeddings = [
            self.word_embeddings[w]
            for w in text.split()
            if self.is_in_vocab(w, self.word_embeddings)
        ]
        if len(embeddings) > 0:
            if self.average_embeddings:
                return np.mean(embeddings, axis=0)
            else:
                return np.vstack(embeddings)
        else:
            return np.zeros((self.dimensionality_,))

    def _embed_texts(self, texts, **kwargs):
        embeddings = [self._embed_text(text) for text in texts]
        if self.average_embeddings:
            return np.vstack(embeddings)
        else:
            return embeddings

    @classmethod
    def from_gensim_embedding_model(cls, model_name="glove-wiki-gigaword-50", **kwargs):
        word_embeddings = load_gensim_embedding_model(model_name)
        return AverageWordEmbeddingsVectorizer(word_embeddings, **kwargs)


class WeightedAverageWordEmbeddingsVectorizer(EmbeddingVectorizer):
    """
    Wrapper for gensim KeyedVectors
    """

    def __init__(self, word_embeddings, weights, average_embeddings=True, **kwargs):
        self.word_embeddings = word_embeddings
        self.average_embeddings = average_embeddings
        self.weights = weights
        self.dimensionality_ = _get_dimensionality(word_embeddings)
        super(WeightedAverageWordEmbeddingsVectorizer, self).__init__(**kwargs)

    def _embed_text(self, text):
        words = [
            w
            for w in text.split()
            if self.is_in_vocab(w, self.word_embeddings) and w in self.weights.index
        ]
        embeddings = [self.word_embeddings[w] for w in words]
        if len(embeddings) > 0:
            if self.average_embeddings:
                return np.mean(embeddings, axis=0)
            else:
                return np.vstack(embeddings)
        else:
            return np.zeros((self.dimensionality_,))

    def _embed_texts(self, texts, **kwargs):
        embeddings = [self._embed_text(text) for text in texts]
        if self.average_embeddings:
            return np.vstack(embeddings)
        else:
            return embeddings

    @classmethod
    def from_gensim_embedding_model(cls, model_name="glove-wiki-gigaword-50", **kwargs):
        word_embeddings = load_gensim_embedding_model(model_name)
        return AverageWordEmbeddingsVectorizer(word_embeddings, **kwargs)


class PCREmbeddingVectorizer(EmbeddingVectorizer):
    """
    sentence embedding with principal component removal
    the idea comes from 'A Simple but Tough-to-Beat Baseline for Sentence Embeddings'
    also see 'A Critique of the Smooth Inverse Frequency Sentence Embeddings'
    """

    def __init__(self, word_embeddings, component_analyzer=None, **kwargs):
        self.word_embeddings = word_embeddings
        self.average_embeddings = (True,)
        self.dimensionality_ = _get_dimensionality(word_embeddings)
        self.component_analyzer = (
            component_analyzer
            if component_analyzer is not None
            else decomposition.TruncatedSVD(n_components=1)
        )
        super(EmbeddingVectorizer, self).__init__(**kwargs)

    def fit(self, texts, **kwargs):
        vectors = self._embed_texts(texts)
        self.component_analyzer.fit(vectors)
        self.components_ = self.component_analyzer.components_

    def transform(self, texts, **kwargs):
        vectors = self._embed_texts(texts, **kwargs)
        deleted_components = self.component_analyzer.transform(vectors)
        return vectors - self.component_analyzer.inverse_transform(deleted_components)

    def _embed_text(self, text):
        embeddings = [
            self.word_embeddings[w]
            for w in text.split()
            if self.is_in_vocab(w, self.word_embeddings)
        ]
        if len(embeddings) > 0:
            if self.average_embeddings:
                return np.mean(embeddings, axis=0)
            else:
                return np.vstack(embeddings)
        else:
            return np.zeros((self.dimensionality_,))

    def fit_transform(self, texts, **kwargs):
        self.fit(texts)
        return self.transform(texts)

    def _embed_texts(self, texts, **kwargs):
        return np.vstack([self._embed_text(t, **kwargs) for t in texts])

    @classmethod
    def _get_vectors_or_default(cls, word_embeddings, words, default=None):
        if default is None:
            default = np.zeros(word_embeddings["."].shape)

        if len(words) > 0:
            return np.vstack([word_embeddings[w] for w in words])
        else:
            return default


def _get_dimensionality(word_embeddings):
    example_key = list(itertools.islice(word_embeddings.key_to_index.keys(), 1))[0]
    vector = word_embeddings[example_key]
    return vector.shape[0]


try:
    import sentence_transformers

    @attr.s
    class SentenceTransformerWrapper:

        model: sentence_transformers.SentenceTransformer = attr.ib()

        def fit(self, *args, **kwargs):
            pass

        def transform(self, X, **kwargs):
            return self.model.encode(list(X), **kwargs)

except ImportError:
    logging.warning("sentence_transformers not found, cannot import SBERTModelWrapper")

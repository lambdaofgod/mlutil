import warnings

import numpy as np
from sklearn.feature_extraction.text import VectorizerMixin
import gensim.downloader as gensim_data_downloader
import logging
import itertools


try:
    import tensorflow as tf
    import tensorflow_hub as hub
except ImportError as e:
    logging.warning("tensorflow or tensorflow-hub not found, loading tfhub models won't work")


def load_gensim_embedding_model(model_name):
    """
    Load word embeddings (gensim KeyedVectors) 
    """
    available_models = gensim_data_downloader.info()['models'].keys()
    assert model_name in available_models, 'Invalid model_name: {}. Choose one from {}'.format(model_name, ', '.join(available_models))
    
    # gensim throws some nasty warnings about vocabulary
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
        model = gensim_data_downloader.load(model_name)
    return model


def get_tfhub_encoder(url):

    tfhub_module = hub.Module(url)

    def encode(texts):
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            message_embeddings = session.run(tfhub_module(texts))
        return message_embeddings
    return encode


class EmbeddingVectorizer(VectorizerMixin):
    """
        Base class for word/text embedding wrappers
    """

    def fit_transform(self, X, **kwargs):
        analyzer = self.build_analyzer()
        analyzed_docs = [' '.join(analyzer(doc)) for doc in X]
        return self._embed_texts(analyzed_docs, **kwargs)

    def fit(self, X):
        return self

    def transform(self, X, **kwargs):
        return self.fit_transform(X, **kwargs)

    def _embed_texts(self, texts):
        raise NotImplementedError()


class TextEncoderVectorizer(EmbeddingVectorizer):
    """
        Wrapper for Tensorflow Hub Universal Sentence Encoder
    """

    def __init__(self, text_encoder, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=False, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w+\b",
                 analyzer='word'):
        self.text_encoder = text_encoder
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
        self.ngram_range = (1,1)

    def _embed_texts(self, texts):
        return self.text_encoder(texts)

    @staticmethod
    def from_tfhub_encoder(tfhub_encoder='small', **kwargs):
        if type(tfhub_encoder) is str:
            if tfhub_encoder == 'small':
                url = 'https://tfhub.dev/google/universal-sentence-encoder/2'
                encoder = get_tfhub_encoder(url)
            if tfhub_encoder == 'big':
                url = 'https://tfhub.dev/google/universal-sentence-encoder-large/3'
                encoder = get_tfhub_encoder(url)
        elif type(tfhub_encoder) is hub.Module:
            encoder = tfhub_encoder
        else:
            raise ValueError('Invalid TFHub encoder')
        return TextEncoderVectorizer(encoder, **kwargs)


class WordEmbeddingsVectorizer(EmbeddingVectorizer):
    """
        Wrapper for gensim KeyedVectors
    """

    def __init__(self, word_embeddings, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w+\b",
                 analyzer='word'):
        self.word_embeddings = word_embeddings
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
        self.ngram_range = (1,1)
        self.dimensionality = self._get_dimensionality(word_embeddings)

    def _embed_text(self, text, aggregate):
        embeddings = [self.word_embeddings[w] for w in text.split() if self.word_embeddings.vocab.get(w) is not None]
        if len(embeddings) > 0:
            if aggregate:
                return np.mean(embeddings, axis=0)
            else:
                return np.vstack(embeddings)
        else:
            return np.zeros((self.dimensionality,))

    def _embed_texts(self, texts, aggregate=True):
        embeddings = [self._embed_text(text, aggregate=aggregate) for text in texts]
        if aggregate:
            return np.vstack(embeddings)
        else:
            return embeddings

    @classmethod
    def _get_dimensionality(cls, word_embeddings):
        example_key = list(itertools.islice(word_embeddings.vocab, 1))[0]
        vector = word_embeddings[example_key]
        return vector.shape[0]

    @classmethod
    def from_gensim_embedding_model(cls, model_name='glove-wiki-gigaword-50', **kwargs):
        word_embeddings = load_gensim_embedding_model(model_name)
        return WordEmbeddingsVectorizer(word_embeddings, **kwargs)

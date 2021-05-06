import itertools
import logging
import os
import warnings
import transformers
import numpy as np
import tqdm

import torch


import gensim.downloader as gensim_data_downloader
import numpy as np
from sklearn import decomposition
from mlutil.feature_extraction.text import VectorizerMixin


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

    def encode(texts, session_callback):
        with session_callback() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            message_embeddings = session.run(tfhub_module(texts))
        return message_embeddings
    return encode


class TransformerVectorizer:

    def __init__(self, model_type, aggregation='mean', device=0, batch_size=128):
        self.batch_size = batch_size
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_type)
        self.model = transformers.AutoModel.from_pretrained(model_type)
        if device != -1:
            self.model = self.model.cuda(device=0)
        self.aggregating_function = self.get_batch_aggregating_function(aggregation)

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

    def transform_batch(self, batch):
        tokenizer_output = self.tokenizer(batch, return_tensors='pt', padding=True)
        if self.model.device.type != 'cpu':
            tokenizer_output = {
                k : v.cuda(self.model.device)
                for (k, v) in tokenizer_output.items()
            }
        batch_model_output = self.model(**tokenizer_output).last_hidden_state
        return self.aggregating_function(batch_model_output, tokenizer_output['attention_mask'])

    @classmethod
    def get_batch_aggregating_function(cls, aggregation):
        def mean_aggregating_function(batch, attention_mask):
            return (batch * attention_mask.unsqueeze(2)).sum(axis=1) / attention_mask.sum(axis=1).unsqueeze(1)
        def max_aggregating_function(batch, attention_mask):
            return (batch * attention_mask.unsqueeze(2)).max(axis=1).values
        def min_aggregating_function(batch, attention_mask):
            return (batch * attention_mask.unsqueeze(2)).min(axis=1).values
        def concatpool_aggreggating_function(batch, attention_mask):
            pooled_outputs = [
                aggregating_function(batch, attention_mask)
                for aggregating_function in [mean_aggregating_function, max_aggregating_function, min_aggregating_function]
            ]
            return torch.hstack(pooled_outputs)
        if aggregation == 'mean':
            return mean_aggregating_function
        elif aggregation == 'max':
            return min_aggregating_function
        else:
            return concatpool_aggreggating_function

    @classmethod
    def get_batched_list(cls, lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i+batch_size]


class EmbeddingVectorizer(VectorizerMixin):
    """
        Base class for word/text embedding wrappers
    """

    def transform(self, X, **kwargs):
        analyzer = self.build_analyzer()
        analyzed_docs = [' '.join(analyzer(doc)) for doc in X]
        return self._embed_texts(analyzed_docs, **kwargs)

    def fit(self, X):
        return self

    def fit_transform(self, X, **kwargs):
        self.fit(X)
        return self.transform(X, **kwargs)

    def _embed_texts(self, texts, **kwargs):
        raise NotImplementedError()


class Doc2Vectorizer(EmbeddingVectorizer):

    def __init__(self, doc2vec_model, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=False, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w+\b",
                 analyzer='word'):
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
        self.doc2vec_model = doc2vec_model
        self.dimensionality_ = doc2vec_model.trainables.layer1_size

    def _embed_texts(self, texts, **kwargs):
        return np.row_stack(
            [
                self.doc2vec_model.infer_vector(t.split())
                for t in texts
            ]
        )


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

    def _embed_texts(self, texts, batch_size=256, **kwargs):
        texts_chunked = TextEncoderVectorizer.iter_chunks(texts, chunk_size=batch_size)
        return np.vstack([self.text_encoder(text_chunk) for text_chunk in texts_chunked])

    @staticmethod
    def from_tfhub_encoder(tfhub_encoder='large', **kwargs):
        if type(tfhub_encoder) is str:
            if os.path.exists(tfhub_encoder):
                encoder = get_tfhub_encoder(tfhub_encoder)
            elif tfhub_encoder == 'small':
                url = 'https://tfhub.dev/google/universal-sentence-encoder/2'
                encoder = get_tfhub_encoder(url)
            elif tfhub_encoder == 'large':
                url = 'https://tfhub.dev/google/universal-sentence-encoder-large/3'
                encoder = get_tfhub_encoder(url)
            else:
                raise ValueError('Invalid TFHub encoder name or URL')
        elif type(tfhub_encoder) is hub.Module:
            encoder = tfhub_encoder
        else:
            raise ValueError('Invalid TFHub encoder')
        return TextEncoderVectorizer(encoder, **kwargs)

    @staticmethod
    def iter_chunks(sequence, chunk_size):
        res = []
        for item in sequence:
            res.append(item)
            if len(res) >= chunk_size:
                yield res
                res = []
        if res:
            yield res



class AverageWordEmbeddingsVectorizer(EmbeddingVectorizer):
    """
        Wrapper for gensim KeyedVectors
    """

    def __init__(self, word_embeddings, average_embeddings=True, input='content', encoding='utf-8',
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
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.ngram_range = (1,1)
        self.dimensionality_ = _get_dimensionality(word_embeddings)
        self.average_embeddings = average_embeddings
        self.analyzer = analyzer

    def _embed_text(self, text):
        embeddings = [self.word_embeddings[w] for w in text.split() if self.word_embeddings.wv.vocab.get(w) is not None]
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
    def from_gensim_embedding_model(cls, model_name='glove-wiki-gigaword-50', **kwargs):
        word_embeddings = load_gensim_embedding_model(model_name)
        return AverageWordEmbeddingsVectorizer(word_embeddings, **kwargs)


class PCREmbeddingVectorizer(EmbeddingVectorizer):
    """
        sentence embedding with principal component removal
        the idea comes from 'A Simple but Tough-to-Beat Baseline for Sentence Embeddings'
        also see 'A Critique of the Smooth Inverse Frequency Sentence Embeddings' 
    """
    def __init__(
            self,
            word_embeddings,
            component_analyzer=None,
            average_embeddings=True,
            input='content', encoding='utf-8',
            decode_error='strict', strip_accents=None,
            lowercase=True, preprocessor=None, tokenizer=None,
            stop_words=None, token_pattern=r"(?u)\b\w+\b",
            analyzer='word'):
        self.component_analyzer = component_analyzer if not component_analyzer is None else decomposition.TruncatedSVD(n_components=1)
        self.word_embeddings = word_embeddings
        self.average_embeddings=True,
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
        self.dimensionality_ = _get_dimensionality(word_embeddings)
        self.analyzer = analyzer

    def fit(self, texts):
        vectors = self._embed_texts(texts)
        self.component_analyzer.fit(vectors)
        self.components_ = self.component_analyzer.components_

    def transform(self, texts, **kwargs):
        vectors = self._embed_texts(texts, **kwargs)
        deleted_components = self.component_analyzer.transform(vectors)
        return vectors - self.component_analyzer.inverse_transform(deleted_components)

    def _embed_text(self, text):
        embeddings = [self.word_embeddings[w] for w in text.split() if self.word_embeddings.vocab.get(w) is not None]
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
            default = np.zeros(word_embeddings['.'].shape)

        if len(words) > 0:
            return np.vstack([word_embeddings[w] for w in words])
        else:
            return default


class SIFEmbeddingVectorizer(PCREmbeddingVectorizer):
    """
        sentence embedding by Smooth Inverse Frequency weighting scheme from 'A Simple but Tough-to-Beat Baseline for Sentence Embeddings'
    """

    def __init__(self, count_vectorizer, a=0.01, **kwargs):
        PCREmbeddingVectorizer.__init__(self, **kwargs)
        self.count_vectorizer = count_vectorizer
        self.a = a

    def fit(self, a=None):
        if not hasattr(self.count_vectorizer, 'vocabulary_'):
            self.count_vectorizer.fit(texts)
        vectors = self._embed_texts(texts, a=a)
        self.component_analyzer.fit(vectors)

    def _embed_text(self, text):
        words = self.analyzer(text)
        filtered_words = [word for word in words if word in self.word_embeddings.wv.vocab.keys()]
        word_vectors = self._get_vectors_or_default(self.word_embeddings, filtered_words)
        smoothed_word_frequencies = self._get_smoothed_inverse_word_frequencies(filtered_words, self.count_vectorizer)
        return (word_vectors * smoothed_word_frequencies).sum(axis=0)

    @classmethod
    def _get_smoothed_inverse_word_frequencies(cls, words, count_vectorizer, eps=1e-8):
        word_counts = count_vectorizer.transform(words).sum(axis=1)
        word_probabilities = 1.0 / (np.asarray(word_counts) + eps)
        return self.a / (self.a + word_probabilities)


def _get_dimensionality(word_embeddings):
    example_key = list(itertools.islice(word_embeddings.wv.vocab, 1))[0]
    vector = word_embeddings[example_key]
    return vector.shape[0]

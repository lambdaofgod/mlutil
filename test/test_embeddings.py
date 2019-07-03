import pytest
import numpy as np
import logging
from mlutil.embeddings import load_gensim_embedding_model, TextEncoderVectorizer, WordEmbeddingsVectorizer
try:
    import tensorflow as tf
    import tensorflow_hub as hub
except ImportError as e:
    logging.warning("tensorflow or tensorflow-hub not found, skipping tensorflow tests")


word = "Elephant"
sentence = "I am a sentence for which I would like to get its embedding."
paragraph = (
    "Universal Sentence Encoder embeddings also support short paragraphs. "
    "There is no hard limit on how long the paragraph is. Roughly, the longer "
    "the more 'diluted' the embedding will be.")
texts = [word, sentence, paragraph]


@pytest.mark.slow
def test_tfhub_embedding_vectorizer():

    vectorizer = TextEncoderVectorizer.from_tfhub_encoder('large')

    text_vectors = vectorizer.transform(texts)

    assert text_vectors.shape == (3, 512)
    assert np.all(np.isclose(text_vectors[:, 0], np.array([0.04498473, 0.05568017, 0.03874939])))


def test_word_embeddings_vectorizer_aggregation():

    keyed_vectors = load_gensim_embedding_model('glove-wiki-gigaword-50')
    vectorizer = WordEmbeddingsVectorizer(keyed_vectors, average_embeddings=True)

    text_vectors = vectorizer.transform(texts)
    assert text_vectors.shape == (3, 50)


def test_word_embeddings_vectorizer_without_aggregation():

    keyed_vectors = load_gensim_embedding_model('glove-wiki-gigaword-50')
    vectorizer = WordEmbeddingsVectorizer(keyed_vectors, average_embeddings=False)

    text_vectors = vectorizer.transform(texts)
    assert text_vectors[0].shape == (1, 50)
    assert text_vectors[1].shape == (13, 50)

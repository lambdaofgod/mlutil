from itertools import combinations
from typing import List, Optional

import nltk
import numpy as np
import pandas as pd
import tqdm
from sklearn import decomposition, feature_extraction, pipeline

from .textmining import LemmaTokenizer, StemTokenizer, get_wordnet_similarity


def create_topic_modeling_pipeline(
    model_cls=decomposition.NMF,
    vectorizer_cls=feature_extraction.text.TfidfVectorizer,
    n_topics=10,
    stop_words="english",
    tokenizer_cls=LemmaTokenizer,
    model_kwargs={},
    token_pattern=r"\b[a-zA-Z]+\S+\b",
    vectorizer_kwargs={},
):
    """
    makes pipeline that can be used for topic modeling with reasonable defaults
    default pipeline consists of TfidfVectorizer and NMF
    """
    vectorizer = vectorizer_cls(
        stop_words=stop_words,
        tokenizer=tokenizer_cls(token_pattern),
        **vectorizer_kwargs
    )
    model = model_cls(n_topics, **model_kwargs)
    return pipeline.make_pipeline(vectorizer, model)


def top_topic_words(model, n_words: int = 5, feature_names: Optional[List[str]] = None):
    """
    model: sklearn transformer or pipeline
    feature_names: List[str] or None, if None we try to infer these from pipeline
    Show `n_words` top words from topic model
    """
    if feature_names is None:
        assert isinstance(model, pipeline.Pipeline)
        vocab = model.steps[0][1].vocabulary_
        feature_names = list(sorted(vocab, key=vocab.get))

    if isinstance(model, pipeline.Pipeline):
        model = model.steps[-1][1]

    def top_words(topic):
        return [feature_names[i] for i in topic.argsort()[: -n_words - 1 : -1]]

    topic_words_dict = {
        "topic_" + str(topic_idx): top_words(topic)
        for topic_idx, topic in enumerate(model.components_)
    }

    return pd.DataFrame(topic_words_dict).T


def get_dominant_pos(word):
    try:
        synsets = nltk.corpus.wordnet.synsets(word)
    except:
        return None
    if len(synsets) == 0:
        return None
    else:
        return synsets[0].pos()


def aggregated_similarity_measure(items, max_items, similarity, aggregation):
    valid_items = [item for item in items if get_dominant_pos(item) in ["v", "n"]][
        :max_items
    ]
    similarities = [
        similarity(x, y)
        for (x, y) in combinations(valid_items, 2)
        if get_dominant_pos(x) == get_dominant_pos(y)
    ]
    return aggregation(np.array(similarities))


def topic_coherence(
    topic_keywords,
    n_top_keywords,
    aggregation=np.mean,
    similarity=get_wordnet_similarity,
):
    return aggregated_similarity_measure(
        topic_keywords, n_top_keywords, similarity=similarity, aggregation=aggregation
    )


def get_topic_coherences(keywords_per_topic, n_used_top_keywords=10, verbose=True):
    """
    Calculate topic coherences for `keywords_per_topic`
    `keywords_per_topic` is expected to be of format returned from top_topic_words
    """
    _iter = keywords_per_topic.iterrows()
    n_topics = keywords_per_topic.shape[0]
    if verbose:
        _iter = tqdm.tqdm(_iter, total=n_topics)
    return pd.Series(
        [
            topic_coherence(keywords.values, n_top_keywords=n_used_top_keywords)
            for (__, keywords) in _iter
        ]
    )

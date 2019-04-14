import numpy as np
import pandas as pd
import nltk
from itertools import combinations
from .textmining import get_wordnet_similarity
import tqdm


def top_topic_words(model, feature_names, n_words=5):
    """
    Show n_words top words from topic model
    Works for models coming from sklearn.decomposition (for example NMF and LDA)
    """

    def top_words(topic):
        return [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]

    topic_words_dict = {'topic_' + str(topic_idx): top_words(topic) for topic_idx, topic in enumerate(model.components_)}

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


def aggregated_similarity_measure(items, similarity, aggregation='mean'):
    similarities = [similarity(x, y) for (x, y) in combinations(items, 2) if get_dominant_pos(x) == get_dominant_pos(y) and get_dominant_pos(x) is not None and get_dominant_pos(x) != 'a']
    if aggregation == 'mean':
        return sum(similarities) / len(similarities)
    if aggregation == 'median':
        return np.median(np.array(similarities))
    else:
        raise ValueError('unsupported aggregation type: "{}"'.format(aggregation))


def calculate_mean_topic_coherence(keywords_per_topic, n_top_keywords=20, verbose=True):
    coherences = []

    for i, topic_keywords in tqdm.tqdm(enumerate(keywords_per_topic.values)):
        topic_keywords = list(topic_keywords)[:n_top_keywords]
        coherence = aggregated_similarity_measure(topic_keywords, similarity=get_wordnet_similarity)
        coherences.append(coherence)
        if verbose:
            print('topic', i, 'mean coherence:', coherence)
    return sum(coherences) / len(coherences)
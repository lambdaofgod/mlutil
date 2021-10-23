import numpy as np
import ot
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances


def get_stem_vectors(filtered_stems, keyed_vectors):
    return np.vstack(
        [
            np.mean([keyed_vectors[w] for w in stem_list], axis=0)
            for stem_list in filtered_stems
            if len(stem_list) > 0
        ]
    )


def get_word_vector_optimal_transport(
    word_vectors1, word_vectors2, ot_method=ot.sinkhorn, reg=0.01, normalize_dists=True
):
    cost = cosine_distances(word_vectors1, word_vectors2)
    height, width = cost.shape
    a = np.ones(height)
    b = np.ones(width)
    if normalize_dists:
        a = a / a.sum()
        b = b / b.sum()
    ot_matrix = ot_method(a, b, cost, reg=reg)
    return ot_matrix, (ot_matrix * cost).sum()


def make_annotated_ot_df(ot_matrix, valid_units1, valid_units2):
    ot_df = pd.DataFrame(ot_matrix)
    ot_df.index = [" ".join(l) for l in valid_units1 if len(l) > 0]
    ot_df.columns = [" ".join(l) for l in valid_units2 if len(l) > 0]
    return ot_df


def word2vec_vectorizer(word2vec, stem_extractor):
    def _vectorize(sent):
        return get_stem_vectors(stem_extractor(sent), word2vec)

    return _vectorize


def laser_vectorizer(laser, language="pl"):
    def _vectorize(sent_list):
        return laser.embed_sentences(sent_list, lang=language)

    return _vectorize


def get_optimal_transport_result(
    text1, text2, vectorizer, ot_method=ot.sinkhorn, **kwargs
):
    word_vectors1 = vectorizer(text1)
    word_vectors2 = vectorizer(text2)
    ot_matrix, ot_cost = get_word_vector_optimal_transport(
        word_vectors1, word_vectors2, ot_method=ot_method, **kwargs
    )
    return ot_matrix, ot_cost


def get_ot_result_df(text1, text2, vectorizer, stem_extractor=None, **kwargs):
    if type(text1) is list and type(text2) is list:
        valid_units1 = text1
        valid_units2 = text2
    else:
        if stem_extractor is None:
            stem_extractor = lambda x: x
        valid_units1 = stem_extractor(text1)
        valid_units2 = stem_extractor(text2)
    ot_matrix, ot_cost = get_optimal_transport_result(
        text1, text2, vectorizer, **kwargs
    )
    return make_annotated_ot_df(ot_matrix, valid_units1, valid_units2), ot_cost


def display_ot_result_df(ot_df, ot_cost, **kwargs):
    print("Total cost:{}".format(round(ot_cost, 3)))
    print(kwargs)
    sns.heatmap(ot_df, cmap="Blues")

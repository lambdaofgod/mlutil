import re

import nltk
import pandas as pd
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder


class LemmaTokenizer:
    def __init__(self, token_pattern):
        self.wnl = WordNetLemmatizer()
        self.token_pattern = re.compile(token_pattern)

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in self.token_pattern.findall(articles)]


class StemTokenizer:
    def __init__(self, token_pattern, stemmer_cls=nltk.stem.PorterStemmer):
        self.stemmer = stemmer_cls()
        self.token_pattern = re.compile(token_pattern)

    def __call__(self, articles):
        return [self.stemmer.stem(t) for t in self.token_pattern.findall(articles)]


def prepare_text_modeling_df(
    examples_df, vectorized_column, target_colum, vectorizer=CountVectorizer, **kwargs
):
    data, transformers = vectorize_text_df(
        examples_df, vectorized_column, vectorizer, **kwargs
    )
    target_data, target_transformers = encode_target(examples_df, target_column)
    return {**data, **target_data}, {**transformers, **target_transformers}


def vectorize_text_df(
    examples_df, vectorized_column, vectorizer=CountVectorizer(), **kwargs
):

    features = vectorizer.fit_transform(examples_df[vectorized_column])

    return ({"features": features}, {"vectorizer": vectorizer})


def encode_target(examples_df, target_column):
    le = LabelEncoder()
    ohe = OneHotEncoder()
    labels = le.fit_transform(examples_df[target_column]).reshape(-1, 1)
    labels_ohe = ohe.fit_transform(labels).todense()

    vectorized_data = {"labels": labels, "labels_onehot": labels_ohe}

    transformers = {"label_encoder": le, "onehot_encoder": ohe}
    return vectorized_data, transformers


##################
# WORD SIMILARITY
##################


def get_wordnet_similarity(
    word, another_word, similarity_method="resnik", pos=None, ic=None
):
    if ic is None:
        ic = wordnet_ic.ic("ic-semcor.dat")
    assert similarity_method in [
        "lin",
        "jcn",
        "resnik",
    ], "Unsupported similarity method: " + str(similarity_method)
    word_synset = wn.synsets(word, pos)[0]
    another_word_synset = wn.synsets(another_word, pos)[0]
    if similarity_method == "lin":
        return word_synset.lin_similarity(another_word_synset, ic)
    elif similarity_method == "jcn":
        return word_synset.jcn_similarity(another_word_synset, ic)
    else:
        return word_synset.res_similarity(another_word_synset, ic)

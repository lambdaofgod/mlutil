import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder

from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic


def prepare_text_modeling_df(examples_df, vectorized_column, target_colum, vectorizer=CountVectorizer, **kwargs):
    data, transformers = vectorize_text_df(examples_df, vectorized_column, vectorizer, **kwargs)
    target_data, target_transformers = encode_target(examples_df, target_column)
    return {**data, **target_data}, {**transformers, **target_transformers}


def vectorize_text_df(examples_df, vectorized_column,  vectorizer=CountVectorizer(), **kwargs):

    features = vectorizer.fit_transform(examples_df[vectorized_column])

    return ({'features': features},
            {'vectorizer': vectorizer})


def encode_target(examples_df, target_column):
    le = LabelEncoder()
    ohe = OneHotEncoder()
    labels = le.fit_transform(examples_df[target_column]).reshape(-1, 1)
    labels_ohe = ohe.fit_transform(labels).todense()
    
    vectorized_data = {
        'labels': labels,
        'labels_onehot': labels_ohe
    }
     
    transformers = {
        'label_encoder': le,
        'onehot_encoder': ohe
    }
    return vectorized_data, transformers

################
# TOPIC MODELING
################


def top_topic_words(model, feature_names, n_words=5):
    """
    Show n_words top words from topic model
    Works for models coming from sklearn.decomposition (for example NMF and LDA)
    """    

    def top_words(topic):
        return [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
    
    topic_words_dict = {'topic_' + str(topic_idx): top_words(topic) for topic_idx, topic in enumerate(model.components_)}
    
    return pd.DataFrame(topic_words_dict).T

##################
# WORD SIMILARITY
##################


def get_wordnet_similarity(word, another_word, similarity_method='resnik', pos=None, ic=None):
    if ic is None:
        ic = wordnet_ic.ic('ic-semcor.dat')
    assert similarity_method in ['lin', 'jcn', 'resnik'], 'Unsupported similarity method: ' + str(similarity_method)
    word_synset = wn.synsets(word, pos)[0]
    another_word_synset = wn.synsets(another_word, pos)[0]
    if similarity_method == 'lin':
        return word_synset.lin_similarity(another_word_synset, ic)
    elif similarity_method == 'jcn': 
        return word_synset.jcn_similarity(another_word_synset, ic)
    else:
        return word_synset.res_similarity(another_word_synset, ic)

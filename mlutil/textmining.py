from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder


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


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder


def vectorize_text_df(examples_df, vectorized_column, target_column, vectorizer=CountVectorizer, **kwargs):

    vectorizer = vectorizer(**kwargs)
    features = vectorizer.fit_transform(examples_df[vectorized_column])

    le = LabelEncoder()
    ohe = OneHotEncoder()
    labels = le.fit_transform(examples_df[target_column]).reshape(-1, 1)
    labels_ohe = ohe.fit_transform(labels).todense()
    vectorized_data = {
        'features': features,
        'labels': labels,
        'labels_onehot' : labels_ohe
    }
    return vectorized_data, (vectorizer, ohe, le)

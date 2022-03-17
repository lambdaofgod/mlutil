from mlutil.feature_extraction import embeddings
from sklearn import base


def get_reduced_embeddings_df(
    data, embedder: embeddings.EmbeddingVectorizer, reducer: base.BaseEstimator
):
    """
    run feature extraction with `embedder` and
    then dimensionality reduction with `reducer`
    """
    data_embeddings = embedder.transform(data)
    reduced_task_embeddings = reducer.fit_transform(data_embeddings)
    return reduced_task_embeddings

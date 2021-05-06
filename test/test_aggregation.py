from mlutil.feature_extraction import aggregation
from sklearn.feature_extraction.text import TfidfVectorizer



def test_aggregated_vectorizer():
    x = ['foo baz', 'foo', 'baz']
    group_keys = [1,2,2]
    vec = TfidfVectorizer()
    vec.fit(x)
    agg_vec = aggregation.AggregateVectorizer(vec)
    keys, grouped_features = agg_vec.aggregate_transform(x, group_keys)
    assert len(keys) == 2
    assert len(grouped_features) == 2

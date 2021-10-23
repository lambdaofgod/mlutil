import attr
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


@attr.s
class AggregateVectorizer:

    vectorizer: TransformerMixin = attr.ib()
    aggregation_method = attr.ib(np.mean)

    def aggregate_transform(self, X, group_keys):
        group_keys = pd.Series(group_keys)
        X_features = self.vectorizer.transform(X)

        keys = pd.Series(group_keys).unique()
        grouped_features = []
        for key in keys:
            group_features = self.aggregation_method(X_features[group_keys == key])
            grouped_features.append(group_features)
        return keys, np.row_stack(grouped_features)

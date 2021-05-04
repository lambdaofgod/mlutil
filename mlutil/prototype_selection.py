#export
import attr
import numpy as np
import pandas as pd
from sklearn import cluster, metrics, mixture
import tqdm


def select_centroid_prototype_indices(features, n_prototypes=10):
    n_prototypes = min(len(features), n_prototypes)
    clusterer = cluster.KMeans(n_prototypes)
    clusterer.fit(features)
    cluster_distances = metrics.pairwise.euclidean_distances(clusterer.cluster_centers_, features)
    prototype_indices = np.unique(cluster_distances.argmin(axis=1))
    return prototype_indices


@attr.s
class PrototypeSelector:

    vectorizer = attr.ib()
    n_prototypes = attr.ib(default=10)

    def fit_prototypes(self, data, labels):
        self.prototypes = {}
        data = np.array(data)
        labels = pd.Series(labels)
        for label in tqdm.tqdm(set(labels)):
            label_data = data[labels == label]
            label_features = self.vectorizer.transform(list(label_data))
            label_prototype_indices = select_centroid_prototype_indices(label_features, self.n_prototypes)
            self.prototypes[label] = np.array(label_data)[label_prototype_indices]

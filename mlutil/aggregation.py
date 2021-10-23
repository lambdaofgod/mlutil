import attr
from mlutil.feature_extraction import embeddings


@attr.s
class ClusteringGroupSubsampler:
    """
    subsamples vectors using group_ids
    for each group returns
    group_vectors: np.ndarray of shape (n, dim) where n <= max_vectors
    (it might be that n < max_vectors
    if the matrix is rank-deficient, or group has less than max_vectors elements)
    """

    vectors = attr.ib()
    group_ids = attr.ib()
    max_vectors: int = attr.ib(default=10)

    def get_group_aggregated_vectors(self, group_key):
        group_vectors = self.vectors[self.group_ids == group_key]
        n_clusters = min(np.linalg.matrix_rank(group_vectors), self.max_vectors)
        kmeans = cluster.KMeans(n_clusters=n_clusters)
        kmeans.fit(group_vectors)
        return kmeans.cluster_centers_

    def get_aggregated_vectors(self):
        unique_group_ids = pd.Series(self.group_ids).unique()
        aggregated_vectors = {}
        for group_name in tqdm.tqdm(unique_group_ids):
            group_vectors = self.vectors[self.group_ids == group_name]
            aggregated_vectors[group_name] = self.get_group_aggregated_vectors(
                group_name
            )
        return aggregated_vectors

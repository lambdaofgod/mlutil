import numpy as np
import tqdm
from sklearn import decomposition


class IncrementalHyperbolicMDS:
    def __init__(self, n_components, dtype="float16"):
        self.ipca = decomposition.IncrementalPCA(n_components=n_components)
        self.dtype = dtype

    def partial_fit(self, D):
        Y = -np.cosh(D)
        self.ipca.partial_fit(Y)

    def fit(self, D, batch_size, verbose=True):
        n = D.shape[0]
        idxs = np.arange(n)
        np.random.shuffle(idxs)
        for i in tqdm.tqdm(range(0, n - batch_size, batch_size)):
            self.partial_fit(D[idxs[i : i + batch_size]].astype(self.dtype))
        self.partial_fit(D[idxs[i:n]].astype(self.dtype))

    def transform(self, D, poincare_projection=True):
        X_reduced = self.ipca.transform(D)
        if poincare_projection:
            return X_reduced / (1 + np.sqrt(1 + np.linalg.norm(X, axis=1) ** 2))
        else:
            return X_reduced

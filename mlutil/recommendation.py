import numpy as np
import attr
from functools import partial
from sklearn import compose, preprocessing


def factorization_machine_column_transformer(columns):
    return compose.ColumnTransformer(
        [(col, preprocessing.OneHotEncoder(), [col]) for col in columns]
    )


def masked_error(true, pred, error_fn):
    """
    masked error for evaluating matrix factorization methods
    error is calculated only for nonzer entries - zero entries are treated as lack of information
    """
    true_nonzero = true > 0
    return (true_nonzero * error_fn(true, pred)).sum() / true_nonzero.sum()


def squared_error(x1, x2):
    return (x1 - x2) ** 2


def absolute_error(x1, x2):
    return np.abs(x1 - x2)


masked_mean_squared_error = partial(masked_error, error_fn=squared_error)


@attr.s
class EmbarrasinglyShallowAutoencoder:
    """
    model from Embarrassingly Shallow Autoencoders for Sparse Data
    https://arxiv.org/pdf/1905.03375.pdf
    """

    l2_regularization = attr.ib(default=1.0)
    eps = 1e-10

    def fit(self, X):
        gram_matrix = X.T @ X
        d = gram_matrix.shape[0]
        self._g_inv = np.linalg.pinv(
            gram_matrix.toarray() + self.l2_regularization * np.eye(d)
        )
        b = self._g_inv / -np.diag(self._g_inv)
        np.fill_diagonal(b, 0)
        self.components_ = b
        self.is_fitted = True
        return self

    def transform(self, X):
        assert self.is_fitted
        return X @ self.components_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def score_reconstruction(self, X, error_fn=squared_error):
        X_reconstruction = self.transform(X)
        return masked_error(X, X_reconstruction, error_fn)

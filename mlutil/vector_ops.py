import numpy as np


def concatpool(x, axis=0, ops=[np.max, np.mean]):
    pooled_vectors = [op(x, axis=axis) for op in ops]
    return np.column_stack(pooled_vectors)

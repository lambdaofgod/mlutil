import numpy as np
import pytest
from mlutil import vector_ops

@pytest.mark.slow
def test_concatpool():
    a = np.array([
        [1, 0, 1],
        [0, 1, 0]
    ])
    pooled_a = vector_ops.concatpool(a)
    expected_pooled_a = np.array([
        [1, 0.5],
        [1, 0.5],
        [1, 0.5]
    ])
    np.testing.assert_allclose(pooled_a, expected_pooled_a)

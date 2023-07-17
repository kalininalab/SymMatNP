import numpy as np

from symmatnp import SymMat


def test_astype():
    m = SymMat(4, 1)
    assert m.dtype == np.float64

    m = m.astype(np.int32)
    assert m.dtype == np.int32
    assert m.dtype != np.float64

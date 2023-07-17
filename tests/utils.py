import numpy as np
import pytest


@pytest.fixture
def a():
    return np.array([
        [1, 2, 3, 4],
        [2, 1, 5, 6],
        [3, 5, 1, 7],
        [4, 6, 7, 1],
    ])


@pytest.fixture
def b():
    return np.array([
        [1, 2, 3, 4],
    ])


@pytest.fixture
def c():
    return np.array([1])


@pytest.fixture
def d():
    return np.array([-1, -2, -3])
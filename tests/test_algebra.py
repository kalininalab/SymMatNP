import numpy as np

from symmatnp.symmat import SymMat
from tests.test_arithmetics import check_class
from tests.utils import a, b, c, d


def test_and(a, c):
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))

    ma = m.__and__(a)
    check_class(ma)
    assert np.allclose(ma.to_square(), m.to_square())

    mc = m.__and__(c)
    check_class(mc)
    assert np.allclose(mc.to_square(), SymMat(4, 1, np.array([0, 1, 0, 1, 0, 1])).to_square())


def test_eq(a, c):
    m = SymMat(4, 1, np.array([2, 1, 4, 1, 6, 7]))

    ma = m.__eq__(a)
    ma_sol = np.array([[True, True, False, True], [True, True, False, True],
                       [False, False, True, True], [True, True, True, True]])
    mc = m.__eq__(c)
    mc_sol = np.array([[True, False, True, False], [False, True, True, False],
                       [True, True, True, False], [False, False, False, True]])
    for i in range(4):
        for j in range(4):
            assert ma[i, j] == ma_sol[i, j]
            assert mc[i, j] == mc_sol[i, j]


def test_ge(a, c):
    m = SymMat(4, -1, np.array([2, 1, 4, 1, 6, 7]))

    ma = m.__ge__(a)
    ma_sol = np.array([[False, True, False, True], [True, False, False, True],
                       [False, False, False, True], [True, True, True, False]])
    mc = m.__ge__(c)
    mc_sol = np.array([[False, True, True, True], [True, False, True, True],
                       [True, True, False, True], [True, True, True, False]])
    for i in range(4):
        for j in range(4):
            assert ma[i, j] == ma_sol[i, j]
            assert mc[i, j] == mc_sol[i, j]


def test_gt(a, c):
    m = SymMat(4, -1, np.array([2, 1, 4, 1, 6, 7]))

    ma = m.__gt__(a)
    ma_sol = np.array([[False, False, False, False], [False, False, False, False],
                       [False, False, False, False], [False, False, False, False]])
    mc = m.__gt__(c)
    mc_sol = np.array([[False, True, False, True], [True, False, False, True],
                       [False, False, False, True], [True, True, True, False]])
    for i in range(4):
        for j in range(4):
            assert ma[i, j] == ma_sol[i, j]
            assert mc[i, j] == mc_sol[i, j]


def test_iand(a, c):
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    ma = m.__iand__(a)
    check_class(m)
    check_class(ma)
    assert np.allclose(m.to_square(), SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7])).to_square())
    assert np.allclose(ma.to_square(), SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7])).to_square())

    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    mc = m.__iand__(c)
    check_class(m)
    check_class(mc)
    assert np.allclose(m.to_square(), SymMat(4, 1, np.array([0, 1, 0, 1, 0, 1])).to_square())
    assert np.allclose(mc.to_square(), SymMat(4, 1, np.array([0, 1, 0, 1, 0, 1])).to_square())


def test_ior(a, c):
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    ma = m.__ior__(a)
    check_class(m)
    check_class(ma)
    assert np.allclose(m.to_square(), SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7])).to_square())
    assert np.allclose(ma.to_square(), SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7])).to_square())

    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    mc = m.__ior__(c)
    check_class(m)
    check_class(mc)
    assert np.allclose(m.to_square(), SymMat(4, 1, np.array([3, 3, 5, 5, 7, 7])).to_square())
    assert np.allclose(mc.to_square(), SymMat(4, 1, np.array([3, 3, 5, 5, 7, 7])).to_square())


def test_ixor(a, c):
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    ma = m.__ixor__(a)
    check_class(m)
    check_class(ma)
    assert np.allclose(m.to_square(), np.zeros((4, 4)))
    assert np.allclose(ma.to_square(), np.zeros((4, 4)))

    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    mc = m.__ixor__(c)
    check_class(m)
    check_class(mc)
    assert np.allclose(m.to_square(), SymMat(4, 0, np.array([3, 2, 5, 4, 7, 6])).to_square())
    assert np.allclose(mc.to_square(), SymMat(4, 0, np.array([3, 2, 5, 4, 7, 6])).to_square())


def test_le(a, c):
    m = SymMat(4, -1, np.array([4, 1, 6, 1, 8, 9]))

    ma = m.__le__(a)
    ma_sol = np.array([[True, False, True, False], [False, True, True, False],
                       [True, True, True, False], [False, False, False, True]])
    mc = m.__le__(c)
    mc_sol = np.array([[True, False, True, False], [False, True, True, False],
                       [True, True, True, False], [False, False, False, True]])
    for i in range(4):
        for j in range(4):
            assert ma[i, j] == ma_sol[i, j]
            assert mc[i, j] == mc_sol[i, j]


def test_lt(a, c):
    m = SymMat(4, -1, np.array([4, 1, 6, 1, 8, 9]))

    ma = m.__lt__(a)
    ma_sol = np.array([[True, False, True, False], [False, True, True, False],
                       [True, True, True, False], [False, False, False, True]])
    mc = m.__lt__(c)
    mc_sol = np.array([[True, False, False, False], [False, True, False, False],
                       [False, False, True, False], [False, False, False, True]])
    for i in range(4):
        for j in range(4):
            assert ma[i, j] == ma_sol[i, j]
            assert mc[i, j] == mc_sol[i, j]


def test_ne(a, c):
    m = SymMat(4, 1, np.array([2, 1, 4, 1, 6, 7]))

    ma = m.__ne__(a)
    ma_sol = np.array([[False, False, True, False], [False, False, True, False],
                       [True, True, False, False], [False, False, False, False]])
    mc = m.__ne__(c)
    mc_sol = np.array([[False, True, False, True], [True, False, False, True],
                       [False, False, False, True], [True, True, True, False]])
    for i in range(4):
        for j in range(4):
            assert ma[i, j] == ma_sol[i, j]
            assert mc[i, j] == mc_sol[i, j]


def test_or(a, c):
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))

    ma = m.__or__(a)
    check_class(ma)
    assert np.allclose(ma.to_square(), SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7])).to_square())

    mc = m.__or__(c)
    check_class(mc)
    assert np.allclose(mc.to_square(), SymMat(4, 1, np.array([3, 3, 5, 5, 7, 7])).to_square())


def test_rand(a, c):
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))

    ma = m.__rand__(a)
    check_class(ma)
    assert np.allclose(ma.to_square(), m.to_square())

    mc = m.__rand__(c)
    check_class(mc)
    assert np.allclose(mc.to_square(), SymMat(4, 1, np.array([0, 1, 0, 1, 0, 1])).to_square())


def test_ror(a, c):
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))

    ma = m.__ror__(a)
    check_class(ma)
    assert np.allclose(ma.to_square(), SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7])).to_square())

    mc = m.__ror__(c)
    check_class(mc)
    assert np.allclose(mc.to_square(), SymMat(4, 1, np.array([3, 3, 5, 5, 7, 7])).to_square())


def test_rxor(a, c):
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))

    ma = m.__rxor__(a)
    check_class(ma)
    assert np.allclose(ma.to_square(), np.zeros((4, 4)))

    mc = m.__rxor__(c)
    check_class(mc)
    assert np.allclose(mc.to_square(), SymMat(4, 0, np.array([3, 2, 5, 4, 7, 6])).to_square())


def test_xor(a, c):
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))

    ma = m.__xor__(a)
    check_class(ma)
    assert np.allclose(ma.to_square(), np.zeros((4, 4)))

    mc = m.__xor__(c)
    check_class(mc)
    assert np.allclose(mc.to_square(), SymMat(4, 0, np.array([3, 2, 5, 4, 7, 6])).to_square())

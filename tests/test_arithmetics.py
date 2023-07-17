import numpy as np
import pytest

from symmatnp import SymMat

from tests.utils import a, b, c, d


def check_class(obj):
    assert isinstance(obj, SymMat)
    assert hasattr(obj, "num")
    assert hasattr(obj, "diag")


def test_abs(a):
    m = SymMat(4, -1, np.array([-2, -3, -4, -5, -6, -7]))
    abs_m = abs(m)

    assert isinstance(abs_m, SymMat)
    assert np.allclose(abs_m.to_square(), a)


def test_add(a, b, c, d):
    m = SymMat(4, 1)

    ma = m.__add__(a)
    check_class(ma)
    assert np.allclose(ma.to_square(), a + np.eye(4))

    with pytest.raises(ValueError):
        m.__add__(b)

    mc = m.__add__(c)
    check_class(mc)
    assert np.allclose(mc.to_square(), np.ones((4, 4)) + np.eye(4))

    with pytest.raises(ValueError):
        m.__add__(d)


def test_complex():
    assert complex(SymMat(1, 1)) == complex(1)

    with pytest.raises(TypeError):
        complex(SymMat(2, 1))


def test_divmod(a, c):
    m = SymMat(4, 5, np.array([10, 10, 10, 10, 10, 10]))

    m_a_q, m_a_r = m.__divmod__(a)
    check_class(m_a_q)
    check_class(m_a_r)
    assert np.allclose(m_a_q.to_square(), SymMat(4, 5, np.array([5, 3, 2, 2, 1, 1])).to_square())
    assert np.allclose(m_a_r.to_square(), SymMat(4, 0, np.array([0, 1, 2, 0, 4, 3])).to_square())

    m_c_q, m_c_r = m.__divmod__(c)
    check_class(m_c_q)
    check_class(m_c_r)
    assert np.allclose(m_c_q.to_square(), m.to_square())
    assert np.allclose(m_c_r.to_square(), np.zeros((4, 4)))


def test_floordiv(a, c):
    m = SymMat(4, 5, np.array([10, 10, 10, 10, 10, 10]))

    m_a_q = m.__floordiv__(a)
    check_class(m_a_q)
    assert np.allclose(m_a_q.to_square(), SymMat(4, 5, np.array([5, 3, 2, 2, 1, 1])).to_square())

    m_c_q = m.__floordiv__(c)
    check_class(m_c_q)
    assert np.allclose(m_c_q.to_square(), m.to_square())


def test_iadd(a, c):
    m = SymMat(4, 1)
    ma = m.__iadd__(a)
    check_class(m)
    check_class(ma)
    assert np.allclose(m.to_square(), a + np.eye(4))
    assert np.allclose(ma.to_square(), a + np.eye(4))

    m = SymMat(4, 1)
    mc = m.__iadd__(c)
    check_class(m)
    check_class(mc)
    assert np.allclose(m.to_square(), np.ones((4, 4)) + np.eye(4))
    assert np.allclose(mc.to_square(), np.ones((4, 4)) + np.eye(4))


def test_ifloordiv(a, c):
    m = SymMat(4, 5, np.array([10, 10, 10, 10, 10, 10], dtype=np.float64))
    ma = m.__ifloordiv__(a)
    check_class(m)
    check_class(ma)
    assert np.allclose(m.to_square(), SymMat(4, 5, np.array([5, 3, 2, 2, 1, 1])).to_square())
    assert np.allclose(ma.to_square(), SymMat(4, 5, np.array([5, 3, 2, 2, 1, 1])).to_square())

    m = SymMat(4, 5, np.array([10, 10, 10, 10, 10, 10], dtype=np.float64))
    mc = m.__ifloordiv__(c)
    check_class(m)
    check_class(mc)
    assert np.allclose(m.to_square(), m.to_square())
    assert np.allclose(mc.to_square(), m.to_square())


def test_ilshift():
    m = SymMat(4, 5, np.array([1, 2, 3, 4, 5, 6]))
    ma = m.__ilshift__(1)
    check_class(m)
    check_class(ma)
    assert np.allclose(m.to_square(), SymMat(4, 10, np.array([2, 4, 6, 8, 10, 12])).to_square())
    assert np.allclose(ma.to_square(), SymMat(4, 10, np.array([2, 4, 6, 8, 10, 12])).to_square())


def test_imod(a, c):
    m = SymMat(4, 5, np.array([10, 10, 10, 10, 10, 10], dtype=np.float64))
    ma = m.__imod__(a)
    check_class(m)
    check_class(ma)
    assert np.allclose(m.to_square(), SymMat(4, 0, np.array([0, 1, 2, 0, 4, 3])).to_square())
    assert np.allclose(ma.to_square(), SymMat(4, 0, np.array([0, 1, 2, 0, 4, 3])).to_square())

    m = SymMat(4, 5, np.array([10, 10, 10, 10, 10, 10], dtype=np.float64))
    mc = m.__imod__(c)
    check_class(m)
    check_class(mc)
    assert np.allclose(m.to_square(), np.zeros((4, 4)))
    assert np.allclose(mc.to_square(), np.zeros((4, 4)))


def test_imul(a, c):
    m = SymMat(4, 0)
    ma = m.__imul__(a)
    check_class(m)
    check_class(ma)
    assert np.allclose(m.to_square(), np.zeros((4, 4)))
    assert np.allclose(ma.to_square(), np.zeros((4, 4)))

    m = SymMat(4, 0)
    mc = m.__imul__(c)
    check_class(m)
    check_class(mc)
    assert np.allclose(m.to_square(), np.zeros((4, 4)))
    assert np.allclose(mc.to_square(), np.zeros((4, 4)))


def test_invert():
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    mi = m.__invert__()
    check_class(mi)
    assert np.allclose(mi.to_square(), np.array([
        [-2, -3, -4, -5],
        [-3, -2, -6, -7],
        [-4, -6, -2, -8],
        [-5, -7, -8, -2],
    ]))


def test_irshift():
    m = SymMat(4, 5, np.array([1, 2, 3, 4, 5, 6]))
    ma = m.__irshift__(1)
    check_class(m)
    check_class(ma)
    assert np.allclose(m.to_square(), SymMat(4, 2, np.array([0, 1, 1, 2, 2, 3])).to_square())
    assert np.allclose(ma.to_square(), SymMat(4, 2, np.array([0, 1, 1, 2, 2, 3])).to_square())


def test_isub(a, c):
    m = SymMat(4, 1)
    ma = m.__isub__(a)
    check_class(m)
    check_class(ma)
    assert np.allclose(m.to_square(), -a + np.eye(4))
    assert np.allclose(ma.to_square(), -a + np.eye(4))

    m = SymMat(4, 1)
    mc = m.__isub__(c)
    check_class(m)
    check_class(mc)
    assert np.allclose(m.to_square(), -np.ones((4, 4)) + np.eye(4))
    assert np.allclose(mc.to_square(), -np.ones((4, 4)) + np.eye(4))


def test_itruediv(a, c):
    m = SymMat(4, 2, np.array([2, 3, 4, 5, 6, 7], dtype=np.float64))
    ma = m.__itruediv__(a)
    check_class(m)
    check_class(ma)
    assert np.allclose(m.to_square(), np.ones((4, 4)) + np.eye(4))
    assert np.allclose(ma.to_square(), np.ones((4, 4)) + np.eye(4))

    m = SymMat(4, 2, np.array([1, 2, 3, 4, 5, 6], dtype=np.float64))
    mc = m.__itruediv__(c)
    check_class(m)
    check_class(mc)
    assert np.allclose(m.to_square(), mc.to_square())


def test_lshift():
    m = SymMat(4, 5, np.array([1, 2, 3, 4, 5, 6]))
    ma = m.__lshift__(1)
    check_class(ma)
    assert np.allclose(ma.to_square(), SymMat(4, 10, np.array([2, 4, 6, 8, 10, 12])).to_square())


def test_mod(a, c):
    m = SymMat(4, 5, np.array([10, 10, 10, 10, 10, 10], dtype=np.float64))
    ma = m.__mod__(a)
    check_class(ma)
    assert np.allclose(ma.to_square(), SymMat(4, 0, np.array([0, 1, 2, 0, 4, 3])).to_square())

    m = SymMat(4, 5, np.array([10, 10, 10, 10, 10, 10], dtype=np.float64))
    mc = m.__mod__(c)
    check_class(mc)
    assert np.allclose(mc.to_square(), np.zeros((4, 4)))


def test_mul(a, c):
    m = SymMat(4, 0)
    ma = m.__mul__(a)
    check_class(ma)
    assert np.allclose(ma.to_square(), np.zeros((4, 4)))

    m = SymMat(4, 0)
    mc = m.__mul__(c)
    check_class(mc)
    assert np.allclose(mc.to_square(), np.zeros((4, 4)))


def test_neg():
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    mn = m.__neg__()
    check_class(mn)
    assert np.allclose(mn.to_square(), SymMat(4, -1, np.array([-2, -3, -4, -5, -6, -7])).to_square())


def test_pow():
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    mp = m.__pow__(2)
    check_class(mp)
    assert np.allclose(mp.to_square(), SymMat(4, 1, np.array([4, 9, 16, 25, 36, 49])).to_square())


def test_radd(a, c):
    m = SymMat(4, 1)

    ma = m.__radd__(a)
    check_class(ma)
    assert np.allclose(ma.to_square(), a + np.eye(4))

    mc = m.__radd__(c)
    check_class(mc)
    assert np.allclose(mc.to_square(), np.ones((4, 4)) + np.eye(4))


def test_rdivmod(a, c):
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    maq, mar = m.__rdivmod__(a)
    check_class(maq)
    check_class(mar)
    assert np.allclose(maq.to_square(), np.ones((4, 4)))
    assert np.allclose(mar.to_square(), np.zeros((4, 4)))

    mcq, mcr = m.__rdivmod__(c)
    check_class(mcq)
    check_class(mcr)
    assert np.allclose(mcq.to_square(), np.eye(4))
    assert np.allclose(mcr.to_square(), np.ones((4, 4)) - np.eye(4))


def test_rfloordiv():
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    md = m.__rfloordiv__(2)
    check_class(md)
    assert np.allclose(md.to_square(), SymMat(4, 2, np.array([1, 0, 0, 0, 0, 0])).to_square())


def test_rlshift():
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    ms = m.__rlshift__(2)
    check_class(ms)
    assert np.allclose(ms.to_square(), SymMat(4, 4, np.array([8, 16, 32, 64, 128, 256])).to_square())


def test_rmod():
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    mm = m.__rmod__(2)
    check_class(mm)
    assert np.allclose(mm.to_square(), SymMat(4, 0, np.array([0, 2, 2, 2, 2, 2])).to_square())


def test_rmul():
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    mm = m.__rmul__(2)
    check_class(mm)
    assert np.allclose(mm.to_square(), SymMat(4, 2, np.array([4, 6, 8, 10, 12, 14])).to_square())


def test_rpow():
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    mp = m.__rpow__(2)
    check_class(mp)
    assert np.allclose(mp.to_square(), SymMat(4, 2, np.array([4, 8, 16, 32, 64, 128])).to_square())


def test_rrshift():
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    mr = m.__rrshift__(2)
    check_class(mr)
    assert np.allclose(mr.to_square(), np.eye(4))


def test_rshift():
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    mr = m.__rshift__(2)
    check_class(mr)
    assert np.allclose(mr.to_square(), SymMat(4, 0, np.array([0, 0, 1, 1, 1, 1])).to_square())

    m = SymMat(4, 5, np.array([1, 2, 3, 4, 5, 6]))
    ma = m.__rshift__(1)
    check_class(ma)
    assert np.allclose(ma.to_square(), SymMat(4, 2, np.array([0, 1, 1, 2, 2, 3])).to_square())


def test_rsub(a, c):
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))

    ma = m.__rsub__(a)
    check_class(ma)
    assert np.allclose(ma.to_square(), np.zeros((4, 4)))

    mc = m.__rsub__(c)
    check_class(mc)
    assert np.allclose(mc.to_square(), SymMat(4, 0, np.array([-1, -2, -3, -4, -5, -6])).to_square())


def test_rtruediv(a, c):
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))

    ma = m.__rtruediv__(a)
    check_class(ma)
    assert np.allclose(ma.to_square(), np.ones((4, 4)))

    mc = m.__rtruediv__(c)
    check_class(mc)
    assert np.allclose(mc.to_square(), SymMat(4, 1, np.array([1/2, 1/3, 1/4, 1/5, 1/6, 1/7])).to_square())


def test_sub(a, c):
    m = SymMat(4, 1)
    ma = m.__sub__(a)
    check_class(ma)
    assert np.allclose(ma.to_square(), -a + np.eye(4))

    m = SymMat(4, 1)
    mc = m.__sub__(c)
    check_class(mc)
    assert np.allclose(mc.to_square(), -np.ones((4, 4)) + np.eye(4))


def test_truediv(a, c):
    m = SymMat(4, 2, np.array([2, 3, 4, 5, 6, 7], dtype=np.float64))
    ma = m.__truediv__(a)
    check_class(ma)
    assert np.allclose(ma.to_square(), np.ones((4, 4)) + np.eye(4))

    m = SymMat(4, 2, np.array([1, 2, 3, 4, 5, 6], dtype=np.float64))
    mc = m.__truediv__(c)
    check_class(mc)
    assert np.allclose(m.to_square(), SymMat(4, 2, np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)).to_square())


def test_clip():
    m = SymMat(4, 2, np.array([5, 6, 7, 8, 9, 8]))
    c = m.clip(5, 6)
    check_class(c)
    assert np.allclose(c.to_square(), SymMat(4, 5, np.array([5, 6, 6, 6, 6, 6])).to_square())


def test_diagonal():
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    f = m.diagonal()
    s = m.diagonal(offset=1)
    assert np.allclose(f, np.ones(4))
    assert np.allclose(s, np.array([2, 5, 7]))


def test_fill():
    m = SymMat(4, 0)
    m.fill(4)
    assert np.allclose(m.to_square(), np.ones((4, 4)) * 4)


def test_max():
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    t = m.max()
    c = m.max(axis=0)
    r = m.max(axis=1)
    assert np.allclose(t, 7)
    assert np.allclose(c, np.array([4, 6, 7, 7]))
    assert np.allclose(r, np.array([4, 6, 7, 7]))


def test_mean():
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    t = m.mean()
    c = m.mean(axis=0)
    r = m.mean(axis=1)
    assert np.allclose(t, np.array([3 + 5/8]))
    assert np.allclose(c, np.array([2.5, 3.5, 4, 4.5]))
    assert np.allclose(r, np.array([2.5, 3.5, 4, 4.5]))


def test_min():
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    t = m.min()
    c = m.min(axis=0)
    r = m.min(axis=1)
    assert np.allclose(t, 1)
    assert np.allclose(c, np.array([1, 1, 1, 1]))
    assert np.allclose(r, np.array([1, 1, 1, 1]))


def test_nonzero():
    m = SymMat(4, 0, np.array([1, 2, 0, 5, 0, 4]))
    indices = m.nonzero()
    assert indices[0].tolist() == [0, 0, 1, 1, 2, 2, 2, 3]
    assert indices[1].tolist() == [1, 2, 0, 2, 0, 1, 3, 2]


def test_prod():
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    t = m.prod()
    c = m.prod(axis=0)
    r = m.prod(axis=1)
    assert np.allclose(t, 25_401_600)
    assert np.allclose(c, np.array([24, 60, 105, 168]))
    assert np.allclose(r, np.array([24, 60, 105, 168]))


def test_sum():
    m = SymMat(4, 1, np.array([2, 3, 4, 5, 6, 7]))
    t = m.sum()
    c = m.sum(axis=0)
    r = m.sum(axis=1)
    assert np.allclose(t, 58)
    assert np.allclose(c, np.array([10, 14, 16, 18]))
    assert np.allclose(r, np.array([10, 14, 16, 18]))

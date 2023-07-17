import numbers
import pickle

import numpy as np


def from_square(array, dtype):
    out = SymMat(array.shape[0], array[0, 0])
    for i in range(array.shape[0]):
        out[i, (i + 1):] = array[i, (i + 1):]
    return out


def convert_single_index(i, j):
    return (i, j) if i < j else (j, i)


def triangular_number(x, y):
    x, y = min(x, y), max(x, y)
    return (y * (y - 1) - x * (x - 1)) // 2


def return_output(output, out, dtype):
    if out is None:
        return np.array(output, dtype=dtype)
    out = output
    return out


class SymMat(np.ndarray):
    """
    X 'T',
      '__array_interface__',
      '__array_priority__',
      '__array_struct__',
      '__doc__',
      '__hash__',
      'base',
    X 'ctypes',
    X 'data',
    X 'dtype',
    X 'flags',
      'flat',
    X 'imag',
      'itemsize',
      'nbytes',
    X 'ndim',
    X 'real',
    X 'shape',
    X 'size',
      'strides'
    """

    def __new__(cls, num: int, diag, initial: np.ndarray = None):
        if initial is None:
            obj = np.zeros(num * (num - 1) // 2).view(cls)
        else:
            if initial.shape != (num * (num - 1) // 2,):
                raise ValueError(f"The shape of the initial array ({initial.shape}) "
                                 f"and the number of rows and columns ({num}) doesn't match")
            obj = np.asarray(initial).view(cls)
        obj.num = num
        obj.diag = np.array(diag, dtype=obj.dtype)

        return obj

    def _convert_index(self, indices):
        if isinstance(indices, int) or isinstance(indices, slice):
            indices = (indices, slice(0, self.num, 1))
        if isinstance(indices, tuple) and len(indices) == 2:
            i, j = indices
            if isinstance(i, int) and isinstance(j, int):
                return [convert_single_index(i, j)], 1
            elif isinstance(i, slice) and isinstance(j, slice):
                return [
                    convert_single_index(a, b)
                    for a in range(i.start or 0, i.stop or self.num, i.step or 1)
                    for b in range(j.start or 0, j.stop or self.num, j.step or 1)
                ], ((i.stop or self.num) - (i.start or 0), -1)
            else:
                if isinstance(i, int) and isinstance(j, slice):
                    return [convert_single_index(i, y) for y in
                            range(j.start or 0, j.stop or self.num, j.step or 1)], (1, -1)
                if isinstance(i, slice) and isinstance(j, int):
                    return [convert_single_index(x, j) for x in
                            range(i.start or 0, i.stop or self.num, i.step or 1)], (-1, 1)

        raise IndexError("Invalid indices.")

    def _broadcast(self, other):
        if isinstance(other, numbers.Number):
            return np.array(other), other

        if isinstance(other, list):
            other = np.ndarray(other)

        if isinstance(other, np.ndarray):
            if other.shape == (1,) and np.isscalar(other[0]):
                return other, other[0]
            else:
                tmp = other.reshape(self.num, self.num)
                if np.allclose(tmp.T, tmp) and np.all(np.diag(tmp) == tmp[0, 0]):
                    triangle = np.zeros(self.num * (self.num - 1) // 2, dtype=other.dtype)
                    for i in range(self.num - 1):
                        t = triangular_number(self.num - i, self.num)
                        x = slice(t, (t + self.num - i - 1))
                        y = slice(i + 1, self.num)
                        triangle[x] = tmp[i, y]
                    return triangle, tmp[0, 0]
                else:
                    raise ValueError("The input matrix is not symmetric. Therefore the output of this operation is not "
                                     "symmetric anymore and cannot be stored in this class. Please call .to_square() "
                                     "to get a regular numpy array and perform the operation on that.")
        raise ValueError(f"Cannot add object of type {type(other)} to a triangular Matrix.")

    @property
    def T(self):
        return self.transpose()

    @property
    def data(self):
        return super().data

    @property
    def dtype(self):
        return super().dtype

    @property
    def flags(self):
        return super().flags

    @property
    def imag(self):
        imag = self.diag.imag if hasattr(self.diag, "imag") else 0
        return SymMat(self.num, imag, super().imag)

    @property
    def ndim(self):
        return 2

    @property
    def real(self):
        real = self.diag.real if hasattr(self.diag, "real") else self.diag
        return SymMat(self.num, real, super().real)

    @property
    def shape(self):
        return self.num, self.num

    @property
    def size(self):
        return self.num ** 2

    def __abs__(self):
        return SymMat(num=self.num, diag=abs(self.diag), initial=self.view(np.ndarray).__abs__())

    def __add__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(num=self.num, diag=self.diag + val, initial=self.view(np.ndarray).__add__(arr))

    def __and__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(num=self.num, diag=self.diag.__and__(val), initial=self.view(np.ndarray).__and__(arr))

    def __array__(self, dtype=None, *args, **kwargs):
        if dtype is None:
            return self
        return SymMat(num=self.num, diag=self.diag, initial=self.view(np.ndarray).__array__(dtype, *args, **kwargs))

    # def __array_finalize__(self, obj, *args, **kwargs):
    #     raise NotImplementedError()

    # def __array_function__(self, *args, **kwargs):
    #     raise NotImplementedError()

    # def __array_prepare__(self, array, context=None, *args, **kwargs):
    #     raise NotImplementedError

    # def __array_ufunc__(self, *args, **kwargs):
    #     raise NotImplementedError()

    # def __array_wrap__(self, array, context=None, *args, **kwargs):
    #     raise NotImplementedError()

    def __bool__(self):
        raise ValueError("The truth value of a symmetric matrix is not defined. "
                         "All logical operations are therefore undefined.")

    # def __class__(self: _T) -> Type[_T]:
    #     raise NotImplementedError()

    # def __class_getitem__(cls, item):
    #     raise NotImplementedError()

    def __complex__(self):
        if self.__len__() == 1:
            return complex(self.diag, 0)
        else:
            raise TypeError("only length-1 arrays can be converted to Python scalars")

    def __contains__(self, item):
        raise NotImplementedError()

    def __copy__(self):
        tmp = SymMat(self.num, self.diag)
        for i in range(self.num):
            for j in range(i + 1, self.num):
                tmp[i, j] = self[i, j]
        return tmp

    def __deepcopy__(self, memodict={}):
        return self.__copy__()

    # def __delattr__(self, item):
    #     raise NotImplementedError()

    # def __dir__(self) -> Iterable[str]:
    #     raise NotImplementedError()

    def __divmod__(self, other):
        arr, val = self._broadcast(other)
        arr_q, arr_r = self.view(np.ndarray).__divmod__(arr)
        return SymMat(self.num, self.diag // val, arr_q), SymMat(self.num, self.diag % val, arr_r)

    # def __dlpack__(self, *args, **kwargs):
    #     raise NotImplementedError()

    # def __dlpack_device__(self):
    #     raise NotImplementedError()

    def __eq__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(self.num, self.diag == val, self.view(np.ndarray).__eq__(arr))

    def __float__(self):
        if self.__len__() == 1:
            return float(self.diag)
        raise TypeError("only length-1 arrays can be converted to Python scalars")

    def __floordiv__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(self.num, self.diag // val, self.view(np.ndarray).__floordiv__(arr))

    # def __format__(self, format_spec):
    #     raise NotImplementedError()

    def __ge__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(self.num, self.diag >= val, self.view(np.ndarray).__ge__(arr))

    # def __getattribute__(self, item):
    #     return getattr(self.obj, item)

    def __getitem__(self, indices):
        converted_indices, shape = self._convert_index(indices)
        values = []
        for i, j in converted_indices:
            if i == j:
                values.append(self.diag)
            else:
                index = triangular_number(self.num - i, self.num) + j - i - 1
                values.append(self.view(np.ndarray).__getitem__(index))
        if shape == 1:
            return np.array(values[0])
        return np.array(values).reshape(shape)

    def __gt__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(self.num, self.diag > val, self.view(np.ndarray).__gt__(arr))

    def __iadd__(self, other):
        arr, val = self._broadcast(other)
        self.view(np.ndarray).__iadd__(arr)
        self.diag.__iadd__(val)
        return self

    def __iand__(self, other):
        arr, val = self._broadcast(other)
        self.view(np.ndarray).__iand__(arr)
        self.diag.__iand__(val)
        return self

    def __ifloordiv__(self, other):
        arr, val = self._broadcast(other)
        self.view(np.ndarray).__ifloordiv__(arr)
        self.diag.__ifloordiv__(val)
        return self

    def __ilshift__(self, other):
        self.view(np.ndarray).__ilshift__(other)
        self.diag <<= other
        return self

    def __imatmul__(self, other):
        raise NotImplementedError(
            "The output of matrix multiplication with a symmetric matrix is not necessarily a symmetric matrix. "
            "Therefore, this cannot be provided in a memory efficient way. Please call .to_square() first."
        )

    def __imod__(self, other):
        arr, val = self._broadcast(other)
        self.view(np.ndarray).__imod__(arr)
        self.diag.__imod__(val)
        return self

    def __imul__(self, other):
        arr, val = self._broadcast(other)
        self.view(np.ndarray).__imul__(arr)
        self.diag.__imul__(val)
        return self

    def __index__(self):
        if self.num == 1:
            return int(self.diag)
        raise TypeError("only length-1 arrays can be converted to Python scalars")

    # def __init__(self, num, diag, initial):
    #     super().__init__(num * (num - 1) // 2)

    # def __init_subclass__(cls, **kwargs):
    #     raise NotImplementedError()

    def __int__(self):
        if self.__len__() == 1:
            return int(self.diag)
        raise TypeError("only length-1 arrays can be converted to Python scalars")

    def __invert__(self):
        return SymMat(num=self.num, diag=~self.diag, initial=self.view(np.ndarray).__invert__())

    def __ior__(self, other):
        arr, val = self._broadcast(other)
        self.view(np.ndarray).__ior__(arr)
        self.diag.__ior__(val)
        return self

    def __ipow__(self, other):
        self.view(np.ndarray).__ipow__(other)
        self.diag.__ipow(other)
        return self

    def __irshift__(self, other):
        self.view(np.ndarray).__irshift__(other)
        self.diag.__irshift__(other)
        return self

    def __isub__(self, other):
        arr, val = self._broadcast(other)
        self.view(np.ndarray).__isub__(arr)
        self.diag.__isub__(val)
        return self

    def __iter__(self):
        raise NotImplementedError(
            "The output of matrix multiplication with a symmetric matrix is not necessarily a symmetric matrix. "
            "Therefore, this cannot be provided in a memory efficient way. Please call .to_square() first."
        )

    def __itruediv__(self, other):
        arr, val = self._broadcast(other)
        self.view(np.ndarray).__itruediv__(arr)
        self.diag.__itruediv__(val)
        return self

    def __ixor__(self, other):
        arr, val = self._broadcast(other)
        self.view(np.ndarray).__ixor__(arr)
        self.diag.__ixor__(val)
        return self

    def __le__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(self.num, self.diag.__le__(val), self.view(np.ndarray).__le__(arr))

    def __len__(self):
        return self.num

    def __lshift__(self, other):
        return SymMat(self.num, self.diag.__lshift__(other), self.view(np.ndarray).__lshift__(other))

    def __lt__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(self.num, self.diag.__lt__(val), self.view(np.ndarray).__lt__(arr))

    def __matmul__(self, other):
        raise NotImplementedError(
            "The output of matrix multiplication with a symmetric matrix is not necessarily a symmetric matrix. "
            "Therefore, this cannot be provided in a memory efficient way. Please call .to_square() first."
        )

    def __mod__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(self.num, self.diag.__mod__(val), self.view(np.ndarray).__mod__(arr))

    def __mul__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(self.num, self.diag.__mul__(val), self.view(np.ndarray).__mul__(arr))

    def __ne__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(self.num, self.diag != val, self.view(np.ndarray).__ne__(arr))

    def __neg__(self):
        return SymMat(self.num, -abs(self.diag), self.view(np.ndarray).__neg__())

    def __or__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(num=self.num, diag=self.diag.__or__(val), initial=self.view(np.ndarray).__or__(arr))

    def __pos__(self):
        self.view(np.ndarray).__pos__()
        self.diag = abs(self.diag)
        return self

    def __pow__(self, power, modulo=None):
        return SymMat(num=self.num, diag=self.diag.__pow__(power, modulo), initial=self.view(np.ndarray).__pow__(power, modulo))

    def __radd__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(num=self.num, diag=self.diag.__add__(val), initial=self.view(np.ndarray).__radd__(arr))

    def __rand__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(num=self.num, diag=self.diag.__and__(val), initial=self.view(np.ndarray).__rand__(arr))

    def __rdivmod__(self, other):
        arr, val = self._broadcast(other)
        q_arr, r_arr = self.view(np.ndarray).__rdivmod__(arr)
        q_val, r_val = self.diag.__rdivmod__(val)
        return SymMat(num=self.num, diag=q_val, initial=q_arr), SymMat(num=self.num, diag=r_val, initial=r_arr)

    def __reduce__(self):
        return (self.num, self.diag, super).__reduce__()

    def __reduce_ex__(self, protocol):
        return (self.num, self.diag, super).__reduce_ex__(protocol)

    def __repr__(self):
        return (self.num, self.diag, super).__repr__()

    def __rfloordiv__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(num=self.num, diag=self.diag.__rfloordiv__(val), initial=self.view(np.ndarray).__rfloordiv__(arr))

    def __rlshift__(self, other):
        return SymMat(num=self.num, diag=self.diag.__rlshift__(other), initial=self.view(np.ndarray).__rlshift__(other))

    def __rmatmul__(self, other):
        raise NotImplementedError()

    def __rmod__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(num=self.num, diag=self.diag.__rmod__(val), initial=self.view(np.ndarray).__rmod__(arr))

    def __rmul__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(num=self.num, diag=self.diag.__rmul__(val), initial=self.view(np.ndarray).__rmul__(arr))

    def __ror__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(num=self.num, diag=self.diag.__ror__(val), initial=self.view(np.ndarray).__ror__(arr))

    def __rpow__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(num=self.num, diag=self.diag.__rpow__(val), initial=self.view(np.ndarray).__rpow__(arr))

    def __rrshift__(self, other):
        return SymMat(num=self.num, diag=self.diag.__rrshift__(other), initial=self.view(np.ndarray).__rrshift__(other))

    def __rshift__(self, other):
        return SymMat(num=self.num, diag=self.diag.__rshift__(other), initial=self.view(np.ndarray).__rshift__(other))

    def __rsub__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(num=self.num, diag=self.diag.__rsub__(val), initial=self.view(np.ndarray).__rsub__(arr))

    def __rtruediv__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(num=self.num, diag=self.diag.__rtruediv__(val), initial=self.view(np.ndarray).__rtruediv__(arr))

    def __rxor__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(num=self.num, diag=self.diag.__rxor__(val), initial=self.view(np.ndarray).__rxor__(arr))

    def __setattr__(self, key, value):
        super().__setattr__(key, value)

    def __setitem__(self, indices, value):
        converted_indices, shape = self._convert_index(indices)
        value = value.flatten()
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        if len(value) != len(converted_indices):
            raise ValueError("Invalid value shape.")

        for val, (i, j) in zip(value, converted_indices):
            if i == j:
                self.diag = val
            else:
                index = triangular_number(i, j) + j - i - 1
                self.view(np.ndarray).__setitem__(index, val)

    # def __setstate__(self, state):
    #     raise NotImplementedError()

    def __sizeof__(self):
        return self.view(np.ndarray).__sizeof__() + self.num.__sizeof__() + self.diag.__sizeof__()

    def __str__(self):
        raise NotImplementedError(
            "This operation is not applicable to symmetric matrices as the symmetry property and therefore the "
            "advantageous memory usage will be lost. Please call .to_square() first."
        )

    def __sub__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(num=self.num, diag=self.diag - val, initial=self.view(np.ndarray).__sub__(arr))

    # def __subclasscheck__(self, subclass):
    #     raise NotImplementedError()

    def __truediv__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(num=self.num, diag=self.diag / val, initial=self.view(np.ndarray).__truediv__(arr))

    def __xor__(self, other):
        arr, val = self._broadcast(other)
        return SymMat(num=self.num, diag=self.diag.__xor__(val), initial=self.view(np.ndarray).__xor__(arr))

    def all(self, axis=None, out=None, keepdims=False, *args, **kwargs):
        return self.view(np.ndarray).all(axis, out, keepdims, *args, **kwargs) and self.diag

    def any(self, axis=None, out=None, keepdims=False, *args, **kwargs):
        return self.view(np.ndarray).any(axis, out, keepdims, *args, **kwargs) or self.diag

    def argmax(self, axis=None, out=None, *args, **kwargs):
        raise NotImplementedError("This functionality is not yet implemented.")

    def argmin(self, axis=None, out=None, *args, **kwargs):
        raise NotImplementedError("This functionality is not yet implemented.")

    def argpartition(self, kth, axis=-1, kind='introselect', order=None):
        raise NotImplementedError("This functionality it not yet implemented.")

    def argsort(self, axis=-1, kind=None, order=None):
        raise NotImplementedError("This functionality is not yet implemented.")

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        return SymMat(self.num, self.diag, self.view(np.ndarray).astype(dtype, order, casting, subok, copy))

    def choose(self, choices, out=None, mode='raise'):
        raise NotImplementedError(
            "This operation is not applicable to symmetric matrices as the symmetry property and therefore the "
            "advantageous memory usage will be lost. Please call .to_square() first."
        )

    def clip(self, mini=None, maxi=None, out=None, **kwargs):
        return SymMat(
            self.num,
            min(max(self.diag, mini or self.diag), maxi or self.diag),
            self.view(np.ndarray).clip(mini, maxi, **kwargs)
        )

    def compress(self, condition, axis=None, out=None):
        raise NotImplementedError(
            "This operation is not applicable to symmetric matrices as the symmetry property and therefore the "
            "advantageous memory usage will be lost. Please call .to_square() first."
        )

    def conj(self):
        return SymMat(self.num, self.diag.conjugate(), self.view(np.ndarray).conj())

    def conjugate(self):
        return SymMat(self.num, self.diag.conjugate(), self.view(np.ndarray).conjugate())

    def copy(self, order='C'):
        return self.__copy__()

    def cumprod(self, axis=None, dtype=None, out=None):
        raise NotImplementedError()

    def cumsum(self, axis=None, dtype=None, out=None):
        raise NotImplementedError()

    def diagonal(self, offset=0, axis1=0, axis2=1):
        offset = abs(offset)
        return np.array([self[i, i + offset] for i in range(self.num - offset)])

    # def dot(self, *args, **kwargs):
    #     raise NotImplementedError()

    def dump(self, file):
        with open(file, "wb") as out:
            pickle.dump((self.num, self.diag, super), out)

    def dumps(self):
        pickle.dumps((self.num, self.diag, super))

    def fill(self, value):
        self.view(np.ndarray).fill(value)
        self.diag = value

    def flatten(self, order='C'):
        raise NotImplementedError(
            "This operation is not applicable to symmetric matrices as the symmetry property and therefore the "
            "advantageous memory usage will be lost. Please call .to_square() first."
        )

    def getfield(self, dtype, offset=0):
        raise NotImplementedError("This functionality is not yet implemented.")

    def item(self, *args):
        if args is None:
            if self.num == 1:
                return np.array(self.diag)
            else:
                raise ValueError("Can only return item for scalar matrix, i.e. one-element matrix.")
        return np.array(self[args])

    def max(self, axis=None, out=None, keepdims=False, initial=None, *args, **kwargs):
        if axis is None:
            output = max(self.view(np.ndarray).max(None, None, keepdims, initial, *args, **kwargs), self.diag)
        else:
            output = [self[i, :].max(initial=initial, *args, **kwargs) for i in range(self.num)]
        return return_output(output, out, self.dtype)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *args, **kwargs):
        if axis is None:
            output = self.diag / self.num + self.view(np.ndarray).mean(*args, **kwargs) * (self.num - 1) / self.num
        else:
            output = [self[i, :].mean(dtype=dtype, *args, **kwargs) for i in range(self.num)]
        return return_output(output, out, dtype)

    def min(self, axis=None, out=None, keepdims=False, initial=None, *args, **kwargs):
        if axis is None:
            output = min(self.view(np.ndarray).min(None, None, keepdims, initial, *args, **kwargs), self.diag)
        else:
            output = [self[i, :].min(initial=initial, *args, **kwargs) for i in range(self.num)]
        return return_output(output, out, self.dtype)

    def newbyteorder(self, new_order='S', *args, **kwargs):
        return SymMat(self.num, self.diag, self.view(np.ndarray).newbyteorder(new_order, *args, **kwargs))

    def nonzero(self):
        output = []
        if self.diag != 0:
            for i in range(self.num):
                output.append([i, i])
        # for idx in super().nonzero():
        #     pass
        for i in range(self.num):
            for j in range(i + 1, self.num):
                if self[i, j] != 0:
                    output += [(i, j), (j, i)]
        output = list(zip(*tuple(sorted(output))))
        return np.array(output[0]), np.array(output[1])

    def partition(self, kth, axis=-1, kind='introselect', order=None):
        raise NotImplementedError(
            "This operation is not applicable to symmetric matrices as the symmetry property and therefore the "
            "advantageous memory usage might be lost. Please call .to_square() first."
        )

    def prod(self, axis=None, dtype=None, out=None, keepdims=False, initial=1, where=True):
        if axis is None:
            output = initial * self.view(np.ndarray).prod(dtype=dtype, where=where) ** 2 * self.diag ** self.num
        else:
            output = [self[i, :].prod(dtype=dtype, initial=initial, where=where) for i in range(self.num)]
        return return_output(output, out, dtype)

    def ptp(self, axis=None, out=None, keepdims=False):
        minimum = self.min(axis=axis, out=out, keepdims=keepdims)
        maximum = self.min(axis=axis, out=out, keepdims=keepdims)
        return return_output(maximum - minimum, out, dtype=self.dtype)

    def put(self, indices, values, mode='raise'):
        raise NotImplementedError(
            "This operation is not applicable to symmetric matrices as the symmetry property and therefore the "
            "advantageous memory usage might be lost. Please call .to_square() first."
        )

    def ravel(self, order=None):
        raise NotImplementedError(
            "This operation is not applicable to symmetric matrices as the symmetry property and therefore the "
            "advantageous memory usage might be lost. Please call .to_square() first."
        )

    def repeat(self, repeats, axis=None):
        raise NotImplementedError(
            "This operation is not applicable to symmetric matrices as the symmetry property and therefore the "
            "advantageous memory usage might be lost. Please call .to_square() first."
        )

    def reshape(self, shape, order='C'):
        raise NotImplementedError(
            "This operation is not applicable to symmetric matrices as the symmetry property and therefore the "
            "advantageous memory usage might be lost. Please call .to_square() first."
        )

    def round(self, decimals=0, out=None):
        output = SymMat(num=self.num, diag=round(self.diag, decimals), initial=self.view(np.ndarray).round(decimals))
        return return_output(output, out, self.dtype)

    def searchsorted(self, v, side='left', sorter=None):
        raise NotImplementedError(
            "This operation is not applicable to symmetric matrices as the symmetry property and therefore the "
            "advantageous memory usage might be lost. Please call .to_square() first."
        )

    def setfield(self, val, dtype, offset=0):
        raise NotImplementedError("This functionality is not yet implemented")

    def setflags(self, write=None, align=None, uic=None):
        self.view(np.ndarray).setflags(write, align, uic)

    def sort(self, axis=-1, kind=None, order=None):
        raise NotImplementedError(
            "This operation is not applicable to symmetric matrices as the symmetry property and therefore the "
            "advantageous memory usage might be lost. Please call .to_square() first."
        )

    def squeeze(self, axis=None):
        raise NotImplementedError(
            "This operation is not applicable to symmetric matrices as the symmetry property and therefore the "
            "advantageous memory usage might be lost. Please call .to_square() first."
        )

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *args, **kwargs):
        if axis is None:
            if self.num % 2 == 0:
                output = np.concatenate([super, np.array([self.diag] * (self.num // 2))]).std(
                    dtype=dtype, ddof=ddof, keepdims=keepdims, *args, **kwargs
                )
            else:
                raise NotImplementedError(
                    "The STD calculation for odd-sized symmetric matrices is not implemented in a memory efficient way."
                )
        else:
            output = [self[i, :].std(dtype=dtype, ddof=ddof, *args, **kwargs) for i in range(self.num)]
        return return_output(output, out, dtype)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False, initial=0, where=True):
        if axis is None:
            output = initial + self.view(np.ndarray).sum(dtype=dtype, where=where) * 2 + self.diag * self.num
        else:
            output = [self[i, :].sum(dtype=dtype, initial=initial, where=where) for i in range(self.num)]
        return return_output(output, out, dtype)

    def swapaxes(self, axis1, axis2):
        return self.__copy__()

    def take(self, indices, axis=None, out=None, mode='raise'):
        if axis is None:
            output = np.array(
                self[index] for index in indices
            )
        elif axis == 0:
            output = np.array(
                self[index, :] for index in indices
            )
        elif axis == 1:
            output = np.array(
                self[:, index] for index in indices
            )
        else:
            raise ValueError(f"Invalid value ({axis}) argument for 'axis'.")
        return return_output(output, out, self.dtype)

    def tobytes(self, order='C'):
        raise NotImplementedError()

    def tolist(self):
        raise NotImplementedError(
            "This operation is not applicable to symmetric matrices as the symmetry property and therefore the "
            "advantageous memory usage might be lost. Please call .to_square() first."
        )

    def tofile(self, fid, sep="", format="%s"):
        raise NotImplementedError()

    def to_square(self):
        out = np.zeros((self.num, self.num))
        for i in range(self.num):
            out[i, (i + 1):] = out[(i + 1):, i] = self[i, (i + 1):]
            out[i, i] = self.diag
        return out

    def tostring(self, order='C'):
        raise NotImplementedError(
            "This operation is not applicable to symmetric matrices as the symmetry property and therefore the "
            "advantageous memory usage might be lost. Please call .to_square() first."
        )

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        offset = abs(offset)
        if offset == 0:
            output = self.diag * self.num
        else:
            output = sum(self[i, i + offset] for i in range(self.num - offset))
        return return_output(output, out, dtype)

    def transpose(self, *axes):
        return self.__copy__()

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *args, **kwargs):
        output = self.std(axis, dtype, out, ddof, keepdims, *args, **kwargs) ** 2
        return return_output(output, out, dtype)

    # def view(self, dtype=None, *args, **kwargs):
    #     raise NotImplementedError(
    #         "This operation is not applicable to symmetric matrices as the symmetry property and therefore the "
    #         "advantageous memory usage might be lost. Please call .to_square() first."
    #     )

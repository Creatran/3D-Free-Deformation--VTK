import numpy as np
from scipy.special import comb

def mesh3d(x, y, z, dtype=np.float32):
    # Form the 3D mesh grid
    grid = np.empty(x.shape + y.shape + z.shape + (3,), dtype=dtype)
    grid[..., 0] = x[:, np.newaxis, np.newaxis]
    grid[..., 1] = y[np.newaxis, :, np.newaxis]
    grid[..., 2] = z[np.newaxis, np.newaxis, :]
    return grid


def extent(x, *args, **kwargs):
    return np.min(x, *args, **kwargs), np.max(x, *args, **kwargs)


def get_stu_deformation_matrix(stu, dims):
    # get the deform matrix
    v = mesh3d(
        *(np.arange(0, d+1, dtype=np.int32) for d in dims),
        dtype=np.int32)
    v = np.reshape(v, (-1, 3))

    weights = bernstein_poly(
        n=np.array(dims, dtype=np.int32),
        v=v,
        stu=np.expand_dims(stu, axis=-2))

    b = np.prod(weights, axis=-1)
    return b



def bernstein_poly(n, v, stu):
    # construct Bernstein poly
    coeff = comb(n, v)
    weights = coeff * ((1 - stu) ** (n - v)) * (stu ** v)
    return weights


def trivariate_bernstein(stu, lattice):
    # get Bernstein matrix
    if len(lattice.shape) != 4 or lattice.shape[3] != 3:
        raise ValueError('lattice must have shape (L, M, N, 3)')
    l, m, n = (d - 1 for d in lattice.shape[:3])
    lmn = np.array([l, m, n], dtype=np.int32)
    v = mesh3d(
        np.arange(l+1, dtype=np.int32),
        np.arange(m+1, dtype=np.int32),
        np.arange(n+1, dtype=np.int32),
        dtype=np.int32)
    stu = np.reshape(stu, (-1, 1, 1, 1, 3))
    weights = bernstein_poly(n=lmn, v=v, stu=stu)
    weights = np.prod(weights, axis=-1, keepdims=True)
    return np.sum(weights * lattice, axis=(1, 2, 3))

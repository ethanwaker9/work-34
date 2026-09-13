import math

import numpy as np
from fpylll import IntegerMatrix


def rot_rows(a, q=None):
    n = len(a)
    rows = []
    for i in range(n):
        row = [0] * n
        for j in range(n):
            k = j - i
            v = a[k] if k >= 0 else -a[k + n]
            row[j] = v % q if q is not None else v
        rows.append(row)
    return rows


def ntru_basis(h, q):
    n = len(h)
    H = rot_rows(h, q)
    A = IntegerMatrix(2 * n, 2 * n)
    for i in range(n):
        A[i, i] = 1
        for j in range(n):
            A[i, n + j] = int(H[i][j])
    for i in range(n):
        A[n + i, n + i] = int(q)
    return A


def descended_basis(hp, q):
    return ntru_basis(hp, q)


def subring_basis(h, q, r, ell):
    n = len(h)
    nprime = n // r
    H = rot_rows(h, q)
    ell = min(ell, n)
    d = ell + nprime
    A = IntegerMatrix(d, d)
    for j in range(nprime):
        src = H[(r * j) % n]
        sgn = -1 if (r * j) >= n else 1
        for i in range(ell):
            A[j, i] = int((sgn * src[i]) % q)
        A[j, ell + j] = 1
    for i in range(ell):
        A[nprime + i, i] = int(q)
    return A


def module_basis(x, y):
    n = len(x)
    Rx = rot_rows(x)
    Ry = rot_rows(y)
    A = IntegerMatrix(n, 2 * n)
    for i in range(n):
        for j in range(n):
            A[i, j] = int(Rx[i][j])
            A[i, n + j] = int(Ry[i][j])
    return A


def log_volume(A):
    m = np.array([[float(A[i, j]) for j in range(A.ncols)] for i in range(A.nrows)])
    scale = np.max(np.abs(m))
    if scale == 0:
        return -math.inf
    m = m / scale
    s = np.linalg.svd(m, compute_uv=False)
    return float(np.sum(np.log(s))) + A.nrows * math.log(scale)


def shortest_row(A):
    best = None
    bi = 0
    for i in range(A.nrows):
        s = sum(int(x) ** 2 for x in A[i])
        if s > 0 and (best is None or s < best):
            best = s
            bi = i
    return math.sqrt(float(best)), bi

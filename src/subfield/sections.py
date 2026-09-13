import math

import numpy as np
from fpylll import BKZ, GSO, IntegerMatrix, LLL, FPLLL
from fpylll.algorithms.bkz2 import BKZReduction

from .reduction import bkz_param


def coordinates(B, rows, prec=256):
    FPLLL.set_precision(prec)
    M = GSO.Mat(IntegerMatrix.from_matrix(B), float_type="mpfr")
    M.update_gso()
    out = []
    for v in rows:
        c = M.babai([int(x) for x in v])
        out.append([int(x) for x in c])
    return out


def kernel_of_form(w):
    m = len(w)
    U = [[1 if i == j else 0 for j in range(m)] for i in range(m)]
    v = [int(x) for x in w]
    piv = 0
    while piv < m and v[piv] == 0:
        piv += 1
    if piv == m:
        return U
    v[0], v[piv] = v[piv], v[0]
    U[0], U[piv] = U[piv], U[0]
    for j in range(1, m):
        while v[j] != 0:
            if abs(v[0]) > abs(v[j]):
                v[0], v[j] = v[j], v[0]
                U[0], U[j] = U[j], U[0]
            t = v[j] // v[0]
            if t:
                v[j] -= t * v[0]
                Uj, U0 = U[j], U[0]
                for x in range(m):
                    Uj[x] -= t * U0[x]
    return [U[j] for j in range(1, m)]


def _lll_rows(rows):
    k = len(rows)
    if k == 0:
        return rows
    d = len(rows[0])
    A = IntegerMatrix(k, d)
    for i in range(k):
        for j in range(d):
            A[i, j] = int(rows[i][j])
    LLL.reduction(A)
    out = []
    for i in range(k):
        r = [int(A[i, j]) for j in range(d)]
        if any(r):
            out.append(r)
    return out


def section_chain(B, G, ks, prec=256):
    d = len(B)
    coords = coordinates(B, G, prec=prec)
    cur = [list(c) for c in coords]
    want = sorted(set(ks), reverse=True)
    rank_full = len(cur)
    res = {}
    Bnp = np.array(B, dtype=object)
    for k in range(rank_full, 0, -1):
        s = d - rank_full + k
        if k in want:
            res[k] = np.array(cur, dtype=object) @ Bnp
        if k == 1:
            break
        col = [c[s - 1] for c in cur]
        ker = kernel_of_form(col)
        nxt = []
        for z in ker:
            v = [0] * d
            for i, zi in enumerate(z):
                if zi:
                    ci = cur[i]
                    for j in range(d):
                        v[j] += zi * ci[j]
            nxt.append(v)
        cur = _lll_rows(nxt)
        if len(cur) != k - 1:
            break
    return {k: [[int(x) for x in row] for row in v] for k, v in res.items()}


def lattice_stats(rows, beta=40, loops=8):
    k = len(rows)
    if k == 0:
        return None
    d = len(rows[0])
    A = IntegerMatrix(k, d)
    for i in range(k):
        for j in range(d):
            A[i, j] = int(rows[i][j])
    LLL.reduction(A)
    if k >= 4:
        for b in (20, min(beta, k)):
            try:
                BKZReduction(A)(bkz_param(min(b, k), max_loops=loops))
            except Exception:
                pass
    m = np.array([[float(A[i, j]) for j in range(d)] for i in range(k)])
    sc = np.max(np.abs(m))
    if sc == 0:
        return None
    sv = np.linalg.svd(m / sc, compute_uv=False)
    logvol = float(np.sum(np.log(sv))) + k * math.log(sc)
    lam = math.sqrt(min(sum(int(A[i, j]) ** 2 for j in range(d)) for i in range(k)))
    return {"k": k, "logvol": logvol, "lam1": lam}

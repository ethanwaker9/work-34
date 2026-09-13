import math
import time

import numpy as np
from fpylll import IntegerMatrix, LLL, BKZ
from fpylll.algorithms.bkz2 import BKZReduction

from .lattices import (descended_basis, module_basis, ntru_basis,
                       subring_basis)
from .reduction import bkz_param
from .ring import (embed_up, inverse_mod, negmul, negmul_mod,
                   relative_norm_mod, relative_trace_mod, sample_gaussian,
                   sample_ternary)
from .shape import family_params, gh_factor, lift_norms


class Instance(object):
    def __init__(self, n, q, sigma2, f, g, h):
        self.n = n
        self.q = q
        self.sigma2 = sigma2
        self.f = f
        self.g = g
        self.h = h

    def target(self):
        return gh_factor(2 * self.n) * math.sqrt(self.q)


def gen_instance(n, q, rng, dist="ternary", sigma=None):
    sigma2 = 2.0 / 3.0 if dist == "ternary" else sigma ** 2
    while True:
        if dist == "ternary":
            f = sample_ternary(n, rng)
            g = sample_ternary(n, rng)
        else:
            f = sample_gaussian(n, sigma, rng)
            g = sample_gaussian(n, sigma, rng)
        if all(x == 0 for x in f) or all(x == 0 for x in g):
            continue
        try:
            fi = inverse_mod(f, q)
        except Exception:
            continue
        h = negmul_mod(g, fi, q)
        if negmul_mod(f, h, q) != [x % q for x in g]:
            continue
        return Instance(n, q, sigma2, f, g, h)


def center(v, q):
    return [((x + q // 2) % q) - q // 2 for x in v]


def is_key_multiple(x, y, f, g):
    if all(v == 0 for v in x) and all(v == 0 for v in y):
        return False
    return negmul(x, g) == negmul(y, f)


def lift(xprime, inst, r):
    n, q = inst.n, inst.q
    x = embed_up(list(xprime), n)
    y = center(negmul_mod(x, inst.h, q), q)
    return x, y


def solution_norm(x, y):
    return math.sqrt(sum(int(a) ** 2 for a in x) + sum(int(b) ** 2 for b in y))


def _candidates(A, count=8):
    rows = []
    for i in range(A.nrows):
        v = [int(t) for t in A[i]]
        s = sum(x * x for x in v)
        if s:
            rows.append((s, v))
    rows.sort(key=lambda t: t[0])
    return [v for _, v in rows[:count]]


def _check_subfield(A, inst, r, nprime, ypos, ylen, best):
    tgt = inst.target()
    for v in _candidates(A):
        yp = v[ypos:ypos + ylen]
        if not any(yp):
            continue
        x, y = lift(yp, inst, r)
        nrm = solution_norm(x, y)
        if nrm > 0 and nrm < tgt:
            return True, (x, y, nrm)
    return False, best


def _module_from_yparts(ys, hp, q, nprime, thresh):
    gens = []
    for yp in ys:
        if not any(yp):
            continue
        zp = center(negmul_mod(yp, hp, q), q)
        nrm = math.sqrt(sum(t * t for t in yp) + sum(t * t for t in zp))
        if 0 < nrm < thresh:
            gens.append((yp, zp))
    if not gens:
        return []
    rows = []
    gens.sort(key=lambda t: sum(x * x for x in t[0]) + sum(y * y for y in t[1]))
    for xp, zp in gens[:2]:
        B = module_basis(xp, zp)
        for i in range(B.nrows):
            rows.append([int(B[i, j]) for j in range(B.ncols)])
    A = IntegerMatrix(len(rows), 2 * nprime)
    for i, rw in enumerate(rows):
        for j, val in enumerate(rw):
            A[i, j] = val
    LLL.reduction(A)
    keep = [[int(A[i, j]) for j in range(2 * nprime)] for i in range(A.nrows)
            if any(A[i, j] for j in range(2 * nprime))]
    if not keep:
        return []
    C = IntegerMatrix(len(keep), 2 * nprime)
    for i, rw in enumerate(keep):
        for j, val in enumerate(rw):
            C[i, j] = val
    if len(keep) >= 8:
        try:
            BKZReduction(C)(bkz_param(min(20, len(keep)), max_loops=2))
        except Exception:
            pass
    return [v[:nprime] for v in _candidates(C, count=6)]


def run_attack(kind, inst, params, beta_max=30, time_budget=120.0, tours=2):
    n, q = inst.n, inst.q
    t0 = time.time()
    r = int(params.get("r", 1))
    nprime = n // r
    family = params.get("family", "descend")
    ell = int(params.get("ell", 0))
    stats = {"kind": kind, "success": False, "beta": 0, "dim": 0, "time": 0.0,
             "norm": None, "key_multiple": False, "r": r, "ell": ell,
             "family": family, "logvol": 0.0, "postgain": 1.0}

    hp = relative_norm_mod(inst.h, r, q) if r > 1 else list(inst.h)
    fam = family_params(n, r, inst.sigma2)
    cov = fam["cov"]
    lift_ref = lift_norms(n, r, inst.sigma2)["reduced"] if r > 1 else 0.0

    if kind in ("full", "dvw21"):
        A = ntru_basis(inst.h, q)
        ypos, ylen = 0, n
    elif kind == "abd16":
        A = descended_basis(hp, q)
        ypos, ylen = 0, nprime
    elif kind == "cjl16":
        tp = relative_trace_mod(inst.h, r, q)
        A = descended_basis(tp, q)
        ypos, ylen = 0, nprime
    elif kind == "kf17":
        A = subring_basis(inst.h, q, r, ell)
        ypos, ylen = ell, nprime
    elif kind == "ours":
        if family == "subring" and ell > 0:
            A = subring_basis(inst.h, q, r, ell)
            ypos, ylen = ell, nprime
        else:
            A = descended_basis(hp, q)
            ypos, ylen = 0, nprime
    else:
        raise ValueError(kind)

    stats["dim"] = A.nrows
    tgt = inst.target()

    def try_lift(yparts):
        best = None
        for yp in yparts:
            if not any(yp):
                continue
            x, y = lift(yp, inst, r)
            nrm = solution_norm(x, y)
            if 0 < nrm < tgt and (best is None or nrm < best[2]):
                best = (x, y, nrm)
        return best

    def check(mat):
        cands = _candidates(mat, count=max(8, 2 * nprime))
        yparts = [v[ypos:ypos + ylen] for v in cands]
        if kind in ("full", "dvw21"):
            best = None
            for v in cands:
                nrm = math.sqrt(sum(t * t for t in v))
                if 0 < nrm < tgt and (best is None or nrm < best[2]):
                    best = (v[:n], v[n:2 * n], nrm)
            return best
        base = try_lift(yparts)
        if kind != "ours" or r == 1 or base is not None:
            return base
        red = _module_from_yparts(yparts, hp, q, nprime, q / cov)
        imp = try_lift(red)
        if imp is not None and (base is None or imp[2] < base[2]):
            if base is not None:
                stats["postgain"] = base[2] / imp[2]
            return imp
        return base

    def finish(beta, res):
        stats["beta"] = beta
        stats["time"] = time.time() - t0
        if res is not None:
            x, y, nrm = res
            stats["success"] = True
            stats["norm"] = nrm
            stats["key_multiple"] = is_key_multiple(x, y, inst.f, inst.g)
        return stats

    LLL.reduction(A)
    res = check(A)
    if res is not None:
        return finish(2, res)
    beta = 4
    while beta <= beta_max:
        bs = min(beta, A.nrows)
        try:
            BKZReduction(A)(bkz_param(bs, max_loops=tours))
        except Exception:
            LLL.reduction(A)
        res = check(A)
        if res is not None:
            return finish(bs, res)
        if time.time() - t0 > time_budget:
            return finish(bs, None)
        beta += 1 if beta < 12 else (2 if beta < 30 else 4)
    return finish(beta_max, None)

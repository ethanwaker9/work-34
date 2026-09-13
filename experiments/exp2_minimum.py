import csv
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from fpylll import BKZ, IntegerMatrix, LLL
from fpylll.algorithms.bkz2 import BKZReduction

from subfield.lattices import module_basis
from subfield.reduction import bkz_param
from subfield.ring import relative_norm, sample_ternary
from subfield.shape import covolume_density, delta_shape, gh_factor

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
CASES = [(256, 1), (256, 2), (256, 4), (256, 8), (256, 16),
         (512, 2), (512, 4), (512, 8), (512, 16),
         (1024, 4), (1024, 8), (1024, 16)]


def shortest(A, beta):
    LLL.reduction(A)
    for b in (20, 30, beta):
        b = min(b, A.nrows)
        if b < 5:
            continue
        try:
            BKZReduction(A)(bkz_param(b, max_loops=8))
        except Exception:
            pass
    return math.sqrt(min(sum(int(A[i, j]) ** 2 for j in range(A.ncols))
                         for i in range(A.nrows)))


def run(trials=12, seed=31415, beta=45):
    rng = np.random.default_rng(seed)
    rows = []
    for n, r in CASES:
        nprime = n // r
        if nprime < 16 or nprime > 256:
            continue
        obs, cov_e, sec_e = [], [], []
        for _ in range(trials):
            f = sample_ternary(n, rng)
            g = sample_ternary(n, rng)
            fp = relative_norm(f, r)
            gp = relative_norm(g, r)
            A = module_basis(fp, gp)
            M = np.array([[float(A[i, j]) for j in range(2 * nprime)]
                          for i in range(nprime)])
            sc = np.max(np.abs(M))
            sv = np.linalg.svd(M / sc, compute_uv=False)
            lcov = (float(np.sum(np.log(sv))) + nprime * math.log(sc)) / nprime
            l1 = shortest(A, beta)
            obs.append(math.log(l1) - lcov)
            cov_e.append(lcov)
            sec_e.append(0.5 * math.log(sum(x * x for x in fp) + sum(x * x for x in gp)) - lcov)
        s2 = 2.0 / 3.0
        d_pred = delta_shape(r, nprime)
        g_pred = gh_factor(nprime)
        rows.append({
            "n": n, "r": r, "nprime": nprime,
            "ratio_obs": math.exp(float(np.mean(obs))),
            "ratio_sd": float(np.std([math.exp(x) for x in obs])),
            "delta_obs": math.exp(float(np.mean(sec_e))),
            "delta_pred": d_pred, "gh_factor": g_pred,
            "ratio_pred": min(d_pred, g_pred),
            "log_cov_obs": float(np.mean(cov_e)),
            "log_cov_pred": math.log(covolume_density(n, r, s2)),
            "gain": math.exp(float(np.mean(sec_e)) - float(np.mean(obs))),
        })
        print(rows[-1], flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exp2_minimum.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    run()

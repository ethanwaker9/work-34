import csv
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from subfield.ring import log_embeddings, sample_ternary
from subfield.shape import (covolume_density, delta_shape, mu, mu_asymptotic,
                            nu, secret_descended_norm)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def coset_logs(a, n, r):
    nprime = n // r
    le = log_embeddings(a)
    j = np.arange(1, 2 * n, 2)
    lab = j % (2 * nprime)
    out = np.zeros(nprime)
    for t, l in enumerate(range(1, 2 * nprime, 2)):
        out[t] = le[lab == l].sum()
    return out


def run(trials=None, seed=2026, dist="ternary"):
    rng = np.random.default_rng(seed)
    rows = []
    from subfield.ring import sample_gaussian
    for n in [256, 512, 1024]:
        for r in [1, 2, 4, 8, 16]:
            nprime = n // r
            if nprime < 16:
                continue
            tr = trials if trials else (300 if nprime <= 64 else 80)
            lcov, lsec = [], []
            for _ in range(tr):
                if dist == "ternary":
                    f = sample_ternary(n, rng)
                    g = sample_ternary(n, rng)
                else:
                    f = sample_gaussian(n, math.sqrt(2.0 / 3.0), rng)
                    g = sample_gaussian(n, math.sqrt(2.0 / 3.0), rng)
                lf = coset_logs(f, n, r)
                lg = coset_logs(g, n, r)
                m = np.maximum(2 * lf, 2 * lg)
                lnW = m + np.log(np.exp(2 * lf - m) + np.exp(2 * lg - m))
                lcov.append(0.5 * float(lnW.mean()))
                mx = lnW.max()
                lsec.append(0.5 * (mx + math.log(float(np.exp(lnW - mx).mean()))))
            s2 = 2.0 / 3.0
            rows.append({
                "n": n, "r": r, "nprime": nprime,
                "log_cov_emp": float(np.mean(lcov)),
                "log_cov_sd": float(np.std(lcov)),
                "log_cov_pred": math.log(covolume_density(n, r, s2)),
                "log_sec_emp": float(np.mean(lsec)),
                "log_sec_pred": math.log(secret_descended_norm(n, r, s2)),
                "delta_emp": math.exp(float(np.mean(lsec)) - float(np.mean(lcov))),
                "delta_pred": delta_shape(r, nprime),
                "mu_r": mu(r), "mu_asym": mu_asymptotic(r), "trials": tr, "dist": dist,
            })
            print(rows[-1], flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exp1_shape.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    run()

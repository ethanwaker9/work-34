import csv
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from fpylll import LLL

from subfield.attacks import gen_instance
from subfield.lattices import ntru_basis, rot_rows
from subfield.predictor import lambda1_corrected, lambda1_gh
from subfield.sections import lattice_stats, section_chain

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def one(n, q, seed, ks, beta=45):
    rng = np.random.default_rng(seed)
    inst = gen_instance(n, q, rng)
    A = ntru_basis(inst.h, q)
    LLL.reduction(A)
    B = [[int(A[i, j]) for j in range(2 * n)] for i in range(2 * n)]
    Rf = rot_rows(inst.f)
    Rg = rot_rows(inst.g)
    G = [Rf[i] + Rg[i] for i in range(n)]
    lam_dense = math.sqrt(sum(x * x for x in inst.f) + sum(x * x for x in inst.g))
    sec = section_chain(B, G, ks)
    out = []
    for k in sorted(sec):
        st = lattice_stats(sec[k], beta=beta)
        if st is None or st["k"] != k:
            continue
        st["lam_dense"] = lam_dense
        out.append(st)
    logvol_dense = None
    if n in sec:
        s = lattice_stats(sec[n], beta=20)
        logvol_dense = s["logvol"]
    return out, lam_dense, logvol_dense


def run(n=64, qs=(97, 193), instances=6, seed0=808):
    ks = [k for k in range(2, n + 1, 2)]
    rows = []
    for q in qs:
        for t in range(instances):
            data, lam_dense, lvd = one(n, q, seed0 + 1000 * q + t, ks)
            V = math.exp(lvd / n)
            for st in data:
                k = st["k"]
                gh = lambda1_gh(k, st["logvol"])
                ours = lambda1_corrected(k, st["logvol"], n, V, lam_dense)
                rows.append({"n": n, "q": q, "inst": t, "k": k,
                             "logvol": st["logvol"], "lam1": st["lam1"],
                             "gh": gh, "ours": ours, "lam_dense": lam_dense,
                             "V": V, "u": st["lam1"] / (math.sqrt(n) * V)})
            print(f"n={n} q={q} inst={t} done ({len(data)} sections)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "exp3_sections.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    agg = {}
    for r in rows:
        agg.setdefault((r["q"], r["k"]), []).append(r)
    print(f"{'q':>5}{'k':>5}{'obs/gh':>9}{'ours/gh':>9}{'err_gh%':>9}{'err_our%':>9}")
    eg, eo = [], []
    for (q, k) in sorted(agg):
        v = agg[(q, k)]
        o = float(np.mean([x["lam1"] / x["gh"] for x in v]))
        p = float(np.mean([x["ours"] / x["gh"] for x in v]))
        e1 = float(np.mean([abs(x["gh"] / x["lam1"] - 1) for x in v]))
        e2 = float(np.mean([abs(x["ours"] / x["lam1"] - 1) for x in v]))
        eg.append(e1)
        eo.append(e2)
        if k % 8 == 0 or k >= n - 4:
            print(f"{q:5d}{k:5d}{o:9.3f}{p:9.3f}{100 * e1:9.1f}{100 * e2:9.1f}")
    print("mean rel err: GH %.1f%%  ours %.1f%%" % (100 * np.mean(eg), 100 * np.mean(eo)))
    return rows


if __name__ == "__main__":
    run()

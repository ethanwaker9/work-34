import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from _runner import run_one
from subfield.predictor import (select_abd16, select_cjl16, select_kf17,
                                select_ours, select_primal)

OUT = os.path.join(HERE, "..", "results")
SIG2 = 2.0 / 3.0

METHODS = [
    ("full", None),
    ("abd16", select_abd16),
    ("cjl16", select_cjl16),
    ("kf17", select_kf17),
    ("ours", select_ours),
]


def sweep(cases, seeds=(11, 23, 37), out_name="exp4_attacks.csv"):
    rows = []
    path = os.path.join(OUT, out_name)
    os.makedirs(OUT, exist_ok=True)
    for n, lq, budgets in cases:
        q = 2 ** lq
        for kind, sel in METHODS:
            if kind not in budgets:
                continue
            if sel is None:
                prm = {"r": 1, "ell": 0, "family": "full"}
                pred = select_primal(n, q, SIG2)
            else:
                c = sel(n, q, SIG2)
                if c is None:
                    rows.append({"n": n, "logq": lq, "kind": kind, "seed": 0,
                                 "success": False,
                                 "note": "no feasible parameter", "dim": 0, "r": 0,
                                 "ell": 0, "beta": 0, "time": 0.0, "peak_mem_mb": 0.0,
                                 "norm": "", "key_multiple": False,
                                 "pred_beta": 0, "pred_dim": 0, "postgain": 1.0})
                    print(rows[-1], flush=True)
                    continue
                prm = {"r": c["r"], "ell": c.get("ell", 0),
                       "family": c.get("family", "descend")}
                pred = c
            use = seeds if kind != "full" else seeds[:1]
            for sd in use:
                cfg = {"n": n, "logq": lq, "kind": kind, "params": prm, "seed": sd,
                       "beta_max": budgets.get("beta_max", 24),
                       "budget": budgets[kind], "mem_cap_gb": 6}
                res = run_one(cfg, hard_timeout=budgets[kind] * 2 + 600)
                res["note"] = "" if res.get("success") else "budget exceeded"
                res["pred_beta"] = pred["beta"] if pred else 0
                res["pred_dim"] = pred["dim"] if pred else 0
                res["seed"] = sd
                res.setdefault("postgain", 1.0)
                res.setdefault("norm", "")
                res.setdefault("key_multiple", False)
                rows.append({k: res.get(k) for k in
                             ["n", "logq", "kind", "seed", "success", "note", "dim",
                              "r", "ell", "beta", "time", "peak_mem_mb", "norm",
                              "key_multiple", "pred_beta", "pred_dim", "postgain"]})
                print(rows[-1], flush=True)
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    return rows


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "small"
    if which == "small":
        b = {"full": 900, "abd16": 300, "cjl16": 300, "kf17": 300, "ours": 300,
             "beta_max": 24}
        cases = [(128, lq, b) for lq in (24, 32, 40, 56, 80, 112)]
        sweep(cases, out_name="exp4_n128.csv")
    elif which == "medium":
        b = {"abd16": 600, "cjl16": 600, "kf17": 600, "ours": 600, "beta_max": 24}
        cases = [(256, lq, b) for lq in (28, 32, 40, 44, 56, 80, 112, 160)]
        sweep(cases, out_name="exp4_n256.csv")
    elif which == "large":
        b = {"abd16": 900, "cjl16": 900, "kf17": 900, "ours": 900, "beta_max": 20}
        cases = [(512, lq, b) for lq in (56, 64, 80, 120, 128)]
        sweep(cases, seeds=(11, 23), out_name="exp4_large.csv")
    elif which == "fullprobe":
        b = {"full": 600, "beta_max": 4}
        sweep([(256, 32, b)], seeds=(11,), out_name="exp4_fullprobe.csv")
    elif which == "fullfield":
        b = {"full": 7200, "beta_max": 4}
        sweep([(256, 40, b)], out_name="exp4_fullfield256.csv")


if __name__ == "__main__":
    main()

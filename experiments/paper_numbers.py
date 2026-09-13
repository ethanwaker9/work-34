import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np

RES = os.path.join(HERE, "..", "results")


def read(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return list(csv.DictReader(fh))


def shape_claims():
    rows = read("exp1_shape.csv")
    if not rows:
        return
    d = [abs(float(x["log_cov_emp"]) - float(x["log_cov_pred"])) for x in rows]
    e = [abs(float(x["delta_pred"]) / float(x["delta_emp"]) - 1) for x in rows]
    lo = min(float(x["log_cov_emp"]) for x in rows)
    hi = max(float(x["log_cov_emp"]) for x in rows)
    print("covolume law: max abs log error %.4f over %d configurations, range %.1f--%.1f"
          % (max(d), len(rows), lo, hi))
    print("shape defect: max relative error %.1f%%" % (100 * max(e)))


def minimum_claims():
    rows = read("exp2_minimum.csv")
    if not rows:
        return
    err = [100 * (float(x["ratio_obs"]) / float(x["ratio_pred"]) - 1) for x in rows]
    g = [float(x["gain"]) for x in rows]
    big = [e for e, x in zip(err, rows) if int(x["nprime"]) >= 32]
    print("lambda_1 model: relative error from %+.1f%% to %+.1f%% over %d configurations"
          % (min(err), max(err), len(rows)))
    print("  mean |error| %.1f%%; for n'>=32 (%d configs) from %+.1f%% to %+.1f%%"
          % (np.mean([abs(e) for e in err]), len(big), min(big), max(big)))
    i = int(np.argmax(g))
    print("module reduction gain: max %.2f at n=%s r=%s"
          % (max(g), rows[i]["n"], rows[i]["r"]))


def section_claims():
    rows = read("exp3_sections.csv")
    if not rows:
        return
    eg = [abs(float(x["gh"]) / float(x["lam1"]) - 1) for x in rows]
    eo = [abs(float(x["ours"]) / float(x["lam1"]) - 1) for x in rows]
    print("sections: mean error GH %.1f%%, corrected %.1f%%, over %d rows, %d instances"
          % (100 * np.mean(eg), 100 * np.mean(eo), len(rows),
             len({(x["q"], x["inst"]) for x in rows})))
    n = int(rows[0]["n"])
    grp = {}
    for x in rows:
        grp.setdefault((int(x["q"]), int(x["k"])), []).append(x)
    tail = [(q, k, v) for (q, k), v in grp.items() if k >= 0.85 * n]
    if tail:
        g = [100 * np.mean([abs(float(x["gh"]) / float(x["lam1"]) - 1) for x in v])
             for _, _, v in tail]
        o = [100 * np.mean([abs(float(x["ours"]) / float(x["lam1"]) - 1) for x in v])
             for _, _, v in tail]
        print("  averaged per k, for k >= 0.85n: GH %.0f--%.0f%%, corrected %.0f--%.0f%%"
              % (min(g), max(g), min(o), max(o)))
        ok = [k for (q, k), v in grp.items()
              if np.mean([abs(float(x["gh"]) / float(x["lam1"]) - 1) for x in v]) < 0.1]
        print("  Gaussian heuristic within 10%% up to k = %d (n = %d)" % (max(ok), n))
    at_n = [x for x in rows if int(x["k"]) == n]
    if at_n:
        print("  ratio at k=n: %.3f" % np.mean([float(x["lam1"]) / float(x["gh"])
                                                for x in at_n]))


def attack_claims():
    for name in ("exp4_n128.csv", "exp4_n256.csv", "exp4_large.csv"):
        rows = read(name)
        if not rows:
            continue
        print("== " + name)
        keys = sorted({(int(x["n"]), int(x["logq"])) for x in rows})
        for n, lq in keys:
            v = [x for x in rows if int(x["n"]) == n and int(x["logq"]) == lq]
            ours = [x for x in v if x["kind"] == "ours" and x["success"] in ("True", True)]
            if not ours:
                print("  n=%d lq=%d ours FAILED" % (n, lq))
                continue
            to = float(np.median([float(x["time"]) for x in ours]))
            mo = float(np.median([float(x["peak_mem_mb"]) for x in ours]))
            line = "  n=%d lq=%3d ours d=%s t=%.3f M=%.2f |" % (
                n, lq, ours[0]["dim"], to, mo)
            for kd in ("full", "abd16", "cjl16", "kf17"):
                w = [x for x in v if x["kind"] == kd and x["success"] in ("True", True)]
                if not w:
                    line += " %s:-" % kd
                    continue
                t = float(np.median([float(x["time"]) for x in w]))
                m = float(np.median([float(x["peak_mem_mb"]) for x in w]))
                line += " %s:%.1fx/%.1fx" % (kd, t / to, m / mo)
            ok = sum(1 for x in v if x["kind"] == "ours"
                     and x["success"] in ("True", True))
            tot = sum(1 for x in v if x["kind"] == "ours")
            print(line + "  (ours %d/%d)" % (ok, tot))


def lift_claims():
    from subfield.shape import gh_factor, lift_norms
    for ns in [(256, 512, 1024)]:
        best = 0.0
        arg = None
        for n in ns:
            for r in (2, 4, 8, 16, 32, 64):
                if n // r < 32:
                    continue
                d = lift_norms(n, r, 2.0 / 3.0)
                g = 2 * (d["log_plain"] - d["log_reduced"]) / math.log(2)
                if g > best:
                    best, arg = g, (n, r)
        print("module reduction lowers the usable log2 q by up to %.1f bits over %s (at %s)"
              % (best, ns, arg))


def predicted_claims():
    from subfield.predictor import select_dvw21, select_kf17, select_ours
    s2 = 2.0 / 3.0
    for n, lq in [(512, 64), (512, 120), (1024, 72), (1024, 136), (2048, 192)]:
        q = 2 ** lq
        out = []
        for nm, fn in (("full", select_dvw21), ("kf17", select_kf17),
                       ("ours", select_ours)):
            c = fn(n, q, s2)
            out.append("%s d=%s log2T=%s" % (
                nm, "-" if c is None else c["dim"],
                "-" if c is None else round(c["log2_time"], 1)))
        print("predicted n=%d lq=%d: %s" % (n, lq, " | ".join(out)))


def delta_claims():
    from subfield.shape import delta_shape
    print("Delta_{16,64} = %.1f" % delta_shape(16, 64))


if __name__ == "__main__":
    shape_claims()
    minimum_claims()
    section_claims()
    delta_claims()
    lift_claims()
    attack_claims()
    predicted_claims()

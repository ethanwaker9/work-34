import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import numpy as np

RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "..", "final_paper", "tables")
os.makedirs(OUT, exist_ok=True)

LAB = {"full": "Full-field reduction~\\cite{KF17,DvW21}",
       "abd16": "Norm descent~\\cite{ABD16}",
       "cjl16": "Trace descent~\\cite{CJL16}",
       "kf17": "Subring projection~\\cite{KF17}",
       "ours": "\\textbf{This work}"}
ORDER = ["full", "abd16", "cjl16", "kf17", "ours"]


def read(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return list(csv.DictReader(fh))


def fmt(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if x >= 1000:
        return "%.0f" % x
    return ("%%.%df" % d) % x


SHORT = {"full": "Full field~\\cite{KF17,DvW21}",
         "abd16": "Norm~\\cite{ABD16}",
         "cjl16": "Trace~\\cite{CJL16}",
         "kf17": "Subring~\\cite{KF17}",
         "ours": "\\textbf{This work}"}


def _cell(rows, lq, kd):
    v = [x for x in rows if int(x["logq"]) == lq and x["kind"] == kd]
    ok = [x for x in v if x["success"] in ("True", True)]
    if not v:
        return ("--", "--", "--")
    if not ok:
        return (">bud", "--", "--")
    t = float(np.median([float(x["time"]) for x in ok]))
    m = float(np.median([float(x["peak_mem_mb"]) for x in ok]))
    return (str(int(ok[0]["dim"])), fmt(t, 3), fmt(m, 2))


def _gain(rows, lq, kinds):
    ours = [x for x in rows if int(x["logq"]) == lq and x["kind"] == "ours"
            and x["success"] in ("True", True)]
    if not ours:
        return ("--", "--")
    to = float(np.median([float(x["time"]) for x in ours]))
    mo = float(np.median([float(x["peak_mem_mb"]) for x in ours]))
    bt, bm = None, None
    for kd in kinds:
        if kd == "ours":
            continue
        v = [x for x in rows if int(x["logq"]) == lq and x["kind"] == kd
             and x["success"] in ("True", True)]
        if not v:
            continue
        t = float(np.median([float(x["time"]) for x in v]))
        m = float(np.median([float(x["peak_mem_mb"]) for x in v]))
        bt = t if bt is None else min(bt, t)
        bm = m if bm is None else min(bm, m)
    return (fmt(bt / to, 1) if bt else "--", fmt(bm / mo, 1) if bm else "--")


def tab_attack_all(csvnames, out):
    all_rows = []
    for c in csvnames:
        all_rows += read(c)
    if not all_rows:
        return
    kinds = [k for k in ORDER if any(x["kind"] == k for x in all_rows)]
    ncol = 1 + 3 * len(kinds) + 2
    head = [r"\begin{tabular}{r%scc}" % ("ccc" * len(kinds)), r"\toprule",
            r"$\log_2 q$ & " + " & ".join(
                r"\multicolumn{3}{c}{%s}" % SHORT[k] for k in kinds)
            + r" & \multicolumn{2}{c}{gain} \\",
            " ".join(r"\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i)
                     for i in range(len(kinds)))
            + r"\cmidrule(lr){%d-%d}" % (2 + 3 * len(kinds), 3 + 3 * len(kinds)),
            " & " + " & ".join([r"$d$ & $t$ & $M$"] * len(kinds))
            + r" & $t$ & $M$ \\"]
    body = []
    keep = {128: (24, 56, 112)}
    for n in sorted({int(x["n"]) for x in all_rows}):
        sub = [x for x in all_rows if int(x["n"]) == n]
        if n in keep:
            sub = [x for x in sub if int(x["logq"]) in keep[n]]
        body.append(r"\midrule")
        body.append(r"\multicolumn{%d}{l}{\textit{ring degree} $n=%d$} \\"
                    % (ncol, n))
        for lq in sorted({int(x["logq"]) for x in sub}):
            cs = " & ".join("%s & %s & %s" % _cell(sub, lq, k) for k in kinds)
            body.append("%d & %s & %s & %s \\\\"
                        % ((lq, cs) + _gain(sub, lq, kinds)))
    tail = [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(OUT, out), "w") as fh:
        fh.write("\n".join(head + body + tail) + "\n")
    print(out, "rows", len(all_rows), "methods", kinds)


if __name__ == "__main__":
    tab_attack_all(["exp4_n128.csv", "exp4_n256.csv", "exp4_large.csv",
                    "exp4_fullprobe.csv"], "tab_attack_all.tex")

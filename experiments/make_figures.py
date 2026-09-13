import csv
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "lines.linewidth": 1.1, "lines.markersize": 3.5,
})

MARK = {"full": "o", "abd16": "s", "cjl16": "^", "kf17": "D", "ours": "*"}
COL = {"full": "#444444", "abd16": "#1f77b4", "cjl16": "#2ca02c",
       "kf17": "#ff7f0e", "ours": "#d62728"}
LAB = {"full": "Full-field reduction", "abd16": "Norm descent [ABD16]",
       "cjl16": "Trace descent [CJL16]", "kf17": "Subring projection [KF17]",
       "ours": "This work"}


def save(fig, name):
    eps = os.path.join(FIG, name + ".eps")
    pdf = os.path.join(FIG, name + ".pdf")
    fig.savefig(eps, format="eps")
    plt.close(fig)
    try:
        subprocess.run(["epstopdf", eps, "--outfile=" + pdf], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        fig.savefig(pdf, format="pdf")
    print("wrote", name)


def read(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return []
    with open(p) as fh:
        return list(csv.DictReader(fh))


def _agg(rows, key=("n", "logq", "kind")):
    out = {}
    for x in rows:
        k = tuple(x[t] for t in key)
        out.setdefault(k, []).append(x)
    return out


def _panel_covol(a, rows):
    ns = sorted({int(x["n"]) for x in rows})
    cols = {256: COL["abd16"], 512: COL["kf17"], 1024: COL["ours"]}
    for n in ns:
        sel = [x for x in rows if int(x["n"]) == n]
        rs = [int(x["r"]) for x in sel]
        a.plot(rs, [float(x["log_cov_emp"]) for x in sel], "o", ms=3,
               color=cols.get(n, "k"), label=r"$n=%d$" % n)
        a.plot(rs, [float(x["log_cov_pred"]) for x in sel], "-",
               color=cols.get(n, "k"))
    a.set_xscale("log", base=2)
    a.set_xlabel(r"$r$")
    a.set_title(r"$\frac{1}{n'}\ln\,\mathrm{covol}(\mathcal{D}')$", fontsize=6)
    a.legend(frameon=False, handlelength=1.1, fontsize=6)
    a.grid(alpha=0.3)


def _panel_lam1(a, r2):
    idx = np.arange(len(r2))
    a.plot(idx, [float(x["delta_pred"]) for x in r2], "-s", ms=3,
           color=COL["abd16"], label=r"$\Delta_{r,n'}$")
    a.plot(idx, [float(x["gh_factor"]) for x in r2], "-^", ms=3,
           color=COL["kf17"], label=r"$\gamma_{n'}$")
    a.plot(idx, [float(x["ratio_pred"]) for x in r2], "-", lw=1.6,
           color=COL["ours"], label="model (1)")
    a.plot(idx, [float(x["ratio_obs"]) for x in r2], "ko", ms=3,
           label="measured")
    a.set_xticks(idx)
    a.set_xticklabels([r"$%d,%s$" % (int(x["n"]), x["r"]) for x in r2],
                      rotation=90, fontsize=4.4)
    a.set_yscale("log")
    a.set_xlabel(r"$(n,r)$")
    a.set_title(r"ratio to $\mathrm{covol}^{1/n'}$", fontsize=6)
    a.set_ylim(top=90)
    a.legend(frameon=False, handlelength=1.0, fontsize=5.4,
             loc="upper center", ncol=2, columnspacing=0.6,
             borderpad=0.0, labelspacing=0.2)
    a.grid(alpha=0.3, which="both")


def _panel_section(a, sec, q, legend):
    sel = [x for x in sec if int(x["q"]) == q]
    ks = sorted({int(x["k"]) for x in sel if int(x["k"]) >= 6})
    obs = [np.mean([float(x["lam1"]) / float(x["gh"]) for x in sel
                    if int(x["k"]) == k]) for k in ks]
    our = [np.mean([float(x["ours"]) / float(x["gh"]) for x in sel
                    if int(x["k"]) == k]) for k in ks]
    n = int(sel[0]["n"])
    a.plot(ks, obs, "o", ms=3, color="k", label="measured")
    a.axhline(1.0, ls="--", color=COL["kf17"], label="Gauss. heur.")
    a.plot(ks, our, "-", color=COL["ours"], label="Thm. 11")
    a.axhline(math.sqrt(2 * math.pi * math.e / n), ls=":", color="gray",
              lw=0.9, label=r"$\sqrt{2\pi e/n}$")
    a.set_xlabel(r"section rank $k$")
    
    a.set_title(r"$\lambda_1/\mathrm{gh}$, $n=%d$, $q=%d$" % (n, q), fontsize=6)
    a.grid(alpha=0.3)
    if legend:
        a.legend(frameon=False, handlelength=1.0, fontsize=5.4,
                 loc="lower left", labelspacing=0.2, borderpad=0.0)


def _panel_cost(a0, a1, rows):
    agg = _agg(rows)
    for kd in ["full", "abd16", "cjl16", "kf17", "ours"]:
        xs, ts, ms = [], [], []
        for (n, lq, k), v in sorted(agg.items(), key=lambda t: int(t[0][1])):
            if k != kd:
                continue
            ok = [x for x in v if x["success"] in ("True", True)]
            if not ok:
                continue
            xs.append(int(lq))
            ts.append(float(np.median([float(x["time"]) for x in ok])))
            ms.append(float(np.median([float(x["peak_mem_mb"]) for x in ok])))
        if not xs:
            continue
        a0.plot(xs, ts, marker=MARK[kd], ms=3, color=COL[kd], label=LAB[kd])
        a1.plot(xs, ms, marker=MARK[kd], ms=3, color=COL[kd])
    for a, yl in zip([a0, a1], ["time (s)", "mem. (MB)"]):
        a.set_yscale("log")
        a.set_xlabel(r"$\log_2 q$")
        a.set_ylabel(yl)
        a.grid(alpha=0.3, which="both")
    a0.legend(frameon=False, fontsize=5.0, handlelength=0.9,
          labelspacing=0.2, loc="upper right", borderpad=0.0)


def _panel_bits(a):
    from subfield.shape import lift_norms, gh_factor
    for n, c in [(256, COL["abd16"]), (512, COL["kf17"]), (1024, COL["ours"])]:
        rs, a1, b1 = [], [], []
        for r in [2, 4, 8, 16, 32]:
            if n // r < 8:
                continue
            d = lift_norms(n, r, 2.0 / 3.0)
            base = math.log(0.8 * gh_factor(2 * n))
            rs.append(r)
            a1.append(2 * (d["log_plain"] - base) / math.log(2))
            b1.append(2 * (d["log_reduced"] - base) / math.log(2))
        a.plot(rs, a1, marker="s", ms=3, color=c, ls="--", label=r"$n=%d$" % n)
        a.plot(rs, b1, marker="o", ms=3, color=c, ls="-")
    a.set_xscale("log", base=2)
    a.set_xlabel(r"$r$")
    a.set_title(r"smallest usable $\log_2 q$", fontsize=6)
    a.grid(alpha=0.3)
    a.legend(frameon=False, fontsize=6, handlelength=1.1)


def _panel_surface(a3):
    from subfield.predictor import select_ours, select_kf17, select_dvw21
    s2 = 2.0 / 3.0
    ns = [128, 256, 512, 1024]
    lqs = list(range(32, 209, 16))
    Z = np.zeros((len(ns), len(lqs)))
    Zk = np.zeros_like(Z)
    for i, n in enumerate(ns):
        for j, lq in enumerate(lqs):
            q = 2 ** lq
            o = select_ours(n, q, s2)
            k = select_kf17(n, q, s2) or select_dvw21(n, q, s2)
            Z[i, j] = o["log2_time"] if o else np.nan
            Zk[i, j] = k["log2_time"] if k else np.nan
    X, Y = np.meshgrid(np.array(lqs, dtype=float),
                       np.log2(np.array(ns, dtype=float)))
    a3.plot_surface(X, Y, Zk, cmap=cm.Blues, linewidth=0.15, edgecolor="0.4",
                    rstride=1, cstride=1)
    a3.plot_surface(X, Y, Z, cmap=cm.Reds, linewidth=0.15, edgecolor="0.4",
                    rstride=1, cstride=1)
    a3.set_xlabel(r"$\log_2 q$", labelpad=-4, fontsize=6)
    a3.set_ylabel(r"$n$", labelpad=-5, fontsize=6)
    a3.set_zlabel(r"$\log_2 T$", labelpad=-5, fontsize=6)
    a3.set_yticks(np.log2(np.array(ns, dtype=float)))
    a3.set_yticklabels([str(x) for x in ns], fontsize=5)
    a3.tick_params(pad=-2, labelsize=5)
    a3.view_init(elev=20, azim=-127)


def fig_all(csvname):
    rows = read("exp1_shape.csv")
    r2 = read("exp2_minimum.csv")
    sec = read("exp3_sections.csv")
    fig = plt.figure(figsize=(7.1, 1.22), layout="constrained")
    ax = [fig.add_subplot(1, 5, i + 1) for i in range(4)]
    a3 = fig.add_subplot(1, 5, 5, projection="3d")
    _panel_covol(ax[0], rows)
    _panel_lam1(ax[1], r2)
    qs = sorted({int(x["q"]) for x in sec})[:2]
    _panel_section(ax[2], sec, qs[0], True)
    _panel_bits(ax[3])
    _panel_surface(a3)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.01, hspace=0.0,
                                wspace=0.02)
    save(fig, "fig_all")


if __name__ == "__main__":
    fig_all("exp4_n256.csv")

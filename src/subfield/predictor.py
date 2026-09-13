import math

import numpy as np

from .complexity import log2_total_memory, log2_total_time
from .excess import Xi_fast as Xi
from .profiles import delta_bkz, zgsa_profile
from .shape import family_params, gh_factor, lift_norms


def _digamma(x):
    r = 0.0
    while x < 6:
        r -= 1.0 / x
        x += 1
    f = 1.0 / (x * x)
    return r + math.log(x) - 0.5 / x + f * (-1.0 / 12 + f * (1.0 / 120 + f * (-1.0 / 252 + f / 240)))


_ZR = {}


def _zeta_ratio(l):
    if l in _ZR:
        return _ZR[l]
    if l >= 24:
        val = -math.log(2.0) * 2.0 ** (-float(l))
    else:
        m = np.arange(1, 4000, dtype=float)
        t = m ** (-float(l))
        val = -float(np.sum(np.log(m) * t) / np.sum(t))
    _ZR[l] = val
    return val


_LSUM = {}


def _lsum(dim, rank):
    key = (dim, rank)
    if key in _LSUM:
        return _LSUM[key]
    acc = np.zeros(rank + 2)
    run = 0.0
    for l in range(rank, 0, -1):
        acc[l] = run
        run += _digamma(l / 2.0) - _digamma((dim - rank + l) / 2.0) + _zeta_ratio(l)
    _LSUM[key] = acc
    return acc


_GHF = {}


def _ghf(k):
    if k not in _GHF:
        _GHF[k] = gh_factor(k)
    return _GHF[k]


def log_lambda1_gh(k, logvol):
    return math.log(_ghf(k)) + logvol / k


def log_lambda1_corrected(k, logvol, rank, log_density, log_lam_dense):
    lgh = log_lambda1_gh(k, logvol)
    llam = lgh
    for _ in range(14):
        c = math.exp(min(50.0, llam - 0.5 * math.log(rank) - log_density))
        new = lgh - (rank / float(k)) * Xi(rank, c)
        if abs(new - llam) < 1e-9:
            llam = new
            break
        llam = 0.5 * (llam + new)
    return max(llam, log_lam_dense)


_KG = {}


def _kgrid(rank):
    if rank in _KG:
        return _KG[rank]
    ks = set(range(1, min(rank, 24) + 1))
    x = 24.0
    while x < rank:
        x *= 1.12
        ks.add(min(rank, int(round(x))))
    ks.add(rank)
    _KG[rank] = sorted(ks)
    return _KG[rank]


USVP_MIN_BETA = 40
PT_FRACTION = 0.25


def _fires(beta, dim, nq, logq, rank, log_vol_dense, lam_dense, corrected, mode,
           lsum, dense_density, kstep=1, cert=None):
    prof = np.array(zgsa_profile(dim, logq, nq, beta))
    suf = np.concatenate([np.cumsum(prof[::-1])[::-1], [0.0]])
    if cert is not None:
        if dim * math.log(delta_bkz(beta)) + nq * logq / float(dim) <= cert:
            return "cert", 0
    if mode in ("both", "skr") and beta >= USVP_MIN_BETA:
        idx = dim - beta
        if idx >= 0 and 0.5 * math.log(beta / float(dim)) + lam_dense <= prof[idx]:
            return "skr", 0
    if mode in ("both", "dsd"):
        for k in _kgrid(rank):
            s = dim - rank + k
            idx = s - beta
            if idx < 0:
                continue
            bound = float(prof[idx])
            lfac = 0.5 * math.log(beta / float(s))
            if lfac + lam_dense > bound:
                continue
            lv = log_vol_dense - float(suf[s]) + float(lsum[k])
            lam_g = max(log_lambda1_gh(k, lv), lam_dense)
            if lfac + lam_g <= bound:
                return "dsd", k
            if not corrected:
                continue
            lam = log_lambda1_corrected(k, lv, rank, dense_density, lam_dense)
            if lfac + lam <= bound:
                return "dsd", k
    return None, 0


def predict_beta(dim, nq, logq, rank, log_vol_dense, lam_dense, corrected=True,
                 beta_max=None, mode="both", cert=None):
    if beta_max is None:
        beta_max = dim
    lsum = _lsum(dim, rank)
    dense_density = log_vol_dense / rank
    kstep = max(1, rank // 128)
    args = (dim, nq, logq, rank, log_vol_dense, lam_dense, corrected, mode,
            lsum, dense_density, kstep, cert)
    step = max(1, beta_max // 40)
    lo, hi = 2, None
    b = 2
    while b <= beta_max:
        ev, k = _fires(b, *args)
        if ev:
            hi = b
            break
        lo = b
        b += step
    if hi is None:
        return beta_max, "none", 0
    for beta in range(max(2, lo), hi + 1):
        ev, k = _fires(beta, *args)
        if ev:
            return beta, ev, k
    return hi, "dsd", 0


SAFETY = 0.8


def log_target_norm(n, q):
    return math.log(SAFETY * gh_factor(2 * n)) + 0.5 * math.log(q)


def target_norm(n, q):
    return math.exp(log_target_norm(n, q))


ZMARGIN = 2.5


def config(n, q, sigma2, r, ell=None):
    nprime = n // r
    fp = family_params(n, r, sigma2, ell)
    dim = 2 * nprime if ell is None else ell + nprime
    nq = nprime if ell is None else ell
    lcov = fp["log_cov"] + ZMARGIN * fp["log_cov_sd"]
    lsec = fp["log_secret"] + ZMARGIN * fp["log_cov_sd"]
    llam = math.log(min(fp["delta"], gh_factor(nprime))) + lcov
    cert = math.log(q) - lcov if ell is None else None
    return {"dim": dim, "nq": nq, "logq": math.log(q), "rank": nprime,
            "log_vol_dense": nprime * lcov, "log_lam_dense": llam,
            "log_secret": lsec, "log_cov": lcov,
            "delta": fp["delta"], "r": r, "ell": ell, "cert": cert}


def is_dense(cfg):
    amb = cfg["nq"] * cfg["logq"] / float(cfg["dim"])
    sub = cfg["log_vol_dense"] / float(cfg["rank"])
    slack = math.log(delta_bkz(2)) * (cfg["dim"] - cfg["rank"])
    if sub >= amb - PT_FRACTION * slack:
        return False
    if cfg.get("cert") is not None:
        return cfg["log_lam_dense"] < cfg["cert"]
    return True


def cost_of(cfg, corrected=True, mode="both", tours=8):
    if not is_dense(cfg):
        return {"beta": cfg["dim"], "event": "none", "k": 0,
                "log2_time": float("inf"), "log2_mem": float("inf"),
                "dim": cfg["dim"]}
    beta, ev, k = predict_beta(cfg["dim"], cfg["nq"], cfg["logq"], cfg["rank"],
                               cfg["log_vol_dense"], cfg["log_lam_dense"],
                               corrected=corrected, mode=mode,
                               cert=cfg.get("cert"))
    bits = max(1.0, cfg["logq"] / math.log(2))
    return {"beta": beta, "event": ev, "k": k,
            "log2_time": log2_total_time(cfg["dim"], beta, bits, tours=tours),
            "log2_mem": log2_total_memory(cfg["dim"], beta, bits),
            "dim": cfg["dim"]}


def divisors_pow2(n, nprime_min=8):
    out = []
    r = 1
    while n // r >= nprime_min:
        out.append(r)
        r *= 2
    return out


def hermite_beta(dim, log_gap, beta_max=400):
    for beta in range(2, beta_max + 1):
        if 2.0 * dim * math.log(delta_bkz(beta)) <= log_gap:
            return beta
    return None


def _cost(dim, beta, logq, tours=8):
    bits = max(1.0, logq / math.log(2))
    return {"beta": beta, "dim": dim,
            "log2_time": log2_total_time(dim, beta, bits, tours=tours),
            "log2_mem": log2_total_memory(dim, beta, bits)}


def _best(cands):
    best = None
    for c in cands:
        if c is None or c.get("event") == "none":
            continue
        key = (round(c["log2_time"], 6), round(c["log2_mem"], 6),
               -c.get("nq", 0) * c.get("logq", 0.0))
        if best is None or key < best[0]:
            best = (key, c)
    return None if best is None else best[1]


def select_abd16(n, q, sigma2, trace=False):
    ltau = log_target_norm(n, q)
    cands = []
    for r in divisors_pow2(n):
        if r == 1:
            continue
        nprime = n // r
        if lift_norms(n, r, sigma2)["log_plain"] > ltau:
            continue
        cfg = config(n, q, sigma2, r)
        if not is_dense(cfg):
            continue
        lsn = cfg["log_secret"] + (0.5 * math.log((1.0 + r) / 2.0) if trace else 0.0)
        gap = 0.5 * math.log(q) - lsn
        if gap <= 0:
            continue
        b = hermite_beta(2 * nprime, gap)
        if b is None:
            continue
        c = _cost(2 * nprime, b, math.log(q))
        c.update({"r": r, "ell": 0, "event": "skr", "k": 0, "family": "descend"})
        cands.append(c)
    return _best(cands)


def select_cjl16(n, q, sigma2):
    return select_abd16(n, q, sigma2, trace=True)


def ell_grid(n, r):
    nprime = n // r
    g = set([nprime // 4, nprime // 2, nprime, 2 * nprime, 4 * nprime,
             n // 4, n // 2, n])
    return sorted(x for x in g if x >= 8)


def select_kf17(n, q, sigma2):
    ltau = log_target_norm(n, q)
    cands = []
    for r in divisors_pow2(n):
        if r == 1:
            continue
        if lift_norms(n, r, sigma2)["log_plain"] > ltau:
            continue
        for ell in ell_grid(n, r):
            if not (4 <= ell <= n):
                continue
            cfg = config(n, q, sigma2, r, ell)
            cfg["log_vol_dense"] = cfg["rank"] * cfg["log_secret"]
            cfg["log_lam_dense"] = cfg["log_secret"]
            c = cost_of(cfg, corrected=False)
            c.update({"r": r, "ell": ell, "family": "subring",
                      "nq": cfg["nq"], "logq": cfg["logq"]})
            cands.append(c)
    return _best(cands)


def select_dvw21(n, q, sigma2):
    cfg = config(n, q, sigma2, 1)
    c = cost_of(cfg, corrected=False)
    c.update({"r": 1, "ell": 0, "family": "full"})
    return c if c["event"] != "none" else None


def select_primal(n, q, sigma2):
    cfg = config(n, q, sigma2, 1)
    cfg["log_lam_dense"] = 0.5 * math.log(2.0 * n * sigma2)
    c = cost_of(cfg, corrected=False, mode="skr")
    c.update({"r": 1, "ell": 0, "family": "full"})
    return c if c["event"] != "none" else None


def select_ours(n, q, sigma2):
    ltau = log_target_norm(n, q)
    cands = []
    for r in divisors_pow2(n):
        if r > 1 and lift_norms(n, r, sigma2)["log_reduced"] > ltau:
            continue
        cfg0 = config(n, q, sigma2, r)
        c = cost_of(cfg0, corrected=True)
        c.update({"r": r, "ell": 0, "family": "descend" if r > 1 else "full",
                  "nq": cfg0["nq"], "logq": cfg0["logq"]})
        cands.append(c)
        if r > 1:
            for ell in ell_grid(n, r):
                if 4 <= ell <= n:
                    cfg = config(n, q, sigma2, r, ell)
                    c = cost_of(cfg, corrected=True)
                    c.update({"r": r, "ell": ell, "family": "subring",
                              "nq": cfg["nq"], "logq": cfg["logq"]})
                    cands.append(c)
    return _best(cands)

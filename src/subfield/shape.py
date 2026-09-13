import math

import numpy as np

EULER_GAMMA = 0.5772156649015329


def _logistic_sum_density(r, half_width=300.0, m=1 << 17):
    x = np.linspace(-half_width, half_width, m, endpoint=False)
    dx = x[1] - x[0]
    t = 2 * np.pi * np.fft.fftfreq(m, d=dx)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        phi = np.where(np.abs(t) > 1e-12, np.pi * t / np.sinh(np.pi * t), 1.0)
    phi = np.nan_to_num(phi) ** r
    dens = np.real(np.fft.fftshift(np.fft.ifft(phi))) / dx
    xs = np.fft.fftshift(np.fft.fftfreq(m, d=(t[1] - t[0]) / (2 * np.pi)))
    return xs, dens


_MU_CACHE = {}


def mu(r):
    r = int(r)
    if r in _MU_CACHE:
        return _MU_CACHE[r]
    if r == 1:
        val = 1.0 - EULER_GAMMA
    else:
        xs, dens = _logistic_sum_density(r)
        g = np.log(2 * np.cosh(np.clip(xs, -700, 700) / 2))
        val = -r * EULER_GAMMA + float(np.sum(dens * g) * (xs[1] - xs[0]))
    _MU_CACHE[r] = val
    return val


def c_const(r):
    return math.exp(mu(r))


def mu_asymptotic(r):
    return -r * EULER_GAMMA + math.sqrt(math.pi * r / 6.0)


def log_covolume_density(n, r, sigma2):
    nprime = n // r
    return 0.5 * (r * math.log(n * sigma2) + mu(r))


def covolume_density(n, r, sigma2):
    return math.exp(log_covolume_density(n, r, sigma2))


_NU_CACHE = {}


def nu(r, nprime, samples=150000, seed=12345):
    key = (int(r), int(nprime))
    if key in _NU_CACHE:
        return _NU_CACHE[key]
    rng = np.random.default_rng(seed + 977 * int(r) + int(nprime))
    tot = 0.0
    done = 0
    batch = max(1, min(samples, 2000000 // max(1, nprime * r)))
    while done < samples:
        b = min(batch, samples - done)
        a = rng.exponential(size=(b, nprime, r))
        bb = rng.exponential(size=(b, nprime, r))
        la = np.log(a).sum(axis=2)
        lb = np.log(bb).sum(axis=2)
        m = np.maximum(la, lb)
        lnpsi = m + np.log(np.exp(la - m) + np.exp(lb - m))
        mm = lnpsi.max(axis=1, keepdims=True)
        lmean = (mm + np.log(np.exp(lnpsi - mm).mean(axis=1, keepdims=True))).ravel()
        tot += float(lmean.sum())
        done += b
    val = tot / done
    _NU_CACHE[key] = val
    return val


def delta_shape(r, nprime):
    return math.exp(0.5 * (nu(r, max(1, nprime // 2)) - mu(r)))


def delta_asymptotic(r):
    return math.sqrt(2.0 / c_const(r))


def gh_factor(k):
    if k <= 0:
        return 1.0
    return max(1.0, math.exp(-(0.5 * k * math.log(math.pi) - math.lgamma(k / 2.0 + 1.0)) / k))


def lambda1_dense(n, r, sigma2):
    nprime = n // r
    cov = covolume_density(n, r, sigma2)
    return min(delta_shape(r, nprime), gh_factor(nprime)) * cov


def secret_descended_norm(n, r, sigma2):
    nprime = n // r
    return math.exp(0.5 * (r * math.log(n * sigma2) + nu(r, max(1, nprime // 2))))


_FAM = {}


def _psi_logs(kind, r, size, rng, c=0.0):
    a = rng.exponential(size=(size, r))
    la = np.log(a).sum(axis=1)
    b = rng.exponential(size=(size, r))
    if kind == "norm":
        lb = np.log(b).sum(axis=1)
        m = np.maximum(la, lb)
        return m + np.log(np.exp(la - m) + np.exp(lb - m))
    return la + np.log1p(c * (b / a).sum(axis=1))


def family_stats(kind, r, m, c=0.0, samples=30000, seed=987):
    key = (kind, int(r), int(m), round(float(c), 10))
    if key in _FAM:
        return _FAM[key]
    if kind == "norm":
        mu_v = mu(r)
        nu_v = nu(r, m)
        _FAM[key] = (mu_v, nu_v, psi_variance(r))
        return _FAM[key]
    rng = np.random.default_rng(seed + 131 * int(r) + 7 * int(m) + int(1e6 * c))
    batch = max(1, min(samples, 2000000 // max(1, m * r)))
    tot_mu = 0.0
    tot_sq = 0.0
    tot_nu = 0.0
    done = 0
    while done < samples:
        bsz = min(batch, samples - done)
        lp = _psi_logs(kind, r, bsz * m, rng, c).reshape(bsz, m)
        tot_mu += float(lp.sum())
        tot_sq += float((lp ** 2).sum())
        mm = lp.max(axis=1, keepdims=True)
        tot_nu += float((mm + np.log(np.exp(lp - mm).mean(axis=1, keepdims=True))).sum())
        done += bsz
    mu_v = tot_mu / (done * m)
    var_v = max(0.0, tot_sq / (done * m) - mu_v * mu_v)
    _FAM[key] = (mu_v, tot_nu / done, var_v)
    return _FAM[key]


def family_params(n, r, sigma2, ell=None):
    nprime = n // r
    m = max(1, nprime // 2)
    if ell is None or ell == nprime and r == 1:
        pass
    if ell is None:
        kind, c = "norm", 0.0
    else:
        kind, c = "subring", float(ell) / float(r * n)
    mu_v, nu_v, var_v = family_stats(kind, r, m, c)
    base = r * math.log(n * sigma2)
    lcov = 0.5 * (base + mu_v)
    lsec = 0.5 * (base + nu_v)
    sd = 0.5 * math.sqrt(var_v / max(1.0, 2.0 * nprime))
    return {"log_cov": lcov, "log_secret": lsec, "cov": _safe_exp(lcov),
            "secret_norm": _safe_exp(lsec), "delta": math.exp(0.5 * (nu_v - mu_v)),
            "mu": mu_v, "nu": nu_v, "nprime": nprime, "log_cov_sd": sd}


_PSIVAR = {}


def psi_variance(r, samples=200000, seed=4242):
    r = int(r)
    if r in _PSIVAR:
        return _PSIVAR[r]
    rng = np.random.default_rng(seed + r)
    lp = _psi_logs("norm", r, samples, rng)
    _PSIVAR[r] = float(np.var(lp))
    return _PSIVAR[r]


_LIFT = {}


def lift_stats(r, m, samples=30000, seed=555):
    key = (int(r), int(m))
    if key in _LIFT:
        return _LIFT[key]
    rng = np.random.default_rng(seed + 313 * int(r) + int(m))
    batch = max(1, min(samples, 2000000 // max(1, m * r)))
    tot_mu = 0.0
    tot_nu = 0.0
    done = 0
    while done < samples:
        bsz = min(batch, samples - done)
        a = rng.exponential(size=(bsz * m, r))
        b = rng.exponential(size=(bsz * m, r))
        lp = (np.log(a).sum(axis=1) + np.log(r + (b / a).sum(axis=1))).reshape(bsz, m)
        tot_mu += float(lp.sum())
        mm = lp.max(axis=1, keepdims=True)
        tot_nu += float((mm + np.log(np.exp(lp - mm).mean(axis=1, keepdims=True))).sum())
        done += bsz
    _LIFT[key] = (tot_mu / (done * m), tot_nu / done)
    return _LIFT[key]


def _safe_exp(x):
    if x > 700:
        return float("inf")
    if x < -700:
        return 0.0
    return math.exp(x)


def lift_norms(n, r, sigma2):
    nprime = n // r
    if r == 1:
        lb = 0.5 * math.log(2.0 * n * sigma2)
        return {"log_plain": lb, "log_reduced": lb, "plain": _safe_exp(lb),
                "reduced": _safe_exp(lb), "delta": 1.0, "gain": 1.0}
    m = max(1, nprime // 2)
    mu_l, nu_l = lift_stats(r, m)
    marg = 2.5 * 0.5 * math.sqrt(psi_variance(r) / max(1.0, 2.0 * nprime))
    base = r * math.log(n * sigma2) - math.log(r) + 2.0 * marg
    lplain = 0.5 * (base + nu_l)
    dl = math.exp(0.5 * (nu_l - mu_l))
    lred = math.log(min(dl, gh_factor(nprime))) + 0.5 * (base + mu_l)
    return {"log_plain": lplain, "log_reduced": lred,
            "plain": _safe_exp(lplain), "reduced": _safe_exp(lred), "delta": dl,
            "gain": dl / min(dl, gh_factor(nprime))}

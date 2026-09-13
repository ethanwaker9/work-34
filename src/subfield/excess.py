import math

import numpy as np

_Z = np.arange(-60, 61, dtype=float)
_Z2 = _Z ** 2
_Z4 = _Z ** 4


def _tilt(t):
    w = np.exp(-math.pi * t * _Z2)
    z = float(w.sum())
    m1 = float((_Z2 * w).sum()) / z
    m2 = float((_Z4 * w).sum()) / z
    return z, m1, m2


_TG = None


def _tilt_table():
    global _TG
    if _TG is None:
        ts = np.exp(np.linspace(math.log(1e-6), math.log(80.0), 4000))
        w = np.exp(-math.pi * np.outer(ts, _Z2))
        z = w.sum(axis=1)
        m1 = (w * _Z2).sum(axis=1) / z
        m2 = (w * _Z4).sum(axis=1) / z
        _TG = (ts, np.log(z), m1, m2)
    return _TG


def _tilt_i(t):
    ts, lz, m1, m2 = _tilt_table()
    lt = math.log(min(max(t, ts[0]), ts[-1]))
    g = np.log(ts)
    return (float(np.interp(lt, g, lz)), float(np.interp(lt, g, m1)),
            float(np.interp(lt, g, m2)))


def _solve_t(d, x2):
    ts, lz, m1, m2 = _tilt_table()
    target = x2 / float(d)
    if target >= m1[0]:
        return ts[0]
    if target <= m1[-1]:
        return ts[-1]
    idx = np.searchsorted(-m1, -target)
    lo = max(0, idx - 1)
    hi = min(len(ts) - 1, idx)
    if m1[lo] == m1[hi]:
        return ts[lo]
    w = (m1[lo] - target) / (m1[lo] - m1[hi])
    return math.exp(math.log(ts[lo]) + w * (math.log(ts[hi]) - math.log(ts[lo])))


def log_count_hypercubic(d, x):
    if x <= 0:
        return -math.inf
    x2 = x * x
    t = _solve_t(d, x2)
    lzv, m1v, m2v = _tilt_i(t)
    z, m1, m2 = math.exp(lzv), m1v, m2v
    var = max(1e-300, d * (m2 - m1 * m1))
    s = math.sqrt(var)
    pref = s * math.sqrt(2 * math.pi) * (1.0 - math.exp(-math.pi * t))
    val = d * math.log(z) + math.pi * t * x2 - math.log(max(pref, 1e-300))
    return max(0.0, val)


def log_gh_count(d, c):
    return d * math.log(c * math.sqrt(2 * math.pi * math.e))


def Xi(d, c):
    if c <= 0:
        return math.inf
    x = c * math.sqrt(d)
    return max(0.0, (log_count_hypercubic(d, x) - log_gh_count(d, c)) / float(d))


def Xi_tail(c):
    return 2.0 * math.exp(-2.0 * math.pi ** 2 * c * c)


def Xi_small(c):
    return -math.log(c * math.sqrt(2 * math.pi * math.e))


def gh_validity_radius(d, k, eps=0.05):
    target = math.log(1.0 + eps) * k / float(d)
    lo, hi = 1e-3, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if Xi(d, mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


_XI_GRID = {}
_CGRID = np.concatenate([np.linspace(0.01, 0.30, 70, endpoint=False),
                         np.linspace(0.30, 0.90, 130)])


def _xi_table(d):
    if d not in _XI_GRID:
        _XI_GRID[d] = np.array([Xi(d, float(c)) for c in _CGRID])
    return _XI_GRID[d]


def Xi_fast(d, c):
    if c <= 0:
        return math.inf
    if c <= _CGRID[0]:
        return -math.log(c * math.sqrt(2 * math.pi * math.e))
    if c >= _CGRID[-1]:
        return 0.0
    return float(np.interp(c, _CGRID, _xi_table(d)))

import numpy as np


def _pack(a, bits):
    v = 0
    for c in reversed(a):
        v = (v << bits) + int(c)
    return v


def _unpack(v, bits, length):
    mask = (1 << bits) - 1
    half = 1 << (bits - 1)
    out = []
    for _ in range(length):
        d = v & mask
        if d >= half:
            d -= (1 << bits)
        out.append(d)
        v = (v - d) >> bits
    return out


def polymul(a, b):
    n, m = len(a), len(b)
    ma = max((abs(int(x)) for x in a), default=0)
    mb = max((abs(int(x)) for x in b), default=0)
    if ma == 0 or mb == 0:
        return [0] * (n + m - 1)
    bound = ma * mb * min(n, m)
    bits = bound.bit_length() + 2
    prod = _pack(a, bits) * _pack(b, bits)
    return _unpack(prod, bits, n + m - 1)


def negmul(a, b):
    n = len(a)
    c = polymul(a, b)
    out = [0] * n
    for k, ck in enumerate(c):
        if k < n:
            out[k] += ck
        else:
            out[k - n] -= ck
    return out


def negmul_mod(a, b, q):
    return [x % q for x in negmul(a, b)]


def conj_alt(a):
    return [x if i % 2 == 0 else -x for i, x in enumerate(a)]


def norm_step(a):
    n = len(a)
    p = negmul(a, conj_alt(a))
    return [p[2 * i] for i in range(n // 2)]


def norm_step_mod(a, q):
    n = len(a)
    p = negmul_mod(a, conj_alt(a), q)
    return [p[2 * i] % q for i in range(n // 2)]


def relative_norm(a, r):
    x = list(a)
    while r > 1:
        x = norm_step(x)
        r //= 2
    return x


def relative_norm_mod(a, r, q):
    x = [int(v) % q for v in a]
    while r > 1:
        x = norm_step_mod(x, q)
        r //= 2
    return x


def relative_trace(a, r):
    x = list(a)
    while r > 1:
        n = len(x)
        x = [2 * x[2 * i] for i in range(n // 2)]
        r //= 2
    return x


def relative_trace_mod(a, r, q):
    x = [int(v) % q for v in a]
    while r > 1:
        n = len(x)
        x = [(2 * x[2 * i]) % q for i in range(n // 2)]
        r //= 2
    return x


def norm_cofactor(a, r):
    n = len(a)
    F = [1] + [0] * (n - 1)
    cur = list(a)
    k = r
    while k > 1:
        cbar = conj_alt(cur)
        F = negmul(F, embed_up(cbar, n))
        p = negmul(cur, cbar)
        cur = [p[2 * i] for i in range(len(cur) // 2)]
        k //= 2
    return F


def embed_up(a, n):
    m = len(a)
    r = n // m
    out = [0] * n
    for i, c in enumerate(a):
        out[i * r] = c
    return out


def negacyclic_matrix(a):
    n = len(a)
    M = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            k = j - i
            M[i, j] = a[k] if k >= 0 else -a[k + n]
    return M


def inverse_mod(a, q):
    n = len(a)
    mod = [1] + [0] * (n - 1) + [1]
    r0, r1 = mod, list(a) + [0]
    while len(r1) > 1 and r1[-1] == 0:
        r1.pop()
    s0, s1 = [0], [1]

    def deg(p):
        d = len(p) - 1
        while d > 0 and p[d] % q == 0:
            d -= 1
        return d

    def inv(x):
        return pow(int(x) % q, q - 2, q)

    r0 = [x % q for x in r0]
    r1 = [x % q for x in r1]
    while deg(r1) > 0 or r1[0] % q != 0:
        if deg(r1) == 0:
            c = inv(r1[0])
            res = [(c * x) % q for x in s1]
            res = res + [0] * (n - len(res))
            return res[:n]
        d0, d1 = deg(r0), deg(r1)
        if d0 < d1:
            r0, r1 = r1, r0
            s0, s1 = s1, s0
            d0, d1 = d1, d0
        c = (r0[d0] * inv(r1[d1])) % q
        shift = d0 - d1
        newr = list(r0)
        for i in range(d1 + 1):
            newr[i + shift] = (newr[i + shift] - c * r1[i]) % q
        news = list(s0) + [0] * max(0, len(s1) + shift - len(s0))
        s1p = list(s1) + [0] * shift
        for i in range(len(s1)):
            while len(news) <= i + shift:
                news.append(0)
            news[i + shift] = (news[i + shift] - c * s1[i]) % q
        r0, s0 = r1, s1
        r1, s1 = newr, news
    raise ValueError("not invertible")


def sample_ternary(n, rng):
    return [int(x) for x in rng.integers(-1, 2, n)]


def sample_gaussian(n, sigma, rng):
    return [int(round(x)) for x in rng.normal(0, sigma, n)]


_EMB = {}


def embedding_matrix(n):
    if n not in _EMB:
        j = np.arange(1, 2 * n, 2)
        k = np.arange(n)
        _EMB[n] = np.exp(1j * np.pi * np.outer(j, k) / n)
    return _EMB[n]


def embeddings(a):
    n = len(a)
    return embedding_matrix(n) @ np.array([float(x) for x in a])


def log_embeddings(a):
    n = len(a)
    v = np.array([float(x) for x in a])
    scale = max(1.0, np.max(np.abs(v)))
    e = embeddings((v / scale).tolist())
    return np.log(np.abs(e)) + np.log(scale)

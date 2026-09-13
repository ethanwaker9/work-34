import math

_DELTA_SMALL = {
    2: 1.021900, 3: 1.020807, 4: 1.019713, 5: 1.018620, 6: 1.018128,
    7: 1.017636, 8: 1.017144, 9: 1.016652, 10: 1.016160, 11: 1.015898,
    12: 1.015636, 13: 1.015374, 14: 1.015112, 15: 1.014850, 16: 1.014720,
    17: 1.014590, 18: 1.014460, 19: 1.014330, 20: 1.014200, 21: 1.014044,
    22: 1.013888, 23: 1.013732, 24: 1.013576, 25: 1.013420, 26: 1.013383,
    27: 1.013347, 28: 1.013310, 29: 1.013273, 30: 1.013236, 31: 1.013198,
    32: 1.013161, 33: 1.013124, 34: 1.013087, 35: 1.013050, 36: 1.013013,
    37: 1.012976, 38: 1.012939, 39: 1.012902, 40: 1.012865, 41: 1.012828,
    42: 1.012791, 43: 1.012754, 44: 1.012717, 45: 1.012680, 46: 1.012643,
    47: 1.012606, 48: 1.012569, 49: 1.012532, 50: 1.012495,
}


def delta_bkz(beta):
    b = int(round(beta))
    if b <= 2:
        return _DELTA_SMALL[2]
    if b <= 50:
        return _DELTA_SMALL[b]
    x = float(b)
    return ((math.pi * x) ** (1.0 / x) * x / (2 * math.pi * math.e)) ** (1.0 / (2.0 * (x - 1.0)))


def gsa_profile(dim, log_volume, beta):
    ld = math.log(delta_bkz(beta))
    base = log_volume / dim
    return [base + (dim - 1 - 2 * i) * ld for i in range(dim)]


def zgsa_profile(dim, logq, nq, beta):
    ld = math.log(delta_bkz(beta))
    m = logq / (2.0 * ld)
    a = nq - m / 2.0
    c = dim - a - m
    if a < 0 or c < 0 or m > dim:
        return gsa_profile(dim, nq * logq, beta)
    a_i = int(round(a))
    m_i = max(1, int(round(m)))
    prof = []
    for i in range(dim):
        if i < a_i:
            prof.append(logq)
        elif i < a_i + m_i:
            prof.append(logq - (i - a_i + 0.5) * 2.0 * ld)
        else:
            prof.append(0.0)
    shift = (nq * logq - sum(prof)) / dim
    return [p + shift for p in prof]

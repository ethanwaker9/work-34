import math

SIEVE_TIME_C = 0.292
SIEVE_MEM_C = 0.2075
SIEVE_OVERHEAD = 16.4


def log2_svp_time(beta):
    if beta < 2:
        beta = 2
    return SIEVE_TIME_C * beta + SIEVE_OVERHEAD


def log2_svp_memory(beta):
    if beta < 2:
        beta = 2
    return SIEVE_MEM_C * beta + SIEVE_OVERHEAD


def log2_bkz_time(dim, beta, tours=8):
    if beta <= 2:
        return log2_lll_time(dim, 64)
    return log2_svp_time(beta) + math.log2(max(1.0, dim * tours))


def log2_lll_time(dim, logq_bits):
    return math.log2(max(1.0, dim ** 4 * logq_bits))


def log2_basis_memory(dim, logq_bits):
    return math.log2(max(1.0, dim * dim * logq_bits))


def log2_total_memory(dim, beta, logq_bits):
    a = log2_basis_memory(dim, logq_bits)
    if beta <= 2:
        return a
    b = log2_svp_memory(beta) + math.log2(max(1.0, beta * 64.0))
    return max(a, b) + math.log2(1 + 2 ** (min(a, b) - max(a, b)))


def log2_total_time(dim, beta, logq_bits, tours=8):
    a = log2_lll_time(dim, logq_bits)
    if beta <= 2:
        return a
    b = log2_bkz_time(dim, beta, tours)
    return max(a, b) + math.log2(1 + 2 ** (min(a, b) - max(a, b)))

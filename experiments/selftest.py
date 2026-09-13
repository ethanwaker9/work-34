import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import numpy as np
from sympy import nextprime

from subfield.attacks import gen_instance, run_attack
from subfield.predictor import (select_abd16, select_cjl16, select_kf17,
                                select_ours, select_primal)
from subfield.ring import (inverse_mod, negmul_mod, norm_cofactor, negmul,
                           relative_norm, relative_norm_mod, relative_trace,
                           relative_trace_mod, sample_ternary, embed_up)
from subfield.shape import family_params, lift_norms, mu


def main():
    rng = np.random.default_rng(0)
    n, q, r = 64, int(nextprime(2 ** 20)), 4
    f = sample_ternary(n, rng)
    g = sample_ternary(n, rng)
    h = negmul_mod(g, inverse_mod(f, q), q)
    assert negmul_mod(f, h, q) == [x % q for x in g]
    assert negmul_mod(relative_norm(f, r), relative_norm_mod(h, r, q), q) == \
        [x % q for x in relative_norm(g, r)]
    F = norm_cofactor(f, r)
    assert negmul(f, F) == embed_up(relative_norm(f, r), n)
    A = relative_trace(negmul(g, F), r)
    assert [x % q for x in A] == negmul_mod(relative_norm(f, r),
                                            relative_trace_mod(h, r, q), q)
    assert abs(mu(1) - (1 - 0.5772156649015329)) < 1e-6
    for nn, rr in ((256, 8), (512, 16)):
        d = family_params(nn, rr, 2.0 / 3.0)
        assert d["delta"] >= 1.0 and d["log_cov_sd"] > 0
        assert lift_norms(nn, rr, 2.0 / 3.0)["log_reduced"] <= \
            lift_norms(nn, rr, 2.0 / 3.0)["log_plain"] + 1e-9
    s2 = 2.0 / 3.0
    for fn in (select_primal, select_abd16, select_cjl16, select_kf17, select_ours):
        c = fn(256, 2 ** 80, s2)
        assert c is None or (c["dim"] > 0 and c["beta"] >= 2)
    inst = gen_instance(128, int(nextprime(2 ** 56)), np.random.default_rng(1))
    for kind, prm in (("ours", {"r": 8, "ell": 0, "family": "descend"}),
                      ("abd16", {"r": 8}), ("cjl16", {"r": 8}),
                      ("kf17", {"r": 8, "ell": 16})):
        st = run_attack(kind, inst, prm, beta_max=12, time_budget=60)
        assert st["success"] and st["key_multiple"], (kind, st)
    print("self test passed")


if __name__ == "__main__":
    main()

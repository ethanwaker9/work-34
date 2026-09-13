import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from subfield.excess import Xi, Xi_small, Xi_tail, gh_validity_radius
from subfield.shape import (c_const, delta_asymptotic, delta_shape, gh_factor,
                            lift_norms, mu, mu_asymptotic)


def main():
    print("constants of Theorem 5 and Corollary 7")
    print("%4s %12s %12s %12s %12s" % ("r", "mu_r", "asymptotic", "c_r", "Delta_inf"))
    for r in (1, 2, 4, 8, 16, 32):
        print("%4d %12.6f %12.6f %12.6g %12.4f"
              % (r, mu(r), mu_asymptotic(r), c_const(r), delta_asymptotic(r)))
    print()
    print("shape defect and the constant of the Gaussian heuristic")
    print("%4s %6s %10s %10s %10s" % ("r", "n'", "Delta", "gamma", "min"))
    for r, npr in ((4, 64), (8, 32), (8, 64), (16, 32), (16, 64)):
        d, g = delta_shape(r, npr), gh_factor(npr)
        print("%4d %6d %10.4f %10.4f %10.4f" % (r, npr, d, g, min(d, g)))
    print()
    print("excess exponent of Theorem 11 and its two limits")
    print("%6s %12s %12s %12s" % ("c", "Xi", "small-c", "large-c"))
    for c in (0.05, 0.1, 0.2, 0.3, 0.4, 0.6):
        print("%6.2f %12.6f %12.6f %12.6f"
              % (c, Xi(128, c), Xi_small(c), Xi_tail(c)))
    print()
    print("radius above which the Gaussian heuristic is within 5 percent, m = 128")
    print("  " + "  ".join("k=%d: %.3f" % (k, gh_validity_radius(128, k, 0.05))
                           for k in (1, 4, 16, 64, 128)))
    print()
    print("smallest usable modulus, in bits, with and without the module reduction")
    print("%6s %4s %12s %12s %8s" % ("n", "r", "plain", "reduced", "gain"))
    for n in (256, 512, 1024):
        for r in (4, 8, 16, 32):
            if n // r < 8:
                continue
            d = lift_norms(n, r, 2.0 / 3.0)
            base = math.log(0.8 * gh_factor(2 * n))
            a = 2 * (d["log_plain"] - base) / math.log(2)
            b = 2 * (d["log_reduced"] - base) / math.log(2)
            print("%6d %4d %12.1f %12.1f %8.1f" % (n, r, a, b, a - b))


if __name__ == "__main__":
    main()

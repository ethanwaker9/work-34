import json
import os
import resource
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from sympy import nextprime

from subfield.attacks import gen_instance, run_attack
from subfield.shape import (covolume_density, delta_shape, gh_factor,
                            secret_descended_norm)


def warm(n, r, sigma2):
    from subfield.shape import family_params, lift_norms
    covolume_density(n, r, sigma2)
    secret_descended_norm(n, r, sigma2)
    family_params(n, r, sigma2)
    if r > 1:
        lift_norms(n, r, sigma2)
    delta_shape(r, max(1, (n // r) // 2))
    gh_factor(n // r)


def main():
    cfg = json.loads(sys.argv[1])
    mem_cap = int(cfg.get("mem_cap_gb", 3) * (1 << 30))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (mem_cap, mem_cap))
    except Exception:
        pass
    rng = np.random.default_rng(cfg["seed"])
    q = int(nextprime(2 ** cfg["logq"]))
    inst = gen_instance(cfg["n"], q, rng)
    warm(cfg["n"], int(cfg["params"].get("r", 1)), inst.sigma2)
    import gc
    gc.collect()
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    st = run_attack(cfg["kind"], inst, cfg["params"],
                    beta_max=cfg.get("beta_max", 30),
                    time_budget=cfg.get("budget", 60.0),
                    tours=cfg.get("tours", 2))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    unit = 1.0 if sys.platform == "darwin" else 1024.0
    st["peak_mem_mb"] = max(0.0, (peak - base) * unit / (1 << 20))
    st["base_mem_mb"] = base * unit / (1 << 20)
    st["logq"] = cfg["logq"]
    st["n"] = cfg["n"]
    st["seed"] = cfg["seed"]
    print("RESULT " + json.dumps(st))


if __name__ == "__main__":
    main()

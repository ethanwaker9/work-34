# Efficient Subfield Attacks on Overstretched NTRU

The repository is implementations of our research and contains a precise evaluation of the shape law of the descended dense sublattice of an NTRU instance, an evaluation of the excess factor of a lattice and of the corrected first minimum of a section of a dense sublattice, a cost model for five attacks on overstretched NTRU, and runnable implementations of those five attacks.

## Files and Contents

```
src/subfield/ring.py        negacyclic arithmetic, relative norm and trace towers
src/subfield/lattices.py    NTRU, descended and subring bases, dense sublattices
src/subfield/shape.py       shape constants mu_r, c_r, Delta_{r,n'}, lift norms
src/subfield/excess.py      excess exponent Xi by saddle point evaluation
src/subfield/profiles.py    root Hermite factors and the q-ary basis profile
src/subfield/complexity.py  time and memory of block reduction
src/subfield/predictor.py   the five cost models and the parameter selection
src/subfield/sections.py    precise sections of a dense sublattice
src/subfield/reduction.py   thin wrappers around fplll
src/subfield/attacks.py     the five attacks and the success test
results/                    csv output
```

## Running
Python 3.10 or later is required, together with `fplll` and its Python bindings.  The block reduction calls need the strategy file shipped with `fplll`. `src/subfield/reduction.py` looks for it in the usual installation prefixes; if it is elsewhere, set the environment variable `FPLLL_STRATEGIES` to its path.
```
python run_all.py
```

```
python run_all.py exp1_shape
python run_all.py exp4_attacks
```

```
python experiments/formulas.py       
python experiments/paper_numbers.py  
```

Each attack runs in its own process with a memory cap and a time budget, so a
single run cannot exhaust the machine. The budgets and the parameter grids are
at the bottom of `experiments/exp4_attacks.py`.

```python
import numpy as np
from sympy import nextprime
from subfield.attacks import gen_instance, run_attack
from subfield.predictor import select_ours

n, logq = 256, 56
q = int(nextprime(2 ** logq))
inst = gen_instance(n, q, np.random.default_rng(0))
cfg = select_ours(n, q, inst.sigma2)
params = {"r": cfg["r"], "ell": cfg.get("ell", 0), "family": cfg["family"]}
print(run_attack("ours", inst, params, beta_max=20, time_budget=300))
```

`run_attack` accepts `"full"`, `"abd16"`, `"cjl16"`, `"kf17"` and `"ours"`, so
the same call reproduces any row of the comparison. It returns the reduction
dimension, the block size reached, the wall-clock time, the peak memory of the
reduction, the length of the recovered vector, and a flag saying whether that
vector is an exact multiple of the secret key over the integers.

On one core of an Apple M2 the theory experiments take a few minutes each. The
attack sweep at `n = 128` takes about ten minutes, the one at `n = 256` about
two hours and the one at `n = 512` about two hours, the cost being dominated by
the full field baseline and by the two descent attacks that select a large
reduction dimension. The full field baseline at `n = 256` is a single run that
stops at its ten minute budget.

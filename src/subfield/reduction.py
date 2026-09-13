import glob
import os

from fpylll import BKZ


def _find_strategies():
    env = os.environ.get("FPLLL_STRATEGIES")
    if env and os.path.exists(env):
        return env
    cands = []
    cands += glob.glob("/opt/homebrew/Cellar/fplll/*/share/fplll/strategies/default.json")
    cands += glob.glob("/usr/local/share/fplll/strategies/default.json")
    cands += glob.glob("/usr/share/fplll/strategies/default.json")
    cands += glob.glob(os.path.expanduser("~/*/share/fplll/strategies/default.json"))
    cands += glob.glob("/Applications/SageMath*/**/share/fplll/strategies/default.json", recursive=True)
    for c in cands:
        if os.path.exists(c):
            return c
    return None


STRATEGIES = _find_strategies()


def bkz_param(block_size, max_loops=1):
    if STRATEGIES is not None:
        return BKZ.Param(block_size=block_size, max_loops=max_loops, strategies=STRATEGIES)
    return BKZ.Param(block_size=block_size, max_loops=max_loops,
                     strategies=BKZ.Param(block_size=block_size).strategies)



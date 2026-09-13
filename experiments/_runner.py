import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(cfg, hard_timeout=None):
    if hard_timeout is None:
        hard_timeout = cfg.get("budget", 60.0) * 3 + 120
    try:
        p = subprocess.run([sys.executable, os.path.join(HERE, "_worker.py"), json.dumps(cfg)],
                           capture_output=True, text=True, timeout=hard_timeout)
    except subprocess.TimeoutExpired:
        return {"kind": cfg["kind"], "n": cfg["n"], "logq": cfg["logq"], "success": False,
                "timeout": True, "time": hard_timeout, "peak_mem_mb": float("nan"),
                "r": cfg["params"].get("r", 1), "dim": 0, "beta": 0}
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    return {"kind": cfg["kind"], "n": cfg["n"], "logq": cfg["logq"], "success": False,
            "error": (p.stderr or "")[-400:], "time": 0.0, "peak_mem_mb": float("nan"),
            "r": cfg["params"].get("r", 1), "dim": 0, "beta": 0}

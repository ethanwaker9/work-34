import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")

STAGES = [
    ("self test", ["selftest.py"]),
    ("shape law (Theorem 5)", ["exp1_shape.py"]),
    ("minimum of the descended sublattice (Eq. 1)", ["exp2_minimum.py"]),
    ("Gaussian heuristic on sections (Theorem 11)", ["exp3_sections.py"]),
    ("attack comparison, n = 128", ["exp4_attacks.py", "small"]),
    ("attack comparison, n = 256", ["exp4_attacks.py", "medium"]),
    ("attack comparison, n = 512", ["exp4_attacks.py", "large"]),
    ("full-field baseline at n = 256", ["exp4_attacks.py", "fullprobe"]),
    ("tables", ["make_tables.py"]),
    ("figures", ["make_figures.py"]),
    ("numerical claims of the paper", ["paper_numbers.py"]),
]


def main():
    only = set(sys.argv[1:])
    for i, (name, cmd) in enumerate(STAGES):
        tag = cmd[0].split(".")[0]
        if only and tag not in only and str(i) not in only:
            continue
        print("=" * 70)
        print("stage %d: %s" % (i, name))
        print("=" * 70, flush=True)
        subprocess.run([sys.executable, "-u", os.path.join(EXP, cmd[0])] + cmd[1:],
                       check=False)


if __name__ == "__main__":
    main()

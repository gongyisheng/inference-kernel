"""Discover and run every benchmarks/<category>/bench_*.py script.

Each is invoked as a module so it can import benchmarks._harness etc.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = ROOT / "benchmarks"


def discover() -> list[str]:
    """Return module dotted-paths for every bench_*.py."""
    modules: list[str] = []
    for p in sorted(BENCH_ROOT.glob("*/bench_*.py")):
        rel = p.relative_to(ROOT).with_suffix("")
        modules.append(str(rel).replace("/", "."))
    return modules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    for mod in discover():
        print(f"\n=== {mod} ===")
        rc = subprocess.call(
            [sys.executable, "-m", mod, "--device", args.device],
            cwd=str(ROOT),
        )
        if rc != 0:
            print(f"!! {mod} exited with code {rc}")


if __name__ == "__main__":
    main()

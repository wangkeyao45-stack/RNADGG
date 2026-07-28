"""Run the RNADGG smoke test from a source checkout."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rnadgg.smoke import run_smoke_test


if __name__ == "__main__":
    run_smoke_test()

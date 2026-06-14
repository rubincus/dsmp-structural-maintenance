"""Convenience entry point for running the DSMP experiments."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dsmp.run_all_experiments import main

if __name__ == "__main__":
    main()

"""Convenience entry point for the expanded DSMP validation protocol."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dsmp.validation_experiments import main


if __name__ == "__main__":
    main()


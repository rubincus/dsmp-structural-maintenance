"""Run the revision-stage supplementary experiments (weight sensitivity, multi-seed ablation, cross-seed recalibration)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dsmp.revision_experiments import main


if __name__ == "__main__":
    main()

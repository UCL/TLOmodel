import os
from pathlib import Path

PROFILING_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
PROFILING_RESULTS = (PROFILING_DIR / "results").resolve()

TLO_ROOT = (PROFILING_DIR / ".." / ".." / "..").resolve()
TLO_OUTPUT_DIR = (TLO_ROOT / "outputs").resolve()

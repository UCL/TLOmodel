import os
from pathlib import Path

PROFILING_DIR = Path(os.path.abspath(os.path.dirname(__file__)))

TLO_ROOT = (PROFILING_DIR / ".." / ".." / "..").resolve()
TLO_OUTPUT_DIR = (TLO_ROOT / "outputs").resolve()

PROFILING_RESULTS = (TLO_ROOT / "profiling_results").resolve()

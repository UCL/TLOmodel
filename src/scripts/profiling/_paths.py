import os
from pathlib import Path

PROFILING_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
PROFILING_HTML = (PROFILING_DIR / "html_outputs").resolve()
PROFILING_JSON = (PROFILING_DIR / "json_outputs").resolve()
PROFILING_RTME = (PROFILING_DIR / "runtime_outputs").resolve()

TLO_ROOT = (PROFILING_DIR / ".." / ".." / "..").resolve()
TLO_OUTPUT_DIR = (TLO_ROOT / "outputs").resolve()

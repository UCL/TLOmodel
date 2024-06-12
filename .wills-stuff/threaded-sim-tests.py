import os
from pathlib import Path
import sys

LOC_OF_THIS_FILE = Path(os.path.abspath(os.path.dirname(__file__)))
SPEEDUP_RESULTS_DIR = LOC_OF_THIS_FILE / "first_appt_experiments"
SCALE_RUN_DIR = (LOC_OF_THIS_FILE / ".." / "src" / "scripts" / "profiling").resolve()

# Hacky import :sweat_smile:
sys.path.append(str(SCALE_RUN_DIR))
from scale_run import scale_run
import os
import pandas as pd
import time
from pathlib import Path

import pytest

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    antenatal_care,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthsystem,
    labour,
    newborn_outcomes,
    pregnancy_supervisor,
)

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 500

outputpath = Path("./outputs")  # folder for convenience of storing outputs


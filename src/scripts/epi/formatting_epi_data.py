"""
This is a file written by Tara to process the data from the Malawi EPI JRF
It creates:
* ResourceFile_epi_coverage
plus some lovely plots

pre-processing:
in excel remove the multiline entries in headers
find/replace: find alt 0010, replace ""
doesn't remove all so have to do some manual checks
remove any spaces or special characters in headers

2012 is compiled from the raw data

"""
from pathlib import Path

import numpy as np
import pandas as pd

resourcefilepath = Path("./resources")

epi = pd.read_excel(Path(resourcefilepath) / "ResourceFile_EPI.xlsx", sheet_name=None,)

vax2010 = epi["epi2010"]
vax2011 = epi["epi2011"]
vax2013 = epi["epi2013"]
vax2014 = epi["epi2014"]
vax2015 = epi["epi2015"]
vax2016 = epi["epi2016"]
vax2017 = epi["epi2017"]
vax2018 = epi["epi2018"]

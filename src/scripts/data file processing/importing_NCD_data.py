"""

This file gets the various NCD-related Excel file that contains parameters and prevalence data for NCDs.

This file also contains information on all HSI events relating to hypertension.

The file was generated by Mikaela Smit

"""

import os
import shutil
from pathlib import Path

tlo_dropbox_base = Path.home() / Path('Dropbox/Projects - ongoing/Malawi Project/Thanzi la Onse')

# Import and copy file for hypertension:
shutil.copy(
    tlo_dropbox_base / '04 - Methods Repository/Method_HT.xlsx',
    Path(os.path.dirname(__file__)) / '../../../resources' / 'ResourceFile_Method_HT.xlsx',
)

# Import and copy file for hypertension:
shutil.copy(
    tlo_dropbox_base / '04 - Methods Repository/Method_T2DM.xlsx',
    Path(os.path.dirname(__file__)) / '../../../resources' / 'ResourceFile_Method_T2DM.xlsx',
)

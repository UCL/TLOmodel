"""Make a graph of the GBD data, showing DALYS lived in Malawi (by age and disease), for those diseases currently in the TLO framework.
Tim's initial concept. Might need to ask Robbie to help!"""

import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods import contraception, demography
from tlo.util import create_age_range_lookup

# Where will outputs go
outputpath = Path("./outputs")

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# resource files
resourcefilepath = Path("./resources")

# load gbd data
gbd = pd.read_csv(Path(resourcefilepath) / "ResourceFile_Deaths_And_Causes_DeathRates_GBD.csv")

# limit to 2017
gbd = gbd.loc[gbd.year == 2017]

# collapse into age/cause count of death:
gbd = gbd.groupby(by=['Age_Grp', 'cause_name'])['val'].sum().reset_index()

# Make categorical list of causes to display:
causes = ['HIV/AIDS',
          'TB',
          'Malaria',
          'Other infections',
          'Childhood Diarrhoea',
          'Diabetes',
          'Stroke & Heart Disease',
          'Cancers',
          'Other conditions',
          'Road injuries',
          'Other - Not Modelled'
]

# recode the name to distinguish between those causes in the TLO model currently
gbd['cause'] = 'Other - Not Modelled'

gbd.loc[gbd.cause_name.str.contains('Diarr'), 'cause'] = 'Childhood Diarrhoea'
gbd.loc[gbd.cause_name.str.contains('Lower respiratory infections'), 'cause'] = 'Other infections'
gbd.loc[gbd.cause_name.str.contains('Tuberculosis'), 'cause'] = 'TB'
gbd.loc[gbd.cause_name.str.contains('AIDS'), 'cause'] = 'HIV/AIDS'
gbd.loc[gbd.cause_name.str.contains('Malaria'), 'cause'] = 'Malaria'
gbd.loc[gbd.cause_name.str.contains('Measles'), 'cause'] = 'Other infections'
gbd.loc[gbd.cause_name.str.contains('Schistosomiasis'), 'cause'] = 'Other infections'
gbd.loc[gbd.cause_name.str.contains('Road injuries'), 'cause'] = 'Road injuries'
gbd.loc[gbd.cause_name.str.contains('neoplasm'), 'cause'] = 'Cancers'
gbd.loc[gbd.cause_name.str.contains('cancer'), 'cause'] = 'Cancers'
gbd.loc[gbd.cause_name.str.contains('Self-harm'), 'cause'] = 'Other conditions'
gbd.loc[gbd.cause_name.str.contains('Epilepsy'), 'cause'] = 'Other conditions'
gbd.loc[gbd.cause_name.str.contains('Diabetes'), 'cause'] = 'Diabetes'
gbd.loc[gbd.cause_name.str.contains('Stroke'), 'cause'] = 'Stroke & Heart Disease'
gbd.loc[gbd.cause_name.str.contains('heart disease'), 'cause'] = 'Stroke & Heart Disease'
gbd.loc[gbd.cause_name.str.contains('Neonatal disorders'), 'cause'] = 'Other conditions'

# check no typo
assert gbd.cause.isin(causes).all()

gbd.cause = pd.Categorical(gbd.cause, categories=causes, ordered=True)

# collapse by the selected causes of death
gbd = gbd.groupby(by=['Age_Grp', 'cause'])['val'].sum().unstack()

# Reset index so that it looks in natural order,
gbd.index = ['1-4'] + [f"{x}-{x+4}" for x in range(5, 95, 5)] + ['95+']

cols = plt.cm.Paired(np.arange(len(causes)))

# set last colour (for 'Other-Not Modelled')

h = gbd.plot.bar(stacked=True, cmap='nipy_spectral')
plt.title('Causes of Death (Malawi, 2017)')
plt.xlabel('Age-group')
plt.ylabel('Number of deaths')
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(reversed(handles), reversed(labels), title='Cause', loc='upper right')
plt.show()


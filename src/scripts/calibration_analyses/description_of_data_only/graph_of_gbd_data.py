"""Make a graph of the GBD data, showing DALYS lived in Malawi (by age and disease), for those diseases currently in the
 TLO framework."""

import datetime
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import format_gbd

# Where will outputs go
outputpath = Path("./outputs")

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# resource files
resourcefilepath = Path("./resources")

# load gbd data
gbd = pd.read_csv(Path(resourcefilepath) / "gbd" / "ResourceFile_Deaths_And_DALYS_GBD2019.csv")
gbd = format_gbd(gbd)

# limit to 2017 and deaths only
gbd = gbd.loc[(gbd.Year == 2019) & (gbd.measure_name == "Deaths")]

# collapse into age/cause count of death:
gbd = gbd.groupby(by=['Age_Grp', 'cause_name'])['GBD_Est'].sum().reset_index()

# Make categorical list of causes to display:
causes = ['HIV/AIDS',
          'TB',
          'Malaria',
          'Childhood Diarrhoea',
          'Childhood Pneumonia',
          'Neonatal Disorders',
          'Diabetes',
          'Stroke & Heart Disease',
          'Cancers',
          'Road injuries',
          'Other conditions - Modelled',
          'Other - Not Yet Modelled'
          ]

# recode the name to distinguish between those causes in the TLO model currently
gbd['cause'] = 'Other - Not Yet Modelled'

gbd.loc[gbd.cause_name.str.contains('Diarr'), 'cause'] = 'Childhood Diarrhoea'
gbd.loc[gbd.cause_name.str.contains('Lower respiratory infections'), 'cause'] = 'Childhood Pneumonia'

gbd.loc[gbd.cause_name.str.contains('Tuberculosis'), 'cause'] = 'TB'
gbd.loc[gbd.cause_name.str.contains('AIDS'), 'cause'] = 'HIV/AIDS'
gbd.loc[gbd.cause_name.str.contains('Malaria'), 'cause'] = 'Malaria'

gbd.loc[gbd.cause_name.str.contains('Road injuries'), 'cause'] = 'Road injuries'
gbd.loc[gbd.cause_name.str.contains('neoplasm'), 'cause'] = 'Cancers'
gbd.loc[gbd.cause_name.str.contains('cancer'), 'cause'] = 'Cancers'

gbd.loc[gbd.cause_name.str.contains('Diabetes'), 'cause'] = 'Diabetes'
gbd.loc[gbd.cause_name.str.contains('Stroke'), 'cause'] = 'Stroke & Heart Disease'
gbd.loc[gbd.cause_name.str.contains('heart disease'), 'cause'] = 'Stroke & Heart Disease'
gbd.loc[gbd.cause_name.str.contains('Neonatal disorders'), 'cause'] = 'Neonatal Disorders'
gbd.loc[gbd.cause_name.str.contains('Self-harm'), 'cause'] = 'Other conditions - Modelled'
gbd.loc[gbd.cause_name.str.contains('Epilepsy'), 'cause'] = 'Other conditions - Modelled'
gbd.loc[gbd.cause_name.str.contains('Schistosomiasis'), 'cause'] = 'Other conditions - Modelled'
gbd.loc[gbd.cause_name.str.contains('Measles'), 'cause'] = 'Other conditions - Modelled'

# check no typo and make into Categorical
assert gbd.cause.isin(causes).all()
gbd.cause = pd.Categorical(gbd.cause, categories=causes, ordered=True)
assert not gbd.cause.isna().any()

# collapse by the selected causes of death
gbd = gbd.groupby(by=['Age_Grp', 'cause'])['GBD_Est'].sum().unstack()
gbd = gbd.fillna(0)

# Reset index so that it looks in natural order,
# ordering_index = [x.split(' ', 1)[0] for x in list(gbd.index)]
# ordering_index = [int(x) if (x != '<1') else 0 for x in ordering_index]
# ordering_index = list(pd.Series(dict(zip(ordering_index, list(gbd.index)))).sort_index().values)
# gbd = gbd.reindex(ordering_index)
# gbd.index = [x.replace(' to ', '-').replace(' year', '') for x in gbd.index]
# cols = plt.cm.Paired(np.arange(len(causes)))

# set last colour (for 'Other-Not Modelled')

h = gbd.plot.bar(stacked=True, cmap='nipy_spectral')
plt.title('Causes of Death (Malawi, 2017)')
plt.xlabel('Age-group')
plt.ylabel('Number of deaths')
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(reversed(handles), reversed(labels), loc='upper right', ncol=2)
plt.show()

# percent of deaths due to HIV, NCD (stroke, heart disease, diabetes) and childhood (pneumon, diarrhoa, neonatal)
gbd.sum()[[
    'HIV/AIDS',
    'Childhood Diarrhoea',
    'Childhood Pneumonia',
    'Neonatal Disorders',
    'Diabetes',
    'Stroke & Heart Disease'
]].sum() / gbd.sum().sum()

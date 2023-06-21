
"""
Extracts DALYs and mortality from the TB module
 """
import datetime
import pickle
from pathlib import Path
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import tlo
from tlo.analysis.utils import extract_results, summarize
from tlo import Date


resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
#outputpath = Path("./outputs/nic503@york.ac.uk")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

with open(outputpath / "default_run3.pickle", "rb") as f:
    output = pickle.load(f)

TARGET_PERIOD = (Date(2010, 1, 1), Date(2013, 12, 31))

def get_num_deaths(_df):
    """Return total number of Deaths (total within the TARGET_PERIOD)
    """
    return pd.Series(data=len(_df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)]))

def get_num_dalys(_df):
    """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD)"""
    return pd.Series(
        data=_df
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])]
        .drop(columns=['date', 'sex', 'age_range', 'year'])
        .sum().sum()
    )


# Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
#    extent of the error bar
def make_plot(_df, annotations=None):
    yerr = np.array([
        (_df['mean'] - _df['lower']).values,
        (_df['upper'] - _df['mean']).values,
    ])

    xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}
    fig, ax = plt.subplots()
    ax.bar(
        xticks.keys(),
        _df['mean'].values,
        yerr=yerr,
        alpha=0.5,
        ecolor='black',
        capsize=10,
    )
    if annotations:
        for xpos, ypos, text in zip(xticks.keys(), _df['mean'].values, annotations):
            ax.text(xpos, ypos, text, horizontalalignment='center')
    ax.set_xticks(list(xticks.keys()))
    ax.set_xticklabels(list(xticks.values()), rotation=90)
    ax.grid(axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    plt.savefig(
        #   outputpath / (name_of_plot.replace(" ", "_") + datestamp + ".pdf"), format="pdf"
    )
    return fig, ax

# Quantify the health gains associated with all interventions combined
# with open(outputpath / "default_run.pickle", "rb") as f:
#     output = pickle.load(f)



# num_deaths_summarized = summarize(num_deaths).loc[0].unstack()
# num_dalys_summarized = summarize(num_dalys).loc[0].unstack()

print(output.keys())
#print(output(tlo.methods.healthsystem.summary).keys())
# output serialises availability of  CXR consumables
cons_available = output['tlo.methods.healthsystem.summary']['Consumables'].drop(columns=[])
cons_available .to_excel(outputpath / "cons_available_cxrscaleup.xlsx")

# output YLLs and YLDs

# print(f" expected ylds{output['tlo.methods.healthburden']['yld_by_causes_of_disability']}")
# ylds = output['tlo.methods.healthburden']['yld_by_causes_of_disability'].drop(columns=[])
# ylds.to_excel(outputpath / "ylds_NoXpert.xlsx")
# print(f"expected ylls{output['tlo.methods.healthburden']['yll_by_causes_of_death_stacked']}")
# yll_output = output['tlo.methods.healthburden']['yll_by_causes_of_death_stacked'].drop(columns=[])
# yll_output.to_excel(outputpath / "sample_yll_NoXpert.xlsx")

# Exports TB program indicators
print(f"projected TB incidence{output['tlo.methods.tb']['tb_incidence']}")
TB_incidence= output['tlo.methods.tb']['tb_incidence'].drop(columns=[])
TB_incidence.to_excel(outputpath / "new_TB_cases_cxrscaleup.xlsx")
#
# print(f"projected treatment coverage {output['tlo.methods.tb']['tb_incidence']}")
# TB_treatment_cov= output['tlo.methods.tb']['tb_treatment'].drop(columns=[])
# TB_treatment_cov.to_excel(outputpath / "TB_treatment_base.xlsx")

# output DALYs
print(f"expected dalys{output['tlo.methods.healthburden']['dalys_stacked']}")
#sample_dalys= output['tlo.methods.healthburden']['dalys_stacked'].groupby(['cause', 'sex']).size()
sample_dalys= output['tlo.methods.healthburden']['dalys_stacked'].drop(columns=[])
sample_dalys.to_excel(outputpath / "sample_dalys_cxrscaleup.xlsx")

# output serialises mortality patterns
print(f"expected deaths {output['tlo.methods.demography']['death']}")
sample_deaths = output['tlo.methods.demography']['death'].groupby(['date','cause', 'sex']).size()
#sample_deaths = output['tlo.methods.demography']['death'].drop(columns=[])
sample_deaths.to_excel(outputpath / "expected_mortality_cxrscaleup.xlsx")


"""Module contains functions to be passed to LinearModel.custom function

The following template can be used for implementing:

def predict_for_individual(self, df, rng=None, **externals):
    # this is a single row dataframe. get the individual record.
    person = df.iloc[0]
    params = self.parameters
    result = 0.0  # or other intercept value
    # ...implement model here, adjusting result...
    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)

or

def predict_for_dataframe(self, df, rng=None, **externals):
    params = self.parameters
    result = pd.Series(data=params['some_intercept'], index=df.index)
    # result series has same index as dataframe, update as required
    # e.g. result[df.age == 5.0] += params['some_value']
    return result
"""
import pandas as pd


def predict_parity(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['intercept_parity_lr2010'], index=df.index)
    result += df.age_years * 0.22
    result[df.li_mar_stat == 2] += params['effect_mar_stat_2_parity_lr2010']
    result[df.li_mar_stat == 3] += params['effect_mar_stat_3_parity_lr2010']
    result += df.li_wealth.map(
        {
            1: params[f'effect_wealth_lev_1_parity_lr2010'],
            2: params[f'effect_wealth_lev_2_parity_lr2010'],
            3: params[f'effect_wealth_lev_3_parity_lr2010'],
            4: params[f'effect_wealth_lev_4_parity_lr2010'],
            5: params[f'effect_wealth_lev_5_parity_lr2010'],
        }
    )
    return result


def predict_obstructed_labour_ip(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.module.parameters
    causes = self.module.cause_of_obstructed_labour.to_strings(person.la_obstructed_labour_causes)
    result = 0.0
    if 'cephalopelvic_dis' in causes:
        result += params['prob_obstruction_cpd']
    if 'malposition' in causes:
        result += params['prob_obstruction_malpos']
    if 'malpresentation' in causes:
        result += params['prob_obstruction_malpres']
    return pd.Series(data=[result], index=df.index)


def predict_chorioamnionitis_ip(self, df, rng=None, **externals):
    """individual level"""
    params = self.parameters
    result = params['prob_chorioamnionitis_ip']
    if externals['received_clean_delivery']:
        result *= params['treatment_effect_maternal_infection_clean_delivery']
    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)

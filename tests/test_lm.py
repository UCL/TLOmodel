import io
from pathlib import Path

from tlo.lm import LinearModel, Predictor
import pandas as pd
import numpy as np


def test_of_example_useage():
    # Test the use of basic functions using different syntax and model types

    EXAMPLE_POP = """region_of_residence,li_urban,sex,age_years,sy_vomiting
    Northern,True,M,12,False
    Central,True,M,6,True
    Northern,True,M,24,False
    Southern,True,M,46,True
    Central,True,M,91,True
    Central,False,M,16,False
    Southern,False,F,80,True
    Northern,True,F,99,False
    Western,True,F,63,False
    Central,True,F,51,True
    Central,True,M,57,False
    Central,False,F,2,False
    Central,True,F,93,False
    Western,False,M,15,True
    Western,False,M,5,False
    Northern,True,M,29,True
    Western,True,M,63,False
    Southern,True,F,54,False
    Western,False,M,94,False
    Northern,False,F,91,True
    Northern,True,M,29,False
    """

    # Linear Model
    eq = LinearModel(
        'linear',
        0.0,
        Predictor('region_of_residence').when('Northern', 0.1).when('Central', 0.2).when('Southern', 0.3),
        Predictor('li_urban').when(True, 0.01).otherwise(0.02),
        Predictor('sex').when('M', 0.001).when('F', 0.002),
        Predictor('age_years')
            .when('< 5', 0.0001)
            .when('< 15', 0.0002)
            .when('< 35', 0.0003)
            .when('< 60', 0.0004)
            .otherwise(0.0005),
        Predictor('sy_vomiting').when(True, 0.00001).otherwise(0.00002)
    )

    df = pd.read_csv(io.StringIO(EXAMPLE_POP))
    predicted = eq.predict(df)
    df['predicted'] = predicted
    print(df.to_string())

    # Logistic model
    eq = LinearModel(
        'logistic',
        1.0,
        Predictor('region_of_residence').when('Northern', 1.0).when('Central', 1.1).when('Southern', 0.8),
        Predictor('sy_vomiting').when(True, 2.5).otherwise(1.0),
        Predictor('age_years')
            .when('.between(0,5)', 0.001)
            .otherwise(0),
    )

    df = pd.read_csv(io.StringIO(EXAMPLE_POP))
    predicted = eq.predict(df)
    df['predicted'] = predicted
    print(df.to_string())

    # Multiplicative model
    eq = LinearModel(
        'multiplicative',
        0.02,
        Predictor('region_of_residence').when('Northern', 1.0).when('Central', 1.1).when('Southern', 0.8),
        Predictor('sy_vomiting').when(True, 2.5).otherwise(1.0)
    )

    df = pd.read_csv(io.StringIO(EXAMPLE_POP))
    predicted = eq.predict(df)
    df['predicted'] = predicted
    print(df.to_string())

def test_linear_application():
    # Take an example from ????
    pass

def test_multiplicative_application():
    # Take an example from lifestyle
    pass

def test_logistic_application():
    # Take an example from lifestyle at initiation

    # 1) load a df from a csv file that has is a 'freeze-frame' of for sim.population.props
    df=pd.read_csv('tests/resources/df_at_init_of_lifestyle.csv')

    # 2) generate the probabilities from the model in the 'classical' manner
    init_p_low_ex_urban_m = 0.32
    init_or_low_ex_f=0.6
    init_or_low_ex_rural=0.4
    age_ge15_idx = df.index[df.is_alive & (df.age_years >= 15)]
    init_odds_low_ex_urban_m = init_p_low_ex_urban_m / (1 - init_p_low_ex_urban_m)
    odds_low_ex = pd.Series(init_odds_low_ex_urban_m, index=age_ge15_idx)
    odds_low_ex.loc[df.sex == 'F'] *= init_or_low_ex_f
    odds_low_ex.loc[~df.li_urban] *= init_or_low_ex_rural
    low_ex_probs = odds_low_ex / (1 + odds_low_ex)

    # 3) apply the LinearModel to it and make a prediction of the probabilities assinged to each person
    eq = LinearModel(
        'logistic',
        init_p_low_ex_urban_m/(1-init_p_low_ex_urban_m),
        Predictor('li_urban').when(False, init_or_low_ex_rural).otherwise(1.0),
        Predictor('sex').when('F',init_or_low_ex_f).otherwise(1.0)
    )

    lm_low_ex_probs =eq.predict(df)

    # init_p_low_ex_urban_m=0.32
    # init_log_odds= np.log(init_p_low_ex_urban_m/(1-init_p_low_ex_urban_m))
    # output = 1 / (1 + np.exp(-init_log_odds))

    # 4) confirm that the two methods agree
    assert lm_low_ex_probs.values == low_ex_probs.values



"""
        age_ge15_idx = df.index[df.is_alive & (df.age_years >= 15)]

        init_odds_low_ex_urban_m = m.init_p_low_ex_urban_m / (1 - m.init_p_low_ex_urban_m)

        odds_low_ex = pd.Series(init_odds_low_ex_urban_m, index=age_ge15_idx)

        odds_low_ex.loc[df.sex == 'F'] *= m.init_or_low_ex_f
        odds_low_ex.loc[~df.li_urban] *= m.init_or_low_ex_rural

        low_ex_probs = odds_low_ex / (1 + odds_low_ex)

        random_draw = rng.random_sample(size=len(age_ge15_idx))
        df.loc[age_ge15_idx, 'li_low_ex'] = random_draw < low_ex_probs

        # -------------------- TOBACCO USE ---------------------------------------------------------

        init_odds_tob_age1519_m_wealth1 = m.init_p_tob_age1519_m_wealth1 / (1 - m.init_p_tob_age1519_m_wealth1)

        odds_tob = pd.Series(init_odds_tob_age1519_m_wealth1, index=age_ge15_idx)

        odds_tob.loc[df.sex == 'F'] *= m.init_or_tob_f
        odds_tob.loc[(df.sex == 'M') & (df.age_years >= 20) & (df.age_years < 40)] *= m.init_or_tob_age2039_m
        odds_tob.loc[(df.sex == 'M') & (df.age_years >= 40)] *= m.init_or_tob_agege40_m
        odds_tob.loc[df.li_wealth == 2] *= 2
        odds_tob.loc[df.li_wealth == 3] *= 3
        odds_tob.loc[df.li_wealth == 4] *= 4
        odds_tob.loc[df.li_wealth == 5] *= 5

        tob_probs = odds_tob / (1 + odds_tob)

        random_draw = rng.random_sample(size=len(age_ge15_idx))
"""

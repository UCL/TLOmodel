import io
import os
from pathlib import Path

import pandas as pd

from tlo.lm import LinearModel, Predictor


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


def test_linear_trivial_application():
    eq = LinearModel(
        'linear',
        0.0,
        Predictor('FactorX').when(True, 10),
        Predictor('FactorY').when(True, 100)
    )

    df = pd.DataFrame(data={
        'FactorX': [False, True, False, True],
        'FactorY': [False, False, True, True]
    })

    pred = eq.predict(df)
    assert all(pred.values == [0.0, 10.0, 100.0, 110.0])


def test_multiplier_trivial_application():
    eq = LinearModel(
        'multiplicative',
        1.0,
        Predictor('FactorX').when(True, 5),
        Predictor('FactorY').when(True, -1)
    )

    df = pd.DataFrame(data={
        'FactorX': [False, True, False, True],
        'FactorY': [False, False, True, True]
    })

    pred = eq.predict(df)
    assert all(pred.values == [1.0, 5.0, -1.0, -5.0])


def test_logistic_trivial_application():
    prob = 0.5
    OR_X = 2
    OR_Y = 5

    odds = prob / (1 - prob)

    eq = LinearModel(
        'logistic',
        odds,
        Predictor('FactorX').when(True, OR_X),
        Predictor('FactorY').when(True, OR_Y)
    )

    df = pd.DataFrame(data={
        'FactorX': [False, True, False, True],
        'FactorY': [False, False, True, True]
    })

    pred = eq.predict(df)
    assert all(pred.values == [
        prob,
        (odds * OR_X) / (1 + odds * OR_X),
        (odds * OR_Y) / (1 + odds * OR_Y),
        (odds * OR_X * OR_Y) / (1 + odds * OR_X * OR_Y)
    ])


def test_logistic_application_low_ex():
    # Use an example from lifestyle at initiation: low exercise

    # 1) load a df from a csv file that has is a 'freeze-frame' of for sim.population.props
    df_file = Path(os.path.dirname(__file__)) / 'resources' / 'df_at_init_of_lifestyle.csv'
    df = pd.read_csv(df_file)
    df.set_index('person', inplace=True, drop=True)

    # 2) generate the probabilities from the model in the 'classical' manner
    init_p_low_ex_urban_m = 0.32
    init_or_low_ex_f = 0.6
    init_or_low_ex_rural = 0.4
    age_ge15_idx = df.index[df.is_alive & (df.age_years >= 15)]
    init_odds_low_ex_urban_m = init_p_low_ex_urban_m / (1 - init_p_low_ex_urban_m)
    odds_low_ex = pd.Series(init_odds_low_ex_urban_m, index=age_ge15_idx)
    odds_low_ex.loc[df.sex == 'F'] *= init_or_low_ex_f
    odds_low_ex.loc[~df.li_urban] *= init_or_low_ex_rural
    low_ex_probs = odds_low_ex / (1 + odds_low_ex)

    # 3) apply the LinearModel to it and make a prediction of the probabilities assigned to each person
    eq = LinearModel(
        'logistic',
        init_p_low_ex_urban_m / (1 - init_p_low_ex_urban_m),
        Predictor('li_urban').when(False, init_or_low_ex_rural),
        Predictor('sex').when('F', init_or_low_ex_f)
    )
    lm_low_ex_probs = eq.predict(df.loc[df.is_alive & (df.age_years >= 15)])

    # 4) confirm that the two methods agree
    assert all(lm_low_ex_probs.values == low_ex_probs.values)


def test_logistic_application_tob():
    # Use an example from lifestyle at initiation: tob (tobacco use)

    # 1) load a df from a csv file that has is a 'freeze-frame' of for sim.population.props
    df_file = Path(os.path.dirname(__file__)) / 'resources' / 'df_at_init_of_lifestyle.csv'
    df = pd.read_csv(df_file)
    df.set_index('person', inplace=True, drop=True)

    # 2) generate the probabilities from the model in the 'classical' manner
    init_p_tob_age1519_m_wealth1 = 0.7
    init_or_tob_f = 0.8
    init_or_tob_agege40_m = 0.2
    init_or_tob_age2039_m = 0.9
    age_ge15_idx = df.index[df.is_alive & (df.age_years >= 15)]
    init_odds_tob_age1519_m_wealth1 = init_p_tob_age1519_m_wealth1 / (1 - init_p_tob_age1519_m_wealth1)
    odds_tob = pd.Series(init_odds_tob_age1519_m_wealth1, index=age_ge15_idx)
    odds_tob.loc[df.sex == 'F'] *= init_or_tob_f
    odds_tob.loc[(df.sex == 'M') & (df.age_years >= 20) & (df.age_years < 40)] *= init_or_tob_age2039_m
    odds_tob.loc[(df.sex == 'M') & (df.age_years >= 40)] *= init_or_tob_agege40_m
    odds_tob.loc[df.li_wealth == 2] *= 2
    odds_tob.loc[df.li_wealth == 3] *= 3
    odds_tob.loc[df.li_wealth == 4] *= 4
    odds_tob.loc[df.li_wealth == 5] *= 5
    tob_probs = odds_tob / (1 + odds_tob)

    # 3) apply the LinearModel to it and make a prediction of the probabilities assigned to each person
    # [As there is a joint condition on age and sex, need to build two seperate models]
    eq_adults = LinearModel(
        'multiplicative',
        init_p_tob_age1519_m_wealth1 / (1 - init_p_tob_age1519_m_wealth1),
        Predictor('sex').when('F', init_or_tob_f),
        Predictor('li_wealth').when('2', 2).when('3', 3).when('4', 4).when('5', 5)
    )

    eq_men_only = LinearModel(
        'multiplicative',
        1.0,
        Predictor('age_years').when('.between(20,39)', init_or_tob_age2039_m),
        Predictor('age_years').when('.between(40,120)', init_or_tob_agege40_m)
    )

    men_only = eq_men_only.predict(df.loc[df.is_alive & (df.age_years >= 15) & (df.sex == 'M')])
    men_only = pd.DataFrame(men_only).merge(df[['age_years','sex']], left_index=True, right_index=True)


    lm_tob_odds = eq_adults.predict(df.loc[df.is_alive & (df.age_years >= 15)]).multiply(
        eq_men_only.predict(df.loc[df.is_alive & (df.age_years >= 15) & (df.sex == 'M')]),
        fill_value = 1.0
    )
    lm_tob_probs = lm_tob_odds / (1 + lm_tob_odds)

    assert all(tob_probs == lm_tob_probs)

    # TODO: when the the condition is being equal to an number, it has to be passed into .when() as a string (e.g. li_weath)
    # TODO: a more elegant more to handle conditions that depend on two things? (as per test_logistic_application_tob)
    # TODO: warning when the same state is indicated for twice in then when() statement.

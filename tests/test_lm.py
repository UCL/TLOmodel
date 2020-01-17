import io
import os
from pathlib import Path

import pandas as pd

from tlo.lm import LinearModel, LinearModelType, Predictor

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

EXAMPLE_DF = pd.read_csv(io.StringIO(EXAMPLE_POP))


def test_of_example_usage():
    # Test the use of basic functions using different syntax and model types

    # Linear Model
    eq = LinearModel(
        LinearModelType.ADDITIVE,
        0.0,
        Predictor('region_of_residence').when('Northern', 0.1).when('Central', 0.2).when('Southern', 0.3),
        Predictor('li_urban').when(True, 0.01).otherwise(0.02),
        Predictor('sex').when('M', 0.001).when('F', 0.002),
        Predictor('age_years').when('< 5', 0.0001)
                              .when('< 15', 0.0002)
                              .when('< 35', 0.0003)
                              .when('< 60', 0.0004)
                              .otherwise(0.0005),
        Predictor('sy_vomiting').when(True, 0.00001).otherwise(0.00002)
    )

    eq.predict(EXAMPLE_DF)

    # Logistic model
    eq = LinearModel(
        LinearModelType.LOGISTIC,
        1.0,
        Predictor('region_of_residence').when('Northern', 1.0).when('Central', 1.1).when('Southern', 0.8),
        Predictor('sy_vomiting').when(True, 2.5).otherwise(1.0),
        Predictor('age_years')
        .when('.between(0,5)', 0.001)
        .otherwise(0),
    )
    eq.predict(EXAMPLE_DF)

    # Multiplicative model
    eq = LinearModel(
        LinearModelType.MULTIPLICATIVE,
        0.02,
        Predictor('region_of_residence').when('Northern', 1.0).when('Central', 1.1).when('Southern', 0.8),
        Predictor('sy_vomiting').when(True, 2.5).otherwise(1.0)
    )
    eq.predict(EXAMPLE_DF)


def test_additive_trivial_application():
    eq = LinearModel(
        LinearModelType.ADDITIVE,
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
        LinearModelType.MULTIPLICATIVE,
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
        LinearModelType.LOGISTIC,
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


def test_external_variable():
    eq = LinearModel(
        LinearModelType.ADDITIVE,
        0.0,
        Predictor('region_of_residence').when('Northern', 0.1).otherwise(0.3),
        Predictor('year', external=True).when('.between(0,2019)', 1).when(2020, 2).otherwise(3)
    )

    output = eq.predict(EXAMPLE_DF, year=2010)
    assert output.tolist() == [1.1, 1.3, 1.1, 1.3, 1.3, 1.3, 1.3, 1.1, 1.3, 1.3, 1.3,
                               1.3, 1.3, 1.3, 1.3, 1.1, 1.3, 1.3, 1.3, 1.1, 1.1]

    output = eq.predict(EXAMPLE_DF, year=2020)
    assert output.tolist() == [2.1, 2.3, 2.1, 2.3, 2.3, 2.3, 2.3, 2.1, 2.3, 2.3, 2.3,
                               2.3, 2.3, 2.3, 2.3, 2.1, 2.3, 2.3, 2.3, 2.1, 2.1]

    output = eq.predict(EXAMPLE_DF, year=2021)
    assert output.tolist() == [3.1, 3.3, 3.1, 3.3, 3.3, 3.3, 3.3, 3.1, 3.3, 3.3, 3.3,
                               3.3, 3.3, 3.3, 3.3, 3.1, 3.3, 3.3, 3.3, 3.1, 3.1]


def test_multiple_external_variables():
    eq = LinearModel(
        LinearModelType.ADDITIVE,
        0.0,
        Predictor('region_of_residence').when('Northern', 100).otherwise(200),
        Predictor('tens_digit', external=True).when('a', 10).when('b', 20).otherwise(30),
        Predictor('units_digit', external=True).when('x', 4).when('y', 5).otherwise(6)
    )

    def get_digit(n, i):
        return n // 10**i % 10

    output = eq.predict(EXAMPLE_DF, tens_digit='a', units_digit='z')
    assert (get_digit(output, 1) == 1).all()
    assert (get_digit(output, 0) == 6).all()

    output = eq.predict(EXAMPLE_DF, tens_digit='b', units_digit='y')
    assert (get_digit(output, 1) == 2).all()
    assert (get_digit(output, 0) == 5).all()


def test_callback_value():
    # as lambda
    eq = LinearModel(
        LinearModelType.ADDITIVE,
        0.0,
        Predictor('age_years').apply(lambda x: x / 100)
    )
    output1 = eq.predict(EXAMPLE_DF)

    # as function
    def callback(x):
        return x/100

    eq2 = LinearModel(
        LinearModelType.ADDITIVE,
        0.0,
        Predictor('age_years').apply(callback)
    )
    output2 = eq2.predict(EXAMPLE_DF)

    assert output1.tolist() == (EXAMPLE_DF.age_years/100).tolist()
    assert output1.tolist() == output2.tolist()


def test_callback_with_external_variable():
    eq = LinearModel(
        LinearModelType.ADDITIVE,
        0.0,
        Predictor('region_of_residence').when('Northern', 1).otherwise(2),
        Predictor('year', external=True).apply(lambda x: (x - 10) / 10000)
    )
    output1 = eq.predict(EXAMPLE_DF, year=2019)
    assert output1.tolist() == [1.2009, 2.2009, 1.2009, 2.2009, 2.2009, 2.2009, 2.2009, 1.2009,
                                2.2009, 2.2009, 2.2009, 2.2009, 2.2009, 2.2009, 2.2009, 1.2009,
                                2.2009, 2.2009, 2.2009, 1.2009, 1.2009]

    output2 = eq.predict(EXAMPLE_DF, year=2010)
    assert output2.tolist() == [1.2000, 2.2000, 1.2000, 2.2000, 2.2000, 2.2000, 2.2000, 1.2000,
                                2.2000, 2.2000, 2.2000, 2.2000, 2.2000, 2.2000, 2.2000, 1.2000,
                                2.2000, 2.2000, 2.2000, 1.2000, 1.2000]


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
        LinearModelType.LOGISTIC,
        init_p_low_ex_urban_m / (1 - init_p_low_ex_urban_m),
        Predictor('li_urban').when(False, init_or_low_ex_rural),
        Predictor('sex').when('F', init_or_low_ex_f)
    )
    lm_low_ex_probs = eq.predict(df.loc[df.is_alive & (df.age_years >= 15)])

    # 4) confirm that the two methods agree
    assert lm_low_ex_probs.equals(low_ex_probs)


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
    tob_probs: pd.Series = odds_tob / (1 + odds_tob)

    # 3) apply the LinearModel to it and make a prediction of the probabilities assigned to each person
    eq_tob = LinearModel(
        LinearModelType.LOGISTIC,
        init_p_tob_age1519_m_wealth1 / (1 - init_p_tob_age1519_m_wealth1),
        Predictor('sex').when('F', init_or_tob_f),
        Predictor('li_wealth').when(2, 2).when(3, 3).when(4, 4).when(5, 5),
        Predictor().when('(age_years.between(20,39)) & (sex == "M")', init_or_tob_age2039_m)
                   .when('(age_years.between(40,120)) & (sex == "M")', init_or_tob_agege40_m)
    )

    lm_tob_probs = eq_tob.predict(df.loc[df.is_alive & (df.age_years >= 15)])

    assert tob_probs.equals(lm_tob_probs)

    # TODO: when the the condition is being equal to an number,
    # it has to be passed into .when() as a string (e.g. li_weath)
    # TODO: a more elegant more to handle conditions that depend on two things? (as per test_logistic_application_tob)
    # TODO: warning when the same state is indicated for twice in then when() statement.

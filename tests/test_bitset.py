from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from tlo import Population
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.util import BitsetHandler


def test_bitset():
    # set up a mock population, only has props dataframe
    mock = Mock(spec=Population)
    df = pd.DataFrame(data={
        'symptoms': pd.Series(0, index=range(5), dtype=np.dtype('int64')),
        'is_alive': pd.Series(True, index=range(5), dtype=np.dtype('bool'))
    })
    mock.configure_mock(props=df)

    # create the handler to work with symptoms column on the population dataframe
    symptoms = BitsetHandler(mock, 'symptoms', ['fever', 'cough', 'nausea', 'vomiting'])

    u = symptoms.uncompress()
    assert len(u) == 5
    assert np.all(u.columns == pd.Index(['fever', 'cough', 'nausea', 'vomiting']))
    assert (~u.all()).all()

    # individual
    symptoms.set([0], 'nausea')
    u = symptoms.uncompress()
    assert (u.loc[0] == [False, False, True, False]).all()

    # subset
    symptoms.set(df.index % 2 == 0, 'vomiting')
    u = symptoms.uncompress()
    assert (u.loc[df.index % 2 == 0, 'vomiting']).all()
    assert (~u.loc[df.index % 2 != 0, 'vomiting']).all()

    # all individuals (== who are alive)
    symptoms.set(df.is_alive, 'cough')
    u = symptoms.uncompress()
    assert (u.cough).all()

    # unset one individual (now has no symptoms)
    symptoms.unset([1], 'cough')
    u = symptoms.uncompress()
    assert (~u.loc[1]).all()

    # no one should have fever
    assert (~symptoms.has_all(df.is_alive, 'fever')).all()

    # who has a cough
    assert (symptoms.has_all(df.is_alive, 'cough') == [True, False, True, True, True]).all()

    # does individual 1 have vomiting or fever - use 'pop' to get single entry, not Series
    assert (symptoms.has_any([0], 'fever', 'vomiting', pop=True))

    # who has both cough and vomiting
    assert (symptoms.has_all(df.is_alive, 'cough', 'vomiting') == [True, False, True, False, True]).all()

    u = symptoms.uncompress()
    check = pd.DataFrame(data={
        'fever': False,
        'cough': [True, False, True, True, True],
        'nausea': [True, False, False, False, False],
        'vomiting': [True, False, True, False, True]
    }, index=range(5))
    pd.testing.assert_frame_equal(u, check)

    # get symptoms for single individual
    person_symp = symptoms.uncompress([2])
    person_symp = person_symp.loc[2]  # turn dataframe into series with column as index
    assert not person_symp.fever and person_symp.cough and not person_symp.nausea & person_symp.vomiting

    # get set of strings of elements (might be easier to work with in some circumstances...?)
    sets = symptoms.get(df.is_alive)
    assert len(sets.loc[1]) == 0
    assert len(sets.loc[3]) == 1 and 'cough' in sets.loc[3]

    person_symptoms = symptoms.get([4], pop=True)
    assert person_symptoms.difference({'cough', 'vomiting'}) == set()


def test_linearmodel_with_bitset():
    # set up a mock population, only has props dataframe
    mock = Mock(spec=Population)
    df = pd.DataFrame(data={
        'symptoms': pd.Series(0, index=range(5), dtype=np.dtype('int64')),
        'is_alive': pd.Series(True, index=range(5), dtype=np.dtype('bool')),
        'age': pd.Series([5, 10, 20, 30, 40], index=range(5), dtype=np.dtype('int'))
    })
    mock.configure_mock(props=df)

    # create the handler to work with symptoms column on the population dataframe
    symptoms = BitsetHandler(mock, 'symptoms', ['fever', 'cough', 'nausea', 'vomiting'])

    # set
    symptoms.set(df.index % 2 == 0, 'fever')
    symptoms.set(df.index % 2 != 0, 'cough')
    symptoms.set([4], 'vomiting')
    symptoms.unset([3], 'cough')

    # fever  cough  nausea  vomiting
    # 0   True  False   False     False
    # 1  False   True   False     False
    # 2   True  False   False     False
    # 3  False  False   False     False
    # 4   True  False   False      True

    # have any symptoms
    lm = LinearModel(
        LinearModelType.ADDITIVE,
        1.0,
        Predictor('symptoms').when('>0', 10)
                             .otherwise(20),
    )

    out = lm.predict(df)
    assert pytest.approx(21.0) == out.loc[3]

    # has specific symptoms - vomiting
    lm = LinearModel(
        LinearModelType.ADDITIVE,
        0.0,
        Predictor('symptoms').apply(lambda x: 2.0 if x & symptoms.element_repr('vomiting') else 20.0)
    )

    out = lm.predict(df)
    pd.testing.assert_series_equal(out, pd.Series([20.0, 20.0, 20.0, 20.0, 2.0]))

    # put more complex rules in its own function
    def symptom_coeff_calc(bitset):
        if bitset & symptoms.element_repr('fever'):
            if bitset & symptoms.element_repr('vomiting'):
                return 1
            else:
                return 2
        return 3

    lm = LinearModel(
        LinearModelType.ADDITIVE,
        0.0,
        Predictor('symptoms').apply(symptom_coeff_calc)
    )

    out = lm.predict(df)
    pd.testing.assert_series_equal(out, pd.Series([2.0, 3.0, 2.0, 3.0, 1.0]))

    # use external variables
    lm = LinearModel(
        LinearModelType.ADDITIVE,
        0.0,
        Predictor('fever_and_vomiting', external=True).when(True, 1)
                                                      .otherwise(2)
    )

    out = lm.predict(df.loc[df.is_alive],
                     fever_and_vomiting=symptoms.has_all(df.is_alive, 'fever', 'vomiting'))

    pd.testing.assert_series_equal(out, pd.Series([2.0, 2.0, 2.0, 2.0, 1.0]))

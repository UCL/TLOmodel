from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from tlo import Population
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.util import BitsetHandler


@pytest.fixture
def dataframe():
    return pd.DataFrame(
        data={
            'symptoms': pd.Series(0, index=range(5), dtype=np.dtype('uint64')),
            'sy_stomachache': pd.Series(0, index=range(5), dtype=np.dtype('uint64')),
            'sy_injury':  pd.Series(0, index=range(5), dtype=np.dtype('uint64')),
            'is_alive': pd.Series(True, index=range(5), dtype=np.dtype('bool')),
            'age': pd.Series([5, 10, 20, 30, 40], index=range(5), dtype=np.dtype('int'))
        }
    )


@pytest.fixture
def sy_columns(dataframe):
    return [col for col in dataframe.columns if col[:3] == 'sy_']


@pytest.fixture
def population(dataframe):
    # set up a mock population, only has props dataframe
    population = Mock(spec=Population)
    population.configure_mock(props=dataframe)
    return population


@pytest.fixture
def symptoms(population):
    # create the handler to work with symptoms column on the population dataframe
    return BitsetHandler(
        population, 'symptoms', ['fever', 'cough', 'nausea', 'vomiting']
    )


@pytest.fixture
def updated_symptoms(dataframe, symptoms):
    symptoms.set([0], 'nausea')
    symptoms.set(dataframe.index % 2 == 0, 'vomiting')
    symptoms.set(dataframe.is_alive, 'cough')
    symptoms.unset([1], 'cough')
    return symptoms


@pytest.fixture
def module_bitset_handler(population):
    return BitsetHandler(
        population, None, ['Module1', 'Module2', 'Module3', 'Module4']
    )


def test_error_on_too_many_elements(population):
    with pytest.raises(AssertionError, match='maximum'):
        BitsetHandler(population, 'symptoms', [str(i) for i in range(100)])


def test_error_on_incorrect_column_dtype(population):
    with pytest.raises(AssertionError, match='uint64'):
        BitsetHandler(population, 'is_alive', [str(i) for i in range(8)])


def test_error_on_missing_column(population):
    with pytest.raises(AssertionError, match='not found'):
        BitsetHandler(population, 'test', [str(i) for i in range(8)])


def test_error_on_incorrect_population_type(dataframe):
    with pytest.raises(AssertionError, match='population object'):
        BitsetHandler(dataframe, 'symptoms', [str(i) for i in range(8)])


def test_error_on_set_without_columns_arg(module_bitset_handler):
    with pytest.raises(ValueError, match='columns'):
        module_bitset_handler.set([0], 'Module1')


def test_error_on_unset_without_columns_arg(module_bitset_handler):
    with pytest.raises(ValueError, match='columns'):
        module_bitset_handler.unset([0], 'Module1')


def test_uncompress(symptoms):
    u = symptoms.uncompress()
    assert len(u) == 5
    assert np.all(u.columns == pd.Index(['fever', 'cough', 'nausea', 'vomiting']))
    assert (~u.all()).all()


def test_uncompress_multiple_col(module_bitset_handler, sy_columns):
    uncompressed = module_bitset_handler.uncompress(columns=sy_columns)
    assert isinstance(uncompressed, dict)
    assert all(k in sy_columns for k in uncompressed.keys())
    assert all(isinstance(v, pd.DataFrame) for v in uncompressed.values())
    assert all(len(v) == 5 and (~v).all().all() for v in uncompressed.values())


def test_set_individual(symptoms):
    symptoms.set([0], 'nausea')
    u = symptoms.uncompress()
    assert (u.loc[0] == [False, False, True, False]).all()


def test_unset_individual(symptoms):
    symptoms.unset([0], 'fever')
    u = symptoms.uncompress()
    assert not u.loc[0].any()


def test_set_unset_individual(symptoms):
    symptoms.set([0], 'fever')
    symptoms.unset([0], 'fever')
    u = symptoms.uncompress()
    assert not u.loc[0].any()


def test_set_individual_multiple_columns(module_bitset_handler, sy_columns):
    module_bitset_handler.set([0], 'Module1', columns=sy_columns)
    uncompressed = module_bitset_handler.uncompress(columns=sy_columns)
    for col in sy_columns:
        assert (uncompressed[col].loc[0] == [True, False, False, False]).all()


def test_unset_individual_multiple_columns(module_bitset_handler, sy_columns):
    module_bitset_handler.unset([0], 'Module2', columns=sy_columns)
    uncompressed = module_bitset_handler.uncompress(columns=sy_columns)
    for col in sy_columns:
        assert not uncompressed[col].loc[0].any()


def test_set_unset_individual_multiple_columns(module_bitset_handler, sy_columns):
    module_bitset_handler.set([0], 'Module1', 'Module3', columns=sy_columns)
    module_bitset_handler.unset([0], 'Module3', columns=sy_columns)
    uncompressed = module_bitset_handler.uncompress(columns=sy_columns)
    for col in sy_columns:
        assert (uncompressed[col].loc[0] == [True, False, False, False]).all()


def test_set_multiple_rows(symptoms):
    symptoms.set([1, 2], 'nausea', 'vomiting')
    u = symptoms.uncompress()
    assert (u.loc[1] == [False, False, True, True]).all()
    assert (u.loc[2] == [False, False, True, True]).all()


def test_set_multiple_rows_explicit_col(symptoms):
    symptoms.set([1, 2], 'nausea', 'vomiting', columns='symptoms')
    u = symptoms.uncompress()
    assert (u.loc[1] == [False, False, True, True]).all()
    assert (u.loc[2] == [False, False, True, True]).all()


def test_set_multiple_rows_multiple_columns(module_bitset_handler, sy_columns):
    module_bitset_handler.set([0, 2], 'Module1', 'Module3', columns=sy_columns)
    uncompressed = module_bitset_handler.uncompress(columns=sy_columns)
    for col in sy_columns:
        assert (uncompressed[col].loc[0] == [True, False, True, False]).all()
        assert (uncompressed[col].loc[2] == [True, False, True, False]).all()


def test_set_subset(dataframe, symptoms):
    symptoms.set(dataframe.index % 2 == 0, 'vomiting')
    u = symptoms.uncompress()
    assert (u.loc[dataframe.index % 2 == 0, 'vomiting']).all()
    assert (~u.loc[dataframe.index % 2 != 0, 'vomiting']).all()


def test_set_all(dataframe, symptoms):
    # all individuals (== who are alive)
    symptoms.set(dataframe.is_alive, 'cough')
    u = symptoms.uncompress()
    assert (u.cough).all()
    # unset one individual (now has no symptoms)
    symptoms.unset([1], 'cough')
    u = symptoms.uncompress()
    assert (~u.loc[1]).all()


def test_set_all_multiple_columns(module_bitset_handler, sy_columns):
    module_bitset_handler.set(slice(None), 'Module1', columns=sy_columns)
    uncompressed = module_bitset_handler.uncompress(columns=sy_columns)
    assert all(v.Module1.all() for v in uncompressed.values())


def test_multiple_set(dataframe, symptoms):
    symptoms.set([0], 'nausea')
    symptoms.set(dataframe.index % 2 == 0, 'vomiting')
    symptoms.set(dataframe.is_alive, 'cough')
    symptoms.unset([1], 'cough')
    reference_uncompressed = pd.DataFrame(
        data={
            'fever': False,
            'cough': [True, False, True, True, True],
            'nausea': [True, False, False, False, False],
            'vomiting': [True, False, True, False, True]
        },
        index=range(5)
    )
    pd.testing.assert_frame_equal(symptoms.uncompress(), reference_uncompressed)


def test_has(symptoms):
    symptoms.set([2], 'cough')
    assert symptoms.has([2], 'cough', first=True)


def test_has_multiple_columns(module_bitset_handler, sy_columns):
    module_bitset_handler.set([2], 'Module2', columns=sy_columns)
    assert module_bitset_handler.has([2], 'Module2', columns=sy_columns).all().all()


def test_has_all(dataframe, updated_symptoms):
    # no one should have fever
    assert (~updated_symptoms.has_all(dataframe.is_alive, 'fever')).all()
    # who has a cough
    assert (
        updated_symptoms.has_all(dataframe.is_alive, 'cough')
        == [True, False, True, True, True]
    ).all()
    # who has both cough and vomiting
    assert (
        updated_symptoms.has_all(dataframe.is_alive, 'cough', 'vomiting')
        == [True, False, True, False, True]
    ).all()


def test_has_all_multiple_columns(module_bitset_handler, sy_columns):
    module_bitset_handler.set(
        [0], *(f'Module{i}' for i in range(1, 5)), columns=sy_columns[0]
    )
    module_bitset_handler.set([1], 'Module3', columns=sy_columns[1])
    has_all = module_bitset_handler.has_all(
        [0, 1], 'Module1', 'Module3', columns=sy_columns
    )
    assert (has_all[sy_columns[0]] == [True, False]).all()
    assert (has_all[sy_columns[1]] == [False, False]).all()


def test_has_any(dataframe, updated_symptoms):
    assert (
        updated_symptoms.has_any(dataframe.is_alive, 'cough', 'vomiting')
        == [True, False, True, True, True]
    ).all()
    # does individual 1 have vomiting or fever - use 'pop' to get single entry, not Series
    assert (updated_symptoms.has_any([0], 'fever', 'vomiting', first=True))


def test_has_any_multiple_columns(module_bitset_handler, sy_columns):
    module_bitset_handler.set([0], 'Module1', columns=sy_columns[0])
    module_bitset_handler.set([1], 'Module3', columns=sy_columns[1])
    has_any = module_bitset_handler.has_any(
        [0, 1], 'Module1', 'Module3', columns=sy_columns
    )
    assert (has_any[sy_columns[0]] == [True, False]).all()
    assert (has_any[sy_columns[1]] == [False, True]).all()


def test_uncompress_individual(updated_symptoms):
    # get symptoms for single individual
    person_symp = updated_symptoms.uncompress([2])
    person_symp = person_symp.loc[2]  # turn dataframe into series with column as index
    assert (
        not person_symp.fever
        and person_symp.cough
        and not person_symp.nausea
        and person_symp.vomiting
    )


def test_to_strings(symptoms):
    assert symptoms.to_strings(0) == set()
    assert symptoms.to_strings(1) == {'fever'}
    assert symptoms.to_strings(2**4 - 1) == {'cough', 'nausea', 'fever', 'vomiting'}


def test_get(dataframe, updated_symptoms):
    # get set of strings of elements (might be easier to work with in some circumstances...?)
    sets = updated_symptoms.get(dataframe.is_alive)
    assert len(sets.loc[1]) == 0
    assert len(sets.loc[3]) == 1 and 'cough' in sets.loc[3]
    person_symptoms = updated_symptoms.get([4], first=True)
    assert person_symptoms.difference({'cough', 'vomiting'}) == set()


def test_is_empty(symptoms):
    assert (symptoms.is_empty([0]) == [True]).all()
    assert symptoms.is_empty([0], first=True)


def test_not_empty(updated_symptoms):
    assert (updated_symptoms.not_empty([0]) == [True]).all()
    assert updated_symptoms.not_empty([0], first=True)


def test_clear(updated_symptoms):
    assert updated_symptoms.not_empty([0], first=True)
    updated_symptoms.clear([0])
    assert updated_symptoms.is_empty([0], first=True)


def test_clear_multiple_columns(module_bitset_handler, sy_columns):
    module_bitset_handler.set([0, 1], 'Module1', 'Module2', columns=sy_columns)
    assert module_bitset_handler.not_empty([0, 1], columns=sy_columns).any().any()
    module_bitset_handler.clear([0, 1], columns=sy_columns)
    assert module_bitset_handler.is_empty([0, 1], columns=sy_columns).all().all()


def test_linearmodel_with_bitset(dataframe, symptoms):
    # set
    symptoms.set(dataframe.index % 2 == 0, 'fever')
    symptoms.set(dataframe.index % 2 != 0, 'cough')
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

    out = lm.predict(dataframe)
    assert pytest.approx(21.0) == out.loc[3]
    
    vomitting_element_repr = symptoms.element_repr('vomiting')

    # has specific symptoms - vomiting
    lm = LinearModel(
        LinearModelType.ADDITIVE,
        0.0,
        Predictor('symptoms').apply(lambda x: 2.0 if np.uint64(x) & symptoms.element_repr('vomiting') else 20.0)
    )

    out = lm.predict(dataframe)
    pd.testing.assert_series_equal(out, pd.Series([20.0, 20.0, 20.0, 20.0, 2.0]))

    # put more complex rules in its own function
    def symptom_coeff_calc(bitset):
        if np.uint64(bitset) & symptoms.element_repr('fever'):
            if np.uint64(bitset) & symptoms.element_repr('vomiting'):
                return 1
            else:
                return 2
        return 3

    lm = LinearModel(
        LinearModelType.ADDITIVE,
        0.0,
        Predictor('symptoms').apply(symptom_coeff_calc)
    )

    out = lm.predict(dataframe)
    pd.testing.assert_series_equal(out, pd.Series([2.0, 3.0, 2.0, 3.0, 1.0]))

    # use external variables
    lm = LinearModel(
        LinearModelType.ADDITIVE,
        0.0,
        Predictor('fever_and_vomiting', external=True).when(True, 1)
                                                      .otherwise(2)
    )

    out = lm.predict(dataframe.loc[dataframe.is_alive],
                     fever_and_vomiting=symptoms.has_all(dataframe.is_alive, 'fever', 'vomiting'))

    pd.testing.assert_series_equal(out, pd.Series([2.0, 2.0, 2.0, 2.0, 1.0]))

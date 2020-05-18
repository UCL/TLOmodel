from unittest.mock import Mock

import numpy as np
import pandas as pd

from tlo import Population
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

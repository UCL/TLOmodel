import pandas as pd
import pytest

from tlo import Date, Property, Simulation, Types


def test_categorical():
    categories = ['M', 'F']
    p = Property(Types.CATEGORICAL, description='Biological Sex', categories=categories)
    s = p.create_series('Sex', 10)
    assert s.dtype.name == 'category'
    assert list(s.cat.categories) == categories

    # assignment of valid category
    try:
        s[0] = 'M'
    except ValueError:
        pytest.fail('Unexpected ValueError')

    # assignment of invalid category
    with pytest.raises(ValueError):
        s[0] = 'A'



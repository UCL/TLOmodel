import io
from pathlib import Path

from tlo.lm import LinearModel, Predictor
import pandas as pd

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
    # Take an example from health seeking behaviour
    pass

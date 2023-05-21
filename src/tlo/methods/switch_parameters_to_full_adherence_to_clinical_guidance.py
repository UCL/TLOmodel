from pathlib import Path

import pandas as pd


def parameters_for_full_adherence_to_clinical_guidance(self):
    """
    In this function, we return a dictionary of parameters and their default values and modules to indicate
    full adherence to clinical guidance. By implementing this function, we could switch our tlo model between
    current parameter settings and default/ideal parameter setting and compare the health outcome.
    """

    params = self.parameters

    # get the resource file for the list of parameters, their default values and modules

    # prepare the dictionary, where
    # if the value of the parameter is simply a float or int, add the default value
    # if the value of the parameter is a dataframe, replace specific entries with the default values
    ideal_params = {}

    return ideal_params

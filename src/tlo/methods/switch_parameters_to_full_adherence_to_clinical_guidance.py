from pathlib import Path

import pandas as pd


def parameters_for_an_ideal_health_system(self, scenario):
    """
    In this function, we return a dictionary of parameters and their default values and modules to indicate
    an ideal healthcare system in terms of
    (1) If_HealthCareProvsion: full adherence to clinical guidelines by health care providers,
    (2) If_HealthCareSeeking: full adherence to medication/care and treatment by patients,
    and (3) If_HCWCompetence: ideal HCW competence.
    By implementing this function, we could switch our tlo model between
    current parameter settings and default/ideal parameters setting and compare the health outcome.

    :param scenario: a list for the ideal healthcare system settings; it can be
    [],
    ["If_HealthCareProvision"],
    ["If_HealthCareSeeking"],
    ["If_HCWCompetence"],
    ["If_HealthCareProvision", "If_HealthCareSeeking"],
    ["If_HealthCareProvision", "If_HCWCompetence"],
    ["If_HealthCareSeeking", "If_HCWCompetence"],
    ["If_HealthCareProvision", "If_HealthCareSeeking", "If_HCWCompetence"]
    """

    params = self.parameters

    # get the resource file for the list of parameters, their default values and modules
    # if the parameter is a dataframe, its values are in a separate sheet
    workbook_path = Path('./resources/healthsystem/ResourceFile_Ideal_HealthCare_Provision_And_Seeking.xlsx')
    params_default = pd.read_excel(workbook_path, sheet_name='parameter')

    # prepare the params dictionary to be switched to, where
    # if the value of the parameter is simply a float or int, add the default value
    # if the value of the parameter is a dataframe,
    # add the default dataframe/or have to replace specific entries in original dataframe from "params"?
    # e.g. {'Depression': {'pr_assessed_for_depression_for_perinatal_female': 1.0,
    #                      'pr_assessed_for_depression_in_generic_appt_level1': 1.0},
    #       'Hiv': {'prob_start_art_or_vs': the dataframe from params_default}}
    ideal_params = {k: {} for k in params_default['Module'].drop_duplicates()}
    if len(scenario) > 0:
        for i in params_default.index:
            for s in ['If_HealthCareProvision', 'If_HealthCareSeeking', 'If_HCWCompetence']:
                # make sure the parameter to be updated is in the correct scenario
                if (s in scenario) and (params_default.loc[i, s]):
                    # if the parameter is a single value
                    if ~params_default.loc[i, 'If_DataFrame']:
                        ideal_params[params_default.loc[i, 'Module']].update(
                            {params_default.loc[i, 'Parameter']: params_default.loc[i, 'Default_Value']})
                    # if the parameter is a dataframe
                    else:
                        p = pd.read_excel(workbook_path, sheet_name=params_default.loc[i, 'Parameter'])
                        ideal_params[params_default.loc[i, 'Module']].update(
                            {params_default.loc[i, 'Parameter']: p})

    return ideal_params

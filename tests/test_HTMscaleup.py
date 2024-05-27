""" Tests for setting up the HIV, TB and malaria scenarios used for projections """

import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    malaria,
    simplified_births,
    symptommanager,
    tb,
)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"
except NameError:
    # running interactively
    resourcefilepath = "resources"


def get_sim(seed, scaleup_hiv=False, scaleup_tb=False, scaleup_malaria=False, scaleup_start_date=pd.NaT):
    """
    register all necessary modules for the tests to run
    """

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            service_availability=["*"],  # all treatment allowed
            mode_appt_constraints=1,  # mode of constraints to do with officer numbers and time
            cons_availability="default",  # mode for consumable constraints (if ignored, all consumables available)
            ignore_priority=True,  # do not use the priority information in HSI event to schedule
            capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
            disable=False,  # disables the healthsystem (no constraints and no logging) and every HSI runs
            disable_and_reject_all=False,  # disable healthsystem and no HSI runs
        ),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        epi.Epi(resourcefilepath=resourcefilepath),
        hiv.Hiv(resourcefilepath=resourcefilepath,
                scaleup_hiv=scaleup_hiv,
                scaleup_tb=scaleup_tb,
                scaleup_malaria=scaleup_malaria,
                scaleup_start_date=scaleup_start_date),
        tb.Tb(resourcefilepath=resourcefilepath),
        malaria.Malaria(resourcefilepath=resourcefilepath),
    )

    return sim


def test_hiv_scale_up(seed):
    """ test hiv program scale-up changes parameters correctly
    and on correct date """

    workbook = pd.read_excel(
        os.path.join(resourcefilepath, "ResourceFile_HIV.xlsx"),
        sheet_name=None,
    )

    # Load data on HIV prevalence
    original_params = workbook["parameters"]
    new_params = workbook["scaleup_parameters"]
    scaleup_start_date = Date(2011, 1, 1)

    popsize = 100

    sim = get_sim(seed=seed, scaleup_hiv=True, scaleup_start_date=scaleup_start_date)

    # check initial parameters
    # todo do we need to be exhaustive and check every parameter here?
    assert sim.modules["Hiv"].parameters["beta"] == \
           original_params.loc[original_params.parameter_name == "beta", "value"].values[0]
    assert sim.modules["Hiv"].parameters["prob_prep_for_fsw_after_hiv_test"] == original_params.loc[
        original_params.parameter_name == "prob_prep_for_fsw_after_hiv_test", "value"].values[0]
    assert sim.modules["Hiv"].parameters["prob_prep_for_agyw"] == original_params.loc[
        original_params.parameter_name == "prob_prep_for_agyw", "value"].values[0]
    assert sim.modules["Hiv"].parameters["probability_of_being_retained_on_prep_every_3_months"] == original_params.loc[
        original_params.parameter_name == "probability_of_being_retained_on_prep_every_3_months", "value"].values[0]
    assert sim.modules["Hiv"].parameters["prob_circ_after_hiv_test"] == original_params.loc[
        original_params.parameter_name == "prob_circ_after_hiv_test", "value"].values[0]

    # Make the population
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=scaleup_start_date + pd.DateOffset(days=1))

    # check HIV parameters changed
    assert sim.modules["Hiv"].parameters["beta"] < original_params.loc[original_params.parameter_name == "beta", "value"].values[0]
    assert sim.modules["Hiv"].parameters["prob_prep_for_fsw_after_hiv_test"] == new_params.loc[
        new_params.parameter == "prob_prep_for_fsw_after_hiv_test", "scaleup_value"].values[0]
    assert sim.modules["Hiv"].parameters["prob_prep_for_agyw"] == new_params.loc[
        new_params.parameter == "prob_prep_for_agyw", "scaleup_value"].values[0]
    assert sim.modules["Hiv"].parameters["probability_of_being_retained_on_prep_every_3_months"] == new_params.loc[
        new_params.parameter == "probability_of_being_retained_on_prep_every_3_months", "scaleup_value"].values[0]
    assert sim.modules["Hiv"].parameters["prob_circ_after_hiv_test"] == new_params.loc[
        new_params.parameter == "prob_circ_after_hiv_test", "scaleup_value"].values[0]

    # check TB and malaria parameters unchanged
    mal_workbook = pd.read_excel(
        os.path.join(resourcefilepath, "malaria/ResourceFile_Malaria.xlsx"),
        sheet_name=None,
    )
    mal_original_params = mal_workbook["parameters"]
    mal_rdt_testing = mal_workbook["WHO_TestData2023"]

    assert sim.modules["Malaria"].parameters["prob_malaria_case_tests"] == mal_original_params.loc[
        mal_original_params.parameter_name == "prob_malaria_case_tests", "value"].values[0]
    pd.testing.assert_series_equal(sim.modules["Malaria"].parameters["rdt_testing_rates"]["Rate_rdt_testing"],
                                   mal_rdt_testing["Rate_rdt_testing"])




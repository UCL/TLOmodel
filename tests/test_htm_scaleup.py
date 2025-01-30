""" Tests for setting up the HIV, TB and malaria scenarios used for projections """

import os
from pathlib import Path

import pandas as pd

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
from tlo.util import parse_csv_values_for_columns_with_mixed_datatypes, read_csv_files

resourcefilepath = Path(os.path.dirname(__file__)) / "../resources"

start_date = Date(2010, 1, 1)
scaleup_start_year = 2012  # <-- the scale-up will occur on 1st January of that year
end_date = Date(2013, 1, 1)


def get_sim(seed):
    """
    register all necessary modules for the tests to run
    """

    sim = Simulation(start_date=start_date, seed=seed)

    # Register the appropriate modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        epi.Epi(resourcefilepath=resourcefilepath),
        hiv.Hiv(resourcefilepath=resourcefilepath),
        tb.Tb(resourcefilepath=resourcefilepath),
        malaria.Malaria(resourcefilepath=resourcefilepath),
    )

    return sim


def check_initial_params(sim):

    original_params = read_csv_files(resourcefilepath / 'ResourceFile_HIV', files='parameters')
    original_params.value = original_params.value.apply(parse_csv_values_for_columns_with_mixed_datatypes)

    # check initial parameters
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


def test_hiv_scale_up(seed):
    """ test hiv program scale-up changes parameters correctly
    and on correct date """

    original_params = read_csv_files(resourcefilepath / 'ResourceFile_HIV', files="parameters")
    original_params.value = original_params.value.apply(parse_csv_values_for_columns_with_mixed_datatypes)
    new_params = read_csv_files(resourcefilepath / 'ResourceFile_HIV', files="scaleup_parameters")

    popsize = 100

    sim = get_sim(seed=seed)

    # check initial parameters
    check_initial_params(sim)

    # update parameters to instruct there to be a scale-up
    sim.modules["Hiv"].parameters["type_of_scaleup"] = 'target'
    sim.modules["Hiv"].parameters["scaleup_start_year"] = scaleup_start_year

    # Make the population
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # check HIV parameters changed
    assert sim.modules["Hiv"].parameters["beta"] < original_params.loc[
        original_params.parameter_name == "beta", "value"].values[0]
    assert sim.modules["Hiv"].parameters["prob_prep_for_fsw_after_hiv_test"] == new_params.loc[
        new_params.parameter == "prob_prep_for_fsw_after_hiv_test", "target_value"].values[0]
    assert sim.modules["Hiv"].parameters["prob_prep_for_agyw"] == new_params.loc[
        new_params.parameter == "prob_prep_for_agyw", "target_value"].values[0]
    assert sim.modules["Hiv"].parameters["probability_of_being_retained_on_prep_every_3_months"] == new_params.loc[
        new_params.parameter == "probability_of_being_retained_on_prep_every_3_months", "target_value"].values[0]
    assert sim.modules["Hiv"].parameters["prob_circ_after_hiv_test"] == new_params.loc[
        new_params.parameter == "prob_circ_after_hiv_test", "target_value"].values[0]

    # check malaria parameters unchanged
    mal_original_params = read_csv_files(resourcefilepath / 'malaria' / 'ResourceFile_malaria',
                                        files="parameters")
    mal_original_params.value = mal_original_params.value.apply(parse_csv_values_for_columns_with_mixed_datatypes)

    mal_rdt_testing = read_csv_files(resourcefilepath / 'malaria' / 'ResourceFile_malaria',
                                    files="WHO_TestData2023")

    assert sim.modules["Malaria"].parameters["prob_malaria_case_tests"] == mal_original_params.loc[
        mal_original_params.parameter_name == "prob_malaria_case_tests", "value"].values[0]
    pd.testing.assert_series_equal(sim.modules["Malaria"].parameters["rdt_testing_rates"]["Rate_rdt_testing"],
                                   mal_rdt_testing["Rate_rdt_testing"])

    # all irs coverage levels should be < 1.0
    assert sim.modules["Malaria"].itn_irs['irs_rate'].all() < 1.0
    # itn rates for 2019 onwards
    assert sim.modules["Malaria"].parameters["itn"] == mal_original_params.loc[
        mal_original_params.parameter_name == "itn", "value"].values[0]

    # check tb parameters unchanged
    tb_original_params = read_csv_files(resourcefilepath / 'ResourceFile_TB', files="parameters")
    tb_original_params.value = tb_original_params.value.apply(parse_csv_values_for_columns_with_mixed_datatypes)
    tb_testing = read_csv_files(resourcefilepath / 'ResourceFile_TB', files="NTP2019")

    pd.testing.assert_series_equal(sim.modules["Tb"].parameters["rate_testing_active_tb"]["treatment_coverage"].astype(float),
                                   tb_testing["treatment_coverage"].astype(float))
    assert sim.modules["Tb"].parameters["prob_tx_success_ds"] == tb_original_params.loc[
        tb_original_params.parameter_name == "prob_tx_success_ds", "value"].values[0]
    assert sim.modules["Tb"].parameters["prob_tx_success_mdr"] == tb_original_params.loc[
        tb_original_params.parameter_name == "prob_tx_success_mdr", "value"].values[0]
    assert sim.modules["Tb"].parameters["prob_tx_success_0_4"] == tb_original_params.loc[
        tb_original_params.parameter_name == "prob_tx_success_0_4", "value"].values[0]
    assert sim.modules["Tb"].parameters["prob_tx_success_5_14"] == tb_original_params.loc[
        tb_original_params.parameter_name == "prob_tx_success_5_14", "value"].values[0]
    assert sim.modules["Tb"].parameters["first_line_test"] == tb_original_params.loc[
        tb_original_params.parameter_name == "first_line_test", "value"].values[0]


def test_htm_scale_up(seed):
    """ test hiv/tb/malaria program scale-up changes parameters correctly
    and on correct date """

    # Load data on HIV prevalence
    original_hiv_params = read_csv_files(resourcefilepath / 'ResourceFile_HIV', files="parameters")
    original_hiv_params.value = original_hiv_params.value.apply(parse_csv_values_for_columns_with_mixed_datatypes)
    new_hiv_params = read_csv_files(resourcefilepath / 'ResourceFile_HIV', files="scaleup_parameters")

    popsize = 100

    sim = get_sim(seed=seed)

    # check initial parameters
    check_initial_params(sim)

    # update parameters
    sim.modules["Hiv"].parameters["type_of_scaleup"] = 'target'
    sim.modules["Hiv"].parameters["scaleup_start_year"] = scaleup_start_year
    sim.modules["Tb"].parameters["type_of_scaleup"] = 'target'
    sim.modules["Tb"].parameters["scaleup_start_year"] = scaleup_start_year
    sim.modules["Malaria"].parameters["type_of_scaleup"] = 'target'
    sim.modules["Malaria"].parameters["scaleup_start_year"] = scaleup_start_year

    # Make the population
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # check HIV parameters changed
    assert sim.modules["Hiv"].parameters["beta"] < original_hiv_params.loc[
        original_hiv_params.parameter_name == "beta", "value"].values[0]
    assert sim.modules["Hiv"].parameters["prob_prep_for_fsw_after_hiv_test"] == new_hiv_params.loc[
        new_hiv_params.parameter == "prob_prep_for_fsw_after_hiv_test", "target_value"].values[0]
    assert sim.modules["Hiv"].parameters["prob_prep_for_agyw"] == new_hiv_params.loc[
        new_hiv_params.parameter == "prob_prep_for_agyw", "target_value"].values[0]
    assert sim.modules["Hiv"].parameters["probability_of_being_retained_on_prep_every_3_months"] == new_hiv_params.loc[
        new_hiv_params.parameter == "probability_of_being_retained_on_prep_every_3_months", "target_value"].values[0]
    assert sim.modules["Hiv"].parameters["prob_circ_after_hiv_test"] == new_hiv_params.loc[
        new_hiv_params.parameter == "prob_circ_after_hiv_test", "target_value"].values[0]

    # check malaria parameters changed
    new_mal_params = read_csv_files(resourcefilepath / 'malaria' / 'ResourceFile_malaria',
                                   files="scaleup_parameters")

    assert sim.modules["Malaria"].parameters["prob_malaria_case_tests"] == new_mal_params.loc[
        new_mal_params.parameter == "prob_malaria_case_tests", "target_value"].values[0]
    assert sim.modules["Malaria"].parameters["rdt_testing_rates"]["Rate_rdt_testing"].eq(new_mal_params.loc[
        new_mal_params.parameter == "rdt_testing_rates", "target_value"].values[0]).all()

    # some irs coverage levels should now = 1.0
    assert sim.modules["Malaria"].itn_irs['irs_rate'].any() == 1.0
    # itn rates for 2019 onwards
    assert sim.modules["Malaria"].parameters["itn"] == new_mal_params.loc[
        new_mal_params.parameter == "itn", "target_value"].values[0]

    # check tb parameters changed
    new_tb_params = read_csv_files(resourcefilepath / 'ResourceFile_TB', files="scaleup_parameters")
    new_tb_params.target_value = new_tb_params.target_value.apply(parse_csv_values_for_columns_with_mixed_datatypes)

    assert sim.modules["Tb"].parameters["rate_testing_active_tb"]["treatment_coverage"].eq(new_tb_params.loc[
        new_tb_params.parameter == "tb_treatment_coverage", "target_value"].values[0]).all()
    assert sim.modules["Tb"].parameters["prob_tx_success_ds"] == new_tb_params.loc[
        new_tb_params.parameter == "tb_prob_tx_success_ds", "target_value"].values[0]
    assert sim.modules["Tb"].parameters["prob_tx_success_mdr"] == new_tb_params.loc[
        new_tb_params.parameter == "tb_prob_tx_success_mdr", "target_value"].values[0]
    assert sim.modules["Tb"].parameters["prob_tx_success_0_4"] == new_tb_params.loc[
        new_tb_params.parameter == "tb_prob_tx_success_0_4", "target_value"].values[0]
    assert sim.modules["Tb"].parameters["prob_tx_success_5_14"] == new_tb_params.loc[
        new_tb_params.parameter == "tb_prob_tx_success_5_14", "target_value"].values[0]
    assert sim.modules["Tb"].parameters["first_line_test"] == new_tb_params.loc[
        new_tb_params.parameter == "first_line_test", "target_value"].values[0]


# todo: add a "basic_run" with checks on the logical consistency of all values, as done for the original diabetes/HT
# (inserted here is the basic scaffold)

import os
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    depression,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    pregnancy_supervisor,
    symptommanager, ncds,
)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')

output_files = dict()


def routine_checks(sim):
    """
    Insert checks here:
    """

    # Check types of columns
    df = sim.population.props
    orig = sim.population.new_row
    assert (df.dtypes == orig.dtypes).all()

    # check that someone has had onset of each condition

    df = sim.population.props
    assert df.nc_diabetes.any()
    assert df.nc_hypertension.any()
    assert df.nc_depression.any()
    assert df.nc_chronic_lower_back_pain.any()
    assert df.nc_chronic_kidney_disease.any()
    assert df.nc_chronic_ischemic_hd.any()
    assert df.nc_cancers.any()

    # check that someone has had onset of each event

    assert df.nc_ever_stroke.any()

    # check that someone dies of each condition

    assert df.cause_of_death.loc[~df.is_alive].str.startswith('diabetes').any()
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('hypertension').any()
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('depression').any()
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('chronic_lower_back_pain').any()
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('chronic_ischemic_hd').any()
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('chronic_kidney_disease').any()
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('cancers').any()


def test_basic_run():
    # --------------------------------------------------------------------------
    # Create and run a short but big population simulation for use in the tests
    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 ncds.Ncds(resourcefilepath=resourcefilepath)
                 )

    sim.make_initial_population(n=5000)
    sim.simulate(end_date=Date(year=2015, month=1, day=1))

    routine_checks(sim)


def test_basic_run_with_high_incidence_hypertension():
    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 ncds.Ncds(resourcefilepath=resourcefilepath)
                 )

    # Set incidence of hypertension to 1 and incidence of all other conditions to 0
    sim.modules['Ncds'].params_dict_onset['nc_hypertension'].loc[
        sim.modules['Ncds'].params_dict_onset['nc_hypertension']
            .parameter_name == "baseline_annual_probability", "value"] = 1
    sim.modules['Ncds'].params_dict_onset['nc_diabetes'].loc[sim.modules['Ncds'].params_dict_onset['nc_diabetes']
                                                                 .parameter_name == "baseline_annual_probability", "value"] = 0
    sim.modules['Ncds'].params_dict_onset['nc_depression'].loc[sim.modules['Ncds'].params_dict_onset['nc_depression']
                                                                   .parameter_name == "baseline_annual_probability", "value"] = 0
    sim.modules['Ncds'].params_dict_onset['nc_chronic_lower_back_pain'].loc[
        sim.modules['Ncds'].params_dict_onset['nc_chronic_lower_back_pain']
            .parameter_name == "baseline_annual_probability", "value"] = 0
    sim.modules['Ncds'].params_dict_onset['nc_chronic_kidney_disease'].loc[
        sim.modules['Ncds'].params_dict_onset['nc_chronic_kidney_disease']
            .parameter_name == "baseline_annual_probability", "value"] = 0
    sim.modules['Ncds'].params_dict_onset['nc_cancers'].loc[sim.modules['Ncds'].params_dict_onset['nc_cancers']
                                                                .parameter_name == "baseline_annual_probability", "value"] = 0
    # Increase RR of heart disease very high if individual has hypertension
    sim.modules['Ncds'].params_dict_onset['nc_chronic_ischemic_hd'].loc[
        sim.modules['Ncds'].params_dict_onset['nc_hypertension']
            .parameter_name == "rr_hypertension", "value"] = 1000

    sim.make_initial_population(n=2000)
    sim.simulate(end_date=Date(year=2020, month=1, day=1))

    df = sim.population.props

    # check that no one has any conditions that were set to zero incidence
    assert ~df.nc_diabetes.all()
    assert ~df.nc_depression.all()
    assert ~df.nc_chronic_lower_back_pain.all()
    assert ~df.nc_chronic_kidney_disease.all()
    assert ~df.nc_cancers.all()

    # restrict population to individuals aged >=20 at beginning of sim
    start_date = pd.Timestamp(year=2010, month=1, day=1)
    df['start_date'] = pd.to_datetime(start_date)
    df['diff_years'] = df.start_date - df.date_of_birth
    df['diff_years'] = df.diff_years / np.timedelta64(1, 'Y')
    df = df[df['diff_years'] >= 20]
    df = df[df.is_alive]

    hypertension_prev = (len(df[df.nc_hypertension & df.is_alive & (df.age_years >= 20)])) / \
                        (len(df[df.is_alive & (df.age_years >= 20)]))
    cihd_prev = (len(df[df.nc_chronic_ischemic_hd & df.is_alive & (df.age_years >= 20)])) / \
                (len(df[df.is_alive & (df.age_years >= 20)]))

    # check that everyone has hypertension and CIHD by end
    assert hypertension_prev == 1
    assert cihd_prev == 1

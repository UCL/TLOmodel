import os
from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.methods import mnh_cohort_module
from tlo.methods.fullmodel import fullmodel
from tlo.analysis.utils import parse_log_file

# The resource files
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')

start_date = Date(2024, 1, 1)


def register_modules(sim):
    """Defines sim variable and registers all modules that can be called when running the full suite of pregnancy
    modules"""

    sim.register(*fullmodel(resourcefilepath=resourcefilepath),
                  mnh_cohort_module.MaternalNewbornHealthCohort(resourcefilepath=resourcefilepath),
                 )


def test_run_sim_with_mnh_cohort(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "custom_levels":{
                "*": logging.DEBUG},"directory": tmpdir})

    register_modules(sim)
    sim.make_initial_population(n=5000)
    sim.simulate(end_date=Date(2025, 1, 1))

    output= parse_log_file(sim.log_filepath)
    live_births = len(output['tlo.methods.demography']['on_birth'])

    deaths_df = output['tlo.methods.demography']['death']
    prop_deaths_df = output['tlo.methods.demography.detail']['properties_of_deceased_persons']

    dir_mat_deaths = deaths_df.loc[(deaths_df['label'] == 'Maternal Disorders')]
    init_indir_mat_deaths = prop_deaths_df.loc[(prop_deaths_df['is_pregnant'] | prop_deaths_df['la_is_postpartum']) &
                                  (prop_deaths_df['cause_of_death'].str.contains('Malaria|Suicide|ever_stroke|diabetes|'
                                                                     'chronic_ischemic_hd|ever_heart_attack|'
                                                                     'chronic_kidney_disease') |
                                   (prop_deaths_df['cause_of_death'] == 'TB'))]

    hiv_mat_deaths =  prop_deaths_df.loc[(prop_deaths_df['is_pregnant'] | prop_deaths_df['la_is_postpartum']) &
                              (prop_deaths_df['cause_of_death'].str.contains('AIDS_non_TB|AIDS_TB'))]

    indir_mat_deaths = len(init_indir_mat_deaths) + (len(hiv_mat_deaths) * 0.3)
    total_deaths = len(dir_mat_deaths) + indir_mat_deaths

    # TOTAL_DEATHS
    mmr = (total_deaths / live_births) * 100_000

    print(f'The MMR for this simulation is {mmr}')
    print(f'The maternal deaths for this simulation (unscaled) are {total_deaths}')
    print(f'The total maternal deaths for this simulation (scaled) are '
          f'{total_deaths * output["tlo.methods.population"]["scaling_factor"]["scaling_factor"].values[0]}')

    maternal_dalys = output['tlo.methods.healthburden']['dalys_stacked']['Maternal Disorders'].sum()
    print(f'The maternal DALYs for this simulation (unscaled) are {maternal_dalys}')



    # df = sim.population.props
    # orig = sim.population.new_row
    # assert (df.dtypes == orig.dtypes).all()

# def test_mnh_cohort_module_updates_properties_as_expected(tmpdir, seed):
#     sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "directory": tmpdir})
#
#     register_modules(sim)
#     sim.make_initial_population(n=1000)
#     sim.simulate(end_date=sim.date + pd.DateOffset(days=0))
#     # to do: check properties!!

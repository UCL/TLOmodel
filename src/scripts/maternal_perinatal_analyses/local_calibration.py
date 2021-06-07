
from pathlib import Path
import pandas as pd
from tlo.analysis.utils import parse_log_file
from tlo import Date, Simulation, logging
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    enhanced_lifestyle,
    healthsystem,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    joes_fake_props_module,
    symptommanager
)


log_config = {
    "filename": "calibration",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs/calibration_files",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.DEBUG}}

# %%

# todo: just for reference, can be deleted
antenatal_comps = ['spontaneous_abortion', 'induced_abortion', 'spontaneous_abortion_haemorrhage',
                   'induced_abortion_haemorrhage', 'spontaneous_abortion_sepsis',
                   'induced_abortion_sepsis', 'spontaneous_abortion_injury',
                   'induced_abortion_complication', 'complicated_induced_abortion',
                   'complicated_spontaneous_abortion', 'iron_deficiency', 'folate_deficiency', 'b12_deficiency',
                   'mild_anaemia', 'moderate_anaemia', 'severe_anaemia', 'gest_diab',
                   'mild_pre_eclamp', 'mild_gest_htn', 'severe_pre_eclamp', 'eclampsia', 'severe_gest_htn',
                   'placental_abruption', 'severe_antepartum_haemorrhage', 'mild_mod_antepartum_haemorrhage',
                   'clinical_chorioamnionitis', 'PROM', 'ectopic_unruptured', 'multiple_pregnancy', 'placenta_praevia',
                   'ectopic_ruptured', 'syphilis']

resourcefilepath = Path("./resources")


sim = Simulation(start_date=Date(2010, 1, 1), seed=852, log_config=log_config)
# run the simulation
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             contraception.Contraception(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                       service_availability=['*']),
             joes_fake_props_module.JoesFakePropsModule(resourcefilepath=resourcefilepath),
             labour.Labour(resourcefilepath=resourcefilepath),
             newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
             care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
             pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
             postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath))

sim.make_initial_population(n=5000)
df = sim.population.props

all = df.loc[df.is_alive]
df.loc[all.index, 'sex'] = 'F'
df.loc[all.index, 'is_pregnant'] = True
df.loc[all.index, 'date_of_last_pregnancy'] = sim.start_date
for person in all.index:
    age = sim.rng.randint(15, 49)
    df.at[person, 'age_years'] = age
    df.at[person, 'age_exact_years'] = float(age)
    df.at[person, 'age_days'] = age * 365
    df.at[person, 'date_of_birth'] = Date(2010, 1, 1) - pd.DateOffset(days=(age * 365))

    sim.modules['Labour'].set_date_of_labour(person)

params = sim.modules['PregnancySupervisor'].parameters
params['prob_ectopic_pregnancy'] = [0.004937, 0.00366]

sim.simulate(end_date=Date(2010, 3, 1))

# Get the log

log_df = parse_log_file(filepath="./outputs/calibration_files/calibration__2021-06-05T125542.log")

def get_incidence(module, complication):
    if 'maternal_complication' in log_df[f'tlo.methods.{module}']:
        comps = log_df[f'tlo.methods.{module}']['maternal_complication']
        comps['date'] = pd.to_datetime(comps['date'])
        comps['year'] = comps['date'].dt.year
        return len(comps.loc[(comps['type'] == f'{complication}')])



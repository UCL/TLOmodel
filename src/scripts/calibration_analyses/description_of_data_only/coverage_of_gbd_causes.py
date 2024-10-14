"""This script produces estimates of the fraction of the GBD causes of death/disability that are covered in the model
currently, and eventually."""

from pathlib import Path

import pandas as pd

from tlo import Date, Module, Simulation
from tlo.analysis.utils import format_gbd
from tlo.methods import (
    Metadata,
    bladder_cancer,
    breast_cancer,
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    diarrhoea,
    enhanced_lifestyle,
    epi,
    epilepsy,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    malaria,
    measles,
    newborn_outcomes,
    oesophagealcancer,
    other_adult_cancers,
    postnatal_supervisor,
    pregnancy_supervisor,
    prostate_cancer,
    symptommanager,
)
from tlo.methods.causes import Cause

resourcefilepath = Path("./resources")
outputs = Path("./outputs")


# Get all the GBD causes of death and disability:
cod = pd.Series(
    (pd.read_csv(resourcefilepath / "gbd" / "ResourceFile_CausesOfDeath_GBD2019.csv").set_index(
        ['Sex', 'Age_Grp']).columns)
).sort_values()

codis = pd.Series(
    (pd.read_csv(resourcefilepath / "gbd" / "ResourceFile_CausesOfDALYS_GBD2019.csv", header=None)[0])
).sort_values()

# %% Get all modules that have been completed in Master:
complete = [
    # Core Modules
    demography.Demography(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),

    # Representations of the Healthcare System
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
    epi.Epi(resourcefilepath=resourcefilepath),

    # - Contraception, Pregnancy and Labour
    contraception.Contraception(resourcefilepath=resourcefilepath),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
    care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
    labour.Labour(resourcefilepath=resourcefilepath),
    newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
    postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),

    # - Conditions of Early Childhood
    diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),

    # - Communicable Diseases
    hiv.Hiv(resourcefilepath=resourcefilepath),
    malaria.Malaria(resourcefilepath=resourcefilepath),
    measles.Measles(resourcefilepath=resourcefilepath),

    # - Non-Communicable Conditions
    # -- Cancers
    bladder_cancer.BladderCancer(resourcefilepath=resourcefilepath),
    breast_cancer.BreastCancer(resourcefilepath=resourcefilepath),
    oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath),
    other_adult_cancers.OtherAdultCancer(resourcefilepath=resourcefilepath),
    prostate_cancer.ProstateCancer(resourcefilepath=resourcefilepath),

    # -- Caridometabolic Diorders
    cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),

    # -- Other Non-Communicable Conditions
    depression.Depression(resourcefilepath=resourcefilepath),
    epilepsy.Epilepsy(resourcefilepath=resourcefilepath)
]

# %% Make Dummy modules for causes of death/disability in forthcoming modules


class Tb(Module):
    METADATA = {Metadata.DISEASE_MODULE}
    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'Tuberculosis':
            Cause(gbd_causes={'Tuberculosis'},
                  label='Tuberculosis')
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'Tuberculosis':
            Cause(gbd_causes={'Tuberculosis'},
                  label='Tuberculosis')
    }

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        pass


class Alri(Module):
    METADATA = {Metadata.DISEASE_MODULE}
    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'Lower respiratory infections':
            Cause(gbd_causes={'Lower respiratory infections'},
                  label='Lower respiratory infections')
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'Lower respiratory infections':
            Cause(gbd_causes={'Lower respiratory infections'},
                  label='Lower respiratory infections')
    }

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        pass


class RoadTrafficInjuries(Module):
    METADATA = {Metadata.DISEASE_MODULE}
    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'Road injuries':
            Cause(gbd_causes={'Road injuries'},
                  label='Road injuries')
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'Road injuries':
            Cause(gbd_causes={'Road injuries'},
                  label='Road injuries')
    }

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        pass


class Copd(Module):
    METADATA = {Metadata.DISEASE_MODULE}
    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'Chronic obstructive pulmonary disease':
            Cause(gbd_causes={'Chronic obstructive pulmonary disease'},
                  label='Chronic obstructive pulmonary disease')
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'Chronic obstructive pulmonary disease':
            Cause(gbd_causes={'Chronic obstructive pulmonary disease'},
                  label='Chronic obstructive pulmonary disease')
    }

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        pass


class Schisto(Module):
    METADATA = {Metadata.DISEASE_MODULE}
    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'Schistosomiasis':
            Cause(gbd_causes={'Schistosomiasis'},
                  label='Schistosomiasis')
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'Schistosomiasis':
            Cause(gbd_causes={'Schistosomiasis'},
                  label='Schistosomiasis')
    }

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        pass


class OtherInjuries(Module):
    METADATA = {Metadata.DISEASE_MODULE}
    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'Other Injuries':
            Cause(gbd_causes={'Interpersonal violence',
                              'Fire, heat, and hot substances',
                              'Falls',
                              'Exposure to mechanical forces',
                              'Drowning',
                              'Exposure to forces of nature',
                              'Foreign body',
                              'Other transport injuries',
                              'Other unintentional injuries',
                              'Poisonings',
                              'Police conflict and executions',
                              },
                  label='Other Injuries')
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'Other Injuries':
            Cause(gbd_causes={'Interpersonal violence',
                              'Fire, heat, and hot substances',
                              'Falls',
                              'Exposure to mechanical forces',
                              'Drowning',
                              'Exposure to forces of nature',
                              'Foreign body',
                              'Other transport injuries',
                              'Other unintentional injuries',
                              'Poisonings',
                              'Police conflict and executions'},
                  label='Other Injuries')
    }

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        pass


class Backpain(Module):
    METADATA = {Metadata.DISEASE_MODULE}
    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'chronic_lower_back_pain':
            Cause(gbd_causes={'Low back pain'},
                  label='Low back pain')
    }

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        pass

    def initialise_simulation(self, sim):
        pass


dummies = [
    Tb(),
    Alri(),
    RoadTrafficInjuries(),
    Copd(),
    Schisto(),
    OtherInjuries(),
    Backpain()
]

# %% Create simulation, run simulation and get the mappers from the log files:
sim = Simulation(start_date=Date(2010, 1, 1))
sim.register(*complete)
sim.register(*dummies)

# Modify how 'OtherCancer' is labelled:
list(sim.modules['OtherAdultCancer'].CAUSES_OF_DEATH.values())[0].label = 'Other Cancer'
list(sim.modules['OtherAdultCancer'].CAUSES_OF_DISABILITY.values())[0].label = 'Other Cancer'

sim.make_initial_population(n=100)
sim.simulate(end_date=Date(2010, 1, 1))

_, deaths_mapper_from_gbd_causes = \
    sim.modules['Demography'].create_mappers_from_causes_of_death_to_label()
_, dalys_mapper_from_gbd_causes = \
    sim.modules['HealthBurden'].create_mappers_from_causes_of_disability_to_label()

# %% Load GBD data:
gbd = format_gbd(pd.read_csv(resourcefilepath / "gbd" / "ResourceFile_Deaths_And_DALYS_GBD2019.csv"))

# extract total death (all ages/sex)
deaths = gbd.loc[(gbd.measure_name == 'Deaths') & (gbd.Year == 2019)].copy().groupby(by='cause_name')['GBD_Est'].sum()
deaths = pd.DataFrame(deaths / deaths.sum())

# extract total dalys (all age/sex)
dalys = gbd.loc[(gbd.measure_name == 'DALYs (Disability-Adjusted Life Years)') & (gbd.Year == 2019)].copy().groupby(
    by='cause_name')['GBD_Est'].sum()
dalys = pd.DataFrame(dalys / dalys.sum())

# %% Label TLO causes of death accordingly
level2_conds = [
    'Depression / Self-harm',
    'Epilepsy',
    'Kidney Disease',
    'Other Cancer',
    'Other Injuries',
    'Schistosomiasis',
    'Chronic obstructive pulmonary disease'
]

# TLO Cause
deaths['TLO'] = deaths.index.map(deaths_mapper_from_gbd_causes)

# Level of TLO Cause
deaths['TLO_Level'] = 3
deaths['TLO_Level'].loc[deaths['TLO'].isin(['Other'])] = 1
deaths['TLO_Level'].loc[deaths['TLO'].isin([level2_conds])] = 2

deaths.groupby('TLO')['GBD_Est'].sum()

# %% # %% Label TLO causes of disability accordingly

# TLO Cause
dalys['TLO'] = dalys.index.map(dalys_mapper_from_gbd_causes)
dalys['TLO_Level'] = 3
dalys['TLO_Level'].loc[dalys['TLO'].isin(level2_conds)] = 2
dalys['TLO_Level'].loc[dalys['TLO'].isin(['Other'])] = 1


# %% Get proportion of death according to each category
props_deaths = deaths.groupby(by='TLO_Level')['GBD_Est'].sum()
props_dalys = dalys.groupby(by='TLO_Level')['GBD_Est'].sum()

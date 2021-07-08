"""
A run of the full model at scale (20k population size) using all disease modules currently included in Master for 20
years, including all the modules for birth/labour/newborn outcome

* All logging
* Script including parsing logfile

For use in profiling.

"""
from pathlib import Path

import pandas as pd
import shared

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    diarrhoea,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    epi,
    epilepsy,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    malaria,
    newborn_outcomes,
    oesophagealcancer,
    other_adult_cancers,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)

# Key parameters about the simulation:
start_date = Date(2010, 1, 1)
end_date = start_date + pd.DateOffset(years=20)

popsize = 20_000

# The resource files
resourcefilepath = Path("./resources")

log_config = {
    "filename": "for_profiling",
    "directory": "./outputs",
    "custom_levels": {"*": logging.INFO}
}

sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

# Register the appropriate modules
sim.register(
    # Standard modules:
    demography.Demography(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),

    symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=True),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),

    # HealthSystem
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                              mode_appt_constraints=2,
                              capabilities_coefficient=0.01
                              ),
    dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
    dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),

    # Modules for birth/labour/newborns
    contraception.Contraception(resourcefilepath=resourcefilepath),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
    care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
    labour.Labour(resourcefilepath=resourcefilepath),
    newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
    postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),

    # Disease modules considered complete:
    cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),
    depression.Depression(resourcefilepath=resourcefilepath),
    diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
    epi.Epi(resourcefilepath=resourcefilepath),
    epilepsy.Epilepsy(resourcefilepath=resourcefilepath),
    hiv.Hiv(resourcefilepath=resourcefilepath),
    malaria.Malaria(resourcefilepath=resourcefilepath),
    oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath),
    other_adult_cancers.OtherAdultCancer(resourcefilepath=resourcefilepath)
)

# Run the simulation
sim.make_initial_population(n=popsize)
shared.schedule_profile_log(sim)
sim.simulate(end_date=end_date)
shared.print_checksum(sim)

# Parse the log-file
log_df = parse_log_file(sim.log_filepath)

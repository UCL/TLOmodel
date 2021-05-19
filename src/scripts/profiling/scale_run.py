"""
A run of the model at scale using all disease modules currently included in Master - with no logging

For use in profiling.
"""
from pathlib import Path

import pandas as pd
import shared

from tlo import Date, Simulation, logging
from tlo.methods import (
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
    labour,
    malaria,
    newborn_outcomes,
    oesophagealcancer,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)

# Key parameters about the simulation:
start_date = Date(2010, 1, 1)
end_date = start_date + pd.DateOffset(years=2)

pop_size = 10_000

# The resource files
rfp = Path("./resources")

log_config = {
    "filename": "for_profiling",
    "directory": "./outputs",
    "custom_levels": {"*": logging.WARNING}
}

sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

# Register the appropriate modules
sim.register(
    # Standard modules:
    demography.Demography(resourcefilepath=rfp),
    enhanced_lifestyle.Lifestyle(resourcefilepath=rfp),
    healthsystem.HealthSystem(resourcefilepath=rfp),
    symptommanager.SymptomManager(resourcefilepath=rfp),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=rfp),
    healthburden.HealthBurden(resourcefilepath=rfp),
    contraception.Contraception(resourcefilepath=rfp),
    newborn_outcomes.NewbornOutcomes(resourcefilepath=rfp),
    postnatal_supervisor.PostnatalSupervisor(resourcefilepath=rfp),
    labour.Labour(resourcefilepath=rfp),
    care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=rfp),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=rfp),
    dx_algorithm_child.DxAlgorithmChild(resourcefilepath=rfp),
    dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=rfp),
    #
    # Disease modules considered complete:
    diarrhoea.Diarrhoea(resourcefilepath=rfp),
    malaria.Malaria(resourcefilepath=rfp),
    epi.Epi(resourcefilepath=rfp),
    depression.Depression(resourcefilepath=rfp),
    oesophagealcancer.OesophagealCancer(resourcefilepath=rfp),
    epilepsy.Epilepsy(resourcefilepath=rfp)
)


# Run the simulation
sim.make_initial_population(n=pop_size)
shared.schedule_profile_log(sim)
sim.simulate(end_date=end_date)
shared.print_checksum(sim)

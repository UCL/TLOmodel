import os
from pathlib import Path

import pytest

from tlo import Date, Simulation
from tlo.methods import cardio_metabolic_disorders, demography, enhanced_lifestyle, simplified_births, healthsystem, \
    symptommanager, healthseekingbehaviour, healthburden, diabetic_retinopathy, depression

resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
start_date = Date(2010, 1, 1)
end_date = Date(2010, 1, 2)
@pytest.mark.slow
def test_basic_run(tmpdir, seed, diabetes_retinopathy=None):
    """Run the simulation with the Copd module and read the log from the Copd module."""

    popsize = 1000
    sim = Simulation(
        start_date=start_date,
        seed=seed,
        log_config={
            'filename': 'temp',
            'directory': tmpdir,
        },
    )

    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=False,
                                           cons_availability='all'),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath,
                                                               # force symptoms to lead to health care seeking:
                                                               force_any_symptom_to_lead_to_healthcareseeking=True
                                                               ),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),
                 diabetic_retinopathy.Diabetic_Retinopathy(),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 )
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=Date(2010, 2, 1)) # Long run

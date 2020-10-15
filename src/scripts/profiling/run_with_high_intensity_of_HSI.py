"""
A run of the model that has a lot of HSI's being run:

* Parameters for modules are artificially increased so that there is are lots of HSI

* Use the elastic mode of constraints (all HSI run but some with squeeze factors)

* Spurious symptoms set to True in the SymptomManager

* Any symptom leads to an HSI (force_any_symptom_to_lead_to_healthcareseeking in HealthCareSeeking module)

For use in profiling.
"""
from tlo.lm import LinearModel, LinearModelType

"""
A run of the model at scale using all disease modules currently included in Master - with no logging

For use in profiling.
"""

from pathlib import Path

import pandas as pd

from tlo import Date, Simulation, logging
from tlo.methods import (
    contraception,
    demography,
    depression,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    pregnancy_supervisor,
    symptommanager,
    oesophagealcancer,
    malaria,
    epi,
    epilepsy,
    dx_algorithm_adult, diarrhoea,
)

# Key parameters about the simulation:
start_date = Date(2010, 1, 1)
end_date = start_date + pd.DateOffset(years=1)

popsize = int(100)

# The resource files
resourcefilepath = Path("./resources")

log_config = {
    "filename": "for_profiling",
    "directory": "./outputs",
    "custom_levels": {
        "*": logging.WARNING,
        'healthsystem': logging.INFO
    }
}

sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

# Register the appropriate modules
sim.register(
    # Standard modules:
    demography.Demography(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                              mode_appt_constraints=0
                              ),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath,
                                  spurious_symptoms=True),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath,
                                                  force_any_symptom_to_lead_to_healthcareseeking=True),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    contraception.Contraception(resourcefilepath=resourcefilepath),
    labour.Labour(resourcefilepath=resourcefilepath),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
    dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
    dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
    #
    # Disease modules considered complete:
    diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
    malaria.Malaria(resourcefilepath=resourcefilepath, testing=1),
    epi.Epi(resourcefilepath=resourcefilepath),
    depression.Depression(resourcefilepath=resourcefilepath),
    oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath),
    epilepsy.Epilepsy(resourcefilepath=resourcefilepath)
)

# Adjust parameters so that there are lots of HSI events:

# * Diarrhoea
for param_name in sim.modules['Diarrhoea'].parameters.keys():
    # Increase incidence:
    if param_name.startswith('base_inc_rate_diarrhoea_by_'):
        sim.modules['Diarrhoea'].parameters[param_name] = \
            [4.0 * v for v in sim.modules['Diarrhoea'].parameters[param_name]]

    # Increase symptoms:
    if param_name.startswith('proportion_AWD_by_'):
        sim.modules['Diarrhoea'].parameters[param_name] = 1.0
    if param_name.startswith('fever_by_'):
        sim.modules['Diarrhoea'].parameters[param_name] = 1.0
    if param_name.startswith('vomiting_by_'):
        sim.modules['Diarrhoea'].parameters[param_name] = 1.0
    if param_name.startswith('dehydration_by_'):
        sim.modules['Diarrhoea'].parameters[param_name] = 1.0


#   * Depression
sim.modules['Depression'].parameters['prob_3m_selfharm_depr'] = 0.25
sim.modules['Depression'].linearModels['Risk_of_SelfHarm_per3mo'] = LinearModel(
    LinearModelType.MULTIPLICATIVE,
    sim.modules['Depression'].parameters['prob_3m_selfharm_depr']
)

# * Oesophageal Cancer
sim.modules['OesophagealCancer'].parameters['init_prop_oes_cancer_stage'] = [0.1] * 6
sim.modules['OesophagealCancer'].parameters['r_high_grade_dysplasia_low_grade_dysp'] *= 5
sim.modules['OesophagealCancer'].parameters['r_stage1_high_grade_dysp'] *= 5
sim.modules['OesophagealCancer'].parameters['r_stage2_stage1'] *= 5
sim.modules['OesophagealCancer'].parameters['r_stage3_stage2'] *= 5
sim.modules['OesophagealCancer'].parameters['r_stage4_stage3'] *= 5

# * Malaria
#  Set 'malaria_testing=1' in module when created


# Run the simulation
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)









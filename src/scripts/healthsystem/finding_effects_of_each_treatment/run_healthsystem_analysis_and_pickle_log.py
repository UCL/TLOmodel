"""
A run of the model with logging so as to allow for descriptions of overall Health Burden and usage of the Health System.
TODO -- have turned off spurious_symptoms to make it go faster, but this should be on really!
"""
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
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
    oesophagealcancer,
    pregnancy_supervisor,
    symptommanager,
)

# Define path and filenames
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.today().strftime("%Y_%m_%d")
results_filename_stub = 'health_system_systematic_run'
results_filename = outputpath / f"{datestamp}_{results_filename_stub}.pickle"

# Key parameters about the simulation:
start_date = Date(2010, 1, 1)
end_date = start_date + pd.DateOffset(years=4)
pop_size = 20_000

# The resource files
rfp = Path("./resources")

log_config = {
    "filename": "for_profiling",
    "directory": "./outputs",
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.healthsystem": logging.INFO,
        "tlo.methods.healthburden": logging.INFO,
        "tlo.methods.demography": logging.INFO
    }
}


def run_sim(service_availability):
    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

    # Register the appropriate modules
    sim.register(
        # Standard modules:
        demography.Demography(resourcefilepath=rfp),
        enhanced_lifestyle.Lifestyle(resourcefilepath=rfp),
        healthsystem.HealthSystem(resourcefilepath=rfp, service_availability=service_availability),
        symptommanager.SymptomManager(resourcefilepath=rfp, spurious_symptoms=False),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=rfp),
        healthburden.HealthBurden(resourcefilepath=rfp),
        contraception.Contraception(resourcefilepath=rfp),
        labour.Labour(resourcefilepath=rfp),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=rfp),
        #
        # Disease modules considered complete:
        diarrhoea.Diarrhoea(resourcefilepath=rfp),
        malaria.Malaria(resourcefilepath=rfp),
        hiv.Hiv(resourcefilepath=rfp),
        epi.Epi(resourcefilepath=rfp),
        depression.Depression(resourcefilepath=rfp),
        oesophagealcancer.OesophagealCancer(resourcefilepath=rfp),
        epilepsy.Epilepsy(resourcefilepath=rfp)
    )
    # Run the simulation
    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)

    # Return the parsed log-file
    return parse_log_file(sim.log_filepath)


# %% Define scenarios for the parameter 'service_availability'

# scoop-up all the TREATMENT_IDs that are defined in the code
# (todo - automate this in future)
# (todo - enforce the treatment_id convention of them beginning with the name of the module)
# (todo - or consider letting the gating be at the level of the disease module or not rely on the naming)

treatment_ids = [
    'CareOfWomenDuringPregnancy_PresentsForFirstAntenatalCareVisit',
    'CareOfWomenDuringPregnancy_PresentsForSubsequentAntenatalCareVisit',
    'CareOfWomenDuringPregnancy_EmergencyTreatment',
    'CareOfWomenDuringPregnancy_PresentsForPostAbortionCare',
    'CareOfWomenDuringPregnancy_TreatmentFollowingAntepartumStillbirth',
    "BladderCancer_Investigation_Following_blood_urine",
    "BladderCancer_Investigation_Following_pelvic_pain",
    "BladderCancer_StartTreatment",
    "BladderCancer_MonitorTreatment",
    "BladderCancer_PalliativeCare",
    'Depression_TalkingTherapy',
    'Depression_Antidepressant_Start',
    'Depression_Antidepressant_Refill',
    'Diarrhoea_Treatment_PlanA',
    'Diarrhoea_Treatment_PlanB',
    'Diarrhoea_Treatment_PlanC',
    'Diarrhoea_Severe_Persistent_Diarrhoea',
    'Diarrhoea_Non_Severe_Persistent_Diarrhoea',
    'Diarrhoea_Dysentery',
    "Epi_bcg",
    "Epi_opv",
    "Epi_DtpHibHep",
    "Epi_Rota",
    "Epi_MeaslesRubella",
    "Epi_hpv",
    "Epi_Td",
    'Epilepsy_Start_Anti-Epilpetics',
    "Hiv_TestAndRefer",
    "Hiv_Circumcision",
    "Hiv_StartOrContinueOnPrep",
    "Hiv_Treatment_InitiationOrContinuation",
    'GenericFirstApptAtFacilityLevel1',
    'GenericFirstApptAtFacilityLevel0',
    'GenericEmergencyFirstApptAtFacilityLevel1',
    'Labour_PresentsForSkilledAttendanceInLabour',
    'Labour_ReceivesCareForPostpartumPeriod',
    'Labour_CaesareanSection',
    'Labour_ReceivesBloodTransfusion',
    'Labour_SurgeryForLabourComplicationsFacilityLevel1',
    "Malaria_RDT",
    "Malaria_treatment_child0_5",
    "Malaria_treatment_child5_15",
    "Malaria_treatment_adult",
    "Malaria_treatment_complicated_child",
    "Malaria_treatment_complicated_adult",
    "Malaria_IPTp",
    'NewbornOutcomes_ReceivesSkilledAttendance',
    'NewbornOutcomes_NeonateInpatientDay',
    "OesophagealCancer_Investigation_Following_Dysphagia",
    "OesophagealCancer_StartTreatment",
    "OesophagealCancer_MonitorTreatment",
    "OesophagealCancer_PalliativeCare"
]

stubs = {s.split('_')[0] for s in treatment_ids}

scenarios = {
    'Nothing': [],
    'Everything': ['*'],
}

# create scenarios in which only one of the stubs is permitted.
# Note that 'wildcard is needed to allow all Treatment_IDs with that stub.
for s in stubs:
    scenarios[f"Only {s}"] = [f"{st}*" for st in stubs if st == s]

# %% Run the model
results = dict()
for name, serv_av in scenarios.items():
    results[name] = run_sim(service_availability=serv_av)

# %% Pickle the results dict
with open(results_filename, 'wb') as f:
    pickle.dump({'results': results}, f, pickle.HIGHEST_PROTOCOL)

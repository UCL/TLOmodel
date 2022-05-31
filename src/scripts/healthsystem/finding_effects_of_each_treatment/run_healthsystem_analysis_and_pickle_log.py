"""
A run of the model with logging so as to allow for descriptions of overall Health Burden and usage of the Health System.

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
    symptommanager, rti, prostate_cancer, other_adult_cancers, breast_cancer, bladder_cancer,
    cardio_metabolic_disorders, measles, wasting, stunting, alri, postnatal_supervisor, newborn_outcomes,
    care_of_women_during_pregnancy,
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
resourcefilepath = Path("./resources")

log_config = {
    "filename": "for_profiling",
    "directory": "./outputs",
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.healthburden": logging.INFO,
        "tlo.methods.demography": logging.INFO
    }
}


def run_sim(service_availability):
    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

    # Register the appropriate modules
    sim.register(
        # Standard modules:
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(
            resourcefilepath=resourcefilepath,
            spurious_symptoms=False,
        ),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),

        # HealthSystem
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            service_availability=service_availability
        ),
        epi.Epi(resourcefilepath=resourcefilepath),

        # Modules for birth/labour/newborns
        contraception.Contraception(resourcefilepath=resourcefilepath),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
        care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
        labour.Labour(resourcefilepath=resourcefilepath),
        newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
        postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),

        # Diseases of childhood
        alri.Alri(resourcefilepath=resourcefilepath),
        diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
        stunting.Stunting(resourcefilepath=resourcefilepath),
        wasting.Wasting(resourcefilepath=resourcefilepath),

        # Major Infectious Diseases
        hiv.Hiv(resourcefilepath=resourcefilepath),
        malaria.Malaria(resourcefilepath=resourcefilepath),
        measles.Measles(resourcefilepath=resourcefilepath),

        # Chronic Conditions
        #  - Cardio-metabolic
        cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),

        # - Cancers
        bladder_cancer.BladderCancer(resourcefilepath=resourcefilepath),
        breast_cancer.BreastCancer(resourcefilepath=resourcefilepath),
        oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath),
        other_adult_cancers.OtherAdultCancer(resourcefilepath=resourcefilepath),
        prostate_cancer.ProstateCancer(resourcefilepath=resourcefilepath),

        # - Mental Health
        depression.Depression(resourcefilepath=resourcefilepath),
        epilepsy.Epilepsy(resourcefilepath=resourcefilepath),

        # Other
        rti.RTI(resourcefilepath=resourcefilepath)
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
    "Alri_GenericTreatment",
    "BladderCancer_Investigation_Following_blood_urine",
    "BladderCancer_Investigation_Following_pelvic_pain",
    "BladderCancer_MonitorTreatment",
    "BladderCancer_PalliativeCare",
    "BladderCancer_StartTreatment",
    "BreastCancer_Investigation_Following_breast_lump_discernible",
    "BreastCancer_MonitorTreatment",
    "BreastCancer_PalliativeCare",
    "breastCancer_StartTreatment",
    "CardioMetabolicDisorders_CommunityTestingForHypertension",
    "CardioMetabolicDisorders_InvestigationNotFollowingSymptoms",
    "CardioMetabolicDisorders_Investigation_Following_Symptoms",
    "CardioMetabolicDisorders_Refill_Medication",
    "CardioMetabolicDisorders_SeeksEmergencyCareAndGetsTreatment",
    "CardioMetabolicDisorders_StartWeightLossAndMedication",
    "CareOfWomenDuringPregnancy_AntenatalOutpatientFollowUp",
    "CareOfWomenDuringPregnancy_AntenatalOutpatientManagementOfAnaemia",
    "CareOfWomenDuringPregnancy_AntenatalWardInpatientCare",
    "CareOfWomenDuringPregnancy_EighthAntenatalCareContact",
    "CareOfWomenDuringPregnancy_FifthAntenatalCareContact",
    "CareOfWomenDuringPregnancy_FirstAntenatalCareContact",
    "CareOfWomenDuringPregnancy_FourthAntenatalCareContact",
    "CareOfWomenDuringPregnancy_MaternalEmergencyAssessment",
    "CareOfWomenDuringPregnancy_PostAbortionCaseManagement",
    "CareOfWomenDuringPregnancy_PresentsForInductionOfLabour",
    "CareOfWomenDuringPregnancy_SecondAntenatalCareContact",
    "CareOfWomenDuringPregnancy_SeventhAntenatalCareContact",
    "CareOfWomenDuringPregnancy_SixthAntenatalCareContact",
    "CareOfWomenDuringPregnancy_ThirdAntenatalCareContact",
    "CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy",
    "Contraception_FamilyPlanningAppt",
    "Depression_Antidepressant_Refill",
    "Depression_Antidepressant_Start",
    "Depression_TalkingTherapy",
    "Diarrhoea_Treatment_Inpatient",
    "Diarrhoea_Treatment_Outpatient",
    "Epi_DtpHibHep",
    "Epi_MeaslesRubella",
    "Epi_Pneumo",
    "Epi_Rota",
    "Epi_Td",
    "Epi_bcg",
    "Epi_hpv",
    "Epi_opv",
    "Epilepsy_Start_Anti - Epilpetics",
    # "GenericEmergencyFirstApptAtFacilityLevel1",
    # "GenericFirstApptAtFacilityLevel0",
    "Hiv_Circumcision",
    "Hiv_StartOrContinueOnPrep",
    "Hiv_TestAndRefer",
    "Hiv_Treatment_InitiationOrContinuation",
    "Labour_ReceivesCareFollowingCaesareanSection",
    "Labour_ReceivesComprehensiveEmergencyObstetricCare",
    "Labour_ReceivesSkilledBirthAttendanceDuringLabour",
    "Labour_ReceivesSkilledBirthAttendanceFollowingLabour",
    "Malaria_IPTp",
    "Malaria_RDT",
    "Malaria_treatment_adult",
    "Malaria_treatment_child0_5",
    "Malaria_treatment_child5_15",
    "Malaria_treatment_complicated_adult",
    "Malaria_treatment_complicated_child",
    "Measles_Treatment",
    "NewbornOutcomes_CareOfTheNewbornBySkilledAttendant",
    "OesophagealCancer_Investigation_Following_Dysphagia",
    "OesophagealCancer_MonitorTreatment",
    "OesophagealCancer_PalliativeCare",
    "OesophagealCancer_StartTreatment",
    "OtherAdultCancer_Investigation_Following_other_adult_ca_symptom",
    "OtherAdultCancer_MonitorTreatment",
    "OtherAdultCancer_PalliativeCare",
    "OtherAdultCancer_StartTreatment",
    "PostnatalSupervisor_NeonatalWardInpatientCare",
    "PostnatalSupervisor_PostnatalCareContactOne",
    "PostnatalSupervisor_PostnatalCareContactTwoMaternal",
    "PostnatalSupervisor_PostnatalWardInpatientCare",
    "PostnatalSupervisor_TreatmentForObstetricFistula",
    "ProstateCancer_Investigation_Following_blood_urine",
    "ProstateCancer_Investigation_Following_pelvic_pain",
    "ProstateCancer_Investigation_Following_psa_positive",
    "ProstateCancer_MonitorTreatment",
    "ProstateCancer_PalliativeCare",
    "ProstateCancer_StartTreatment",
    "RTI_Acute_Pain_Management",
    "RTI_Burn_Management",
    "RTI_Fracture_Cast",
    "RTI_Imaging_Event",
    "RTI_Major_Surgeries",
    "RTI_MedicalIntervention",
    "RTI_Minor_Surgeries",
    "RTI_Open_Fracture_Treatment",
    "RTI_Shock_Treatment",
    "RTI_Suture",
    "RTI_Tetanus_Vaccine",
    "Complementary_feeding_for_stunting"
]

stubs = {s.split('_')[0] for s in treatment_ids}

scenarios = {
    'Nothing': [],
    'Everything': ['*'],
}

# create scenarios in which only one of the stubs is permitted.
# Note that 'wildcard is needed to allow all Treatment_IDs with that stub.
# todo - turn this around so that it's about the lost effect when that thing is NOT running.
for s in stubs:
    scenarios[f"Only {s}"] = [f"{st}*" for st in stubs if st == s]

# %% Run the model
results = dict()
for name, serv_av in scenarios.items():
    results[name] = run_sim(service_availability=serv_av)

# %% Pickle the results dict
with open(results_filename, 'wb') as f:
    pickle.dump({'results': results}, f, pickle.HIGHEST_PROTOCOL)

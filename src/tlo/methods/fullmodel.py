from pathlib import Path
from typing import List, Optional

from tlo import Module
from tlo.methods import (
    alri,
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
    rti,
    schisto,
    simplified_births,
    stunting,
    symptommanager,
    wasting,
)


def fullmodel(
    resourcefilepath: Path,
    use_simplified_births: Optional[bool] = False,
    symptommanager_spurious_symptoms: Optional[bool] = True,
    healthsystem_disable: Optional[bool] = False,
    healthsystem_mode_appt_constraints: Optional[int] = 1,
    healthsystem_capabilities_coefficient: Optional[float] = 1.0,
    healthsystem_record_hsi_event_details: Optional[bool] = False
) -> List[Module]:
    """Return the modules that should be registered in a run of the `Full Model`."""

    all_modules = []

    # Standard modules:
    all_modules.extend([
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(
            resourcefilepath=resourcefilepath,
            spurious_symptoms=symptommanager_spurious_symptoms),
    ])

    # HealthSystem and the Expanded Programme on Immunizations
    all_modules.extend([
        epi.Epi(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(
            resourcefilepath=resourcefilepath,
            disable=healthsystem_disable,
            mode_appt_constraints=healthsystem_mode_appt_constraints,
            capabilities_coefficient=healthsystem_capabilities_coefficient,
            record_hsi_event_details=healthsystem_record_hsi_event_details),
    ])

    # Contraception, Pregnancy, Labour, etc. (or SimplifiedBirths)
    if use_simplified_births:
        all_modules.extend([
            simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        ])
    else:
        all_modules.extend([
            contraception.Contraception(resourcefilepath=resourcefilepath, use_healthsystem=True),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
            care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
            labour.Labour(resourcefilepath=resourcefilepath),
            newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
            postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath)
        ])

    # Conditions of Early Childhood
    all_modules.extend([
        alri.Alri(resourcefilepath=resourcefilepath),
        diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
        stunting.Stunting(resourcefilepath=resourcefilepath),
        wasting.Wasting(resourcefilepath=resourcefilepath),
    ])

    # Communicable Diseases
    all_modules.extend([
        hiv.Hiv(resourcefilepath=resourcefilepath),
        malaria.Malaria(resourcefilepath=resourcefilepath),
        measles.Measles(resourcefilepath=resourcefilepath),
        schisto.Schisto(resourcefilepath=resourcefilepath)
        # tb.TB(resourcefilepath=resourcefilepath)  <-- awaiting PR #541
    ])

    # Non-Communicable Conditions
    #  - Cancers
    all_modules.extend([
        bladder_cancer.BladderCancer(resourcefilepath=resourcefilepath),
        breast_cancer.BreastCancer(resourcefilepath=resourcefilepath),
        oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath),
        other_adult_cancers.OtherAdultCancer(resourcefilepath=resourcefilepath),
        prostate_cancer.ProstateCancer(resourcefilepath=resourcefilepath),
    ])

    #  - Cardio-metabolic Disorders
    all_modules.extend([
        cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath)
    ])

    #  - Injuries
    all_modules.extend([
        rti.RTI(resourcefilepath=resourcefilepath)
    ])

    #  - Other Non-Communicable Conditions
    all_modules.extend([
        depression.Depression(resourcefilepath=resourcefilepath),
        epilepsy.Epilepsy(resourcefilepath=resourcefilepath)
    ])

    return all_modules

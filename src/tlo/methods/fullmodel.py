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
    tb,
    wasting,
)


def fullmodel(
    resourcefilepath: Path,
    use_simplified_births: Optional[bool] = False,
    module_kwargs=None,
) -> List[Module]:
    """Return the modules that should be registered in a run of the `Full Model`."""
    if module_kwargs is None:
        module_kwargs = {
            "SymptomManager": {"spurious_symptoms": True},
            "HealthSystem": {"mode_appt_constraints": 1},
        }
    module_classes = [
        # Standard modules
        demography.Demography,
        enhanced_lifestyle.Lifestyle,
        healthburden.HealthBurden,
        healthseekingbehaviour.HealthSeekingBehaviour,
        symptommanager.SymptomManager,
        # HealthSystem and the Expanded Programme on Immunizations
        epi.Epi,
        healthsystem.HealthSystem,
        # Conditions of Early Childhood
        alri.Alri,
        diarrhoea.Diarrhoea,
        stunting.Stunting,
        wasting.Wasting,
        # Communicable Diseases
        hiv.Hiv,
        malaria.Malaria,
        measles.Measles,
        schisto.Schisto,
        tb.Tb,
        # Non-Communicable Conditions
        #  - Cancers
        bladder_cancer.BladderCancer,
        breast_cancer.BreastCancer,
        oesophagealcancer.OesophagealCancer,
        other_adult_cancers.OtherAdultCancer,
        prostate_cancer.ProstateCancer,
        #  - Cardio-metabolic Disorders
        cardio_metabolic_disorders.CardioMetabolicDisorders,
        #  - Injuries
        rti.RTI,
        #  - Other Non-Communicable Conditions
        depression.Depression,
        epilepsy.Epilepsy,
    ]
    # Contraception, Pregnancy, Labour, etc. (or SimplifiedBirths)
    if use_simplified_births:
        module_classes.append(simplified_births.SimplifiedBirths)
    else:
        module_classes += [
            contraception.Contraception,
            pregnancy_supervisor.PregnancySupervisor,
            care_of_women_during_pregnancy.CareOfWomenDuringPregnancy,
            labour.Labour,
            newborn_outcomes.NewbornOutcomes,
            postnatal_supervisor.PostnatalSupervisor,
        ]
    return [
        module_class(
            resourcefilepath=resourcefilepath,
            **module_kwargs.get(module_class.__name__, {})
        )
        for module_class in module_classes
    ]

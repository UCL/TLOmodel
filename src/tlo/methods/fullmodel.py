from typing import Dict, List, Optional

from tlo import Module
from tlo.methods import (
    alri,
    bladder_cancer,
    breast_cancer,
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    cervical_cancer,
    contraception,
    copd,
    demography,
    depression,
    diarrhoea,
    emulated_rti,
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
    use_simplified_births: bool = False,
    use_emulated_RTI = False,
    module_kwargs: Optional[Dict[str, Dict]] = {},
) -> List[Module]:
    """Return a list of modules that should be registered in a run of the full model.

    :param resourcefilepath: Path to root of directory containing resource files.
    :param use_simplified_births: Whether to use ``SimplifiedBirths`` module in place
        of full pregnancy related modules.
    :param use_emulated_RTI: Option to use emulated version of RTI module
    :param module_kwargs: Dictionary mapping from module class names to dictionaries of
        keyword argument names and values to set for the module. If ``None`` (the
        default), the default values for all module keyword arguments are used other
        than ``spurious_symptoms`` being set to ``True`` in ``SymptomManager`` and
        ``mode_appt_constraints`` being set to ``1`` in ``HealthSystem``.
    :return: List of initialised modules that can be passed to ``Simulation.register``
        method.

    :Example:

    The following would initialise all modules in the full model with the ``disable``
    argument to the ``HealthSystem`` module set to ``True``

    >>> from tlo.methods.fullmodel import fullmodel
    >>> modules = fullmodel(
    >>>     module_kwargs={"HealthSystem": {"disable": True}},
    >>> )
    """

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
        # Contraception, Pregnancy, Labour, etc. (or SimplifiedBirths)
        *(
            [simplified_births.SimplifiedBirths] if use_simplified_births else
            [
                contraception.Contraception,
                pregnancy_supervisor.PregnancySupervisor,
                care_of_women_during_pregnancy.CareOfWomenDuringPregnancy,
                labour.Labour,
                newborn_outcomes.NewbornOutcomes,
                postnatal_supervisor.PostnatalSupervisor,
            ]
        ),
        *(
            [emulated_rti.EmulatedRTI] if use_emulated_RTI else
            [
                rti.RTI
            ]
        ),
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
        cervical_cancer.CervicalCancer,
        oesophagealcancer.OesophagealCancer,
        other_adult_cancers.OtherAdultCancer,
        prostate_cancer.ProstateCancer,
        #  - Cardio-metabolic Disorders
        cardio_metabolic_disorders.CardioMetabolicDisorders,
        #  - Other Non-Communicable Conditions
        copd.Copd,
        depression.Depression,
        epilepsy.Epilepsy,
    ]
    return [
        module_class(
            **module_kwargs.get(module_class.__name__, {})
        )
        for module_class in module_classes
    ]

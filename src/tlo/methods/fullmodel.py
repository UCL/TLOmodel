from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from tlo import Date, Module, Simulation
from tlo.analysis.utils import get_root_path, parse_log_file
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
    use_simplified_births: bool = False,
    module_kwargs: Optional[Dict[str, Dict]] = None,
) -> List[Module]:
    """Return a list of modules that should be registered in a run of the full model.

    :param resourcefilepath: Path to root of directory containing resource files.
    :param use_simplified_births: Whether to use ``SimplifiedBirths`` module in place
        of full pregnancy related modules.
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
    >>> resourcefilepath = ...
    >>> modules = fullmodel(
    >>>     resourcefilepath,
    >>>     module_kwargs={"HealthSystem": {"disable": True}},
    >>> )
    """
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
    return [
        module_class(
            resourcefilepath=resourcefilepath,
            **module_kwargs.get(module_class.__name__, {})
        )
        for module_class in module_classes
    ]


def get_mappers_in_fullmodel():
    """Returns the cause-of-death, cause-of-disability and cause-of-DALYS mappers that are created in a run of the
    fullmodel."""
    root = get_root_path()
    tmpdir = root / 'outputs'
    resourcefilepath = root / 'resources'

    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=0, log_config={'filename': 'test_log', 'directory': tmpdir})
    sim.register(*fullmodel(resourcefilepath=resourcefilepath))
    sim.make_initial_population(n=10_000)
    sim.simulate(end_date=start_date)
    demog_log = parse_log_file(sim.log_filepath)['tlo.methods.demography']
    hb_log = parse_log_file(sim.log_filepath)['tlo.methods.healthburden']

    keys = [
        (demog_log, 'mapper_from_tlo_cause_to_common_label'),
        (demog_log, 'mapper_from_gbd_cause_to_common_label'),
        (hb_log, 'disability_mapper_from_tlo_cause_to_common_label'),
        (hb_log, 'disability_mapper_from_gbd_cause_to_common_label'),
        (hb_log, 'daly_mapper_from_gbd_cause_to_common_label'),
        (hb_log, 'daly_mapper_from_tlo_cause_to_common_label'),
    ]

    def extract_mapper(key_tuple):
        return pd.Series(key_tuple[0].get(key_tuple[1]).drop(columns={'date'}).loc[0]).to_dict()

    return {k[1]: extract_mapper(k) for k in keys}

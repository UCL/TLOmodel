from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.cancer_consumables import get_consumable_item_codes_cancers
from tlo.methods.causes import Cause
from tlo.methods.demography import InstantaneousDeath
from tlo.methods.dxmanager import DxTest
from tlo.methods.hsi_event import HSI_Event
from tlo.methods.hsi_generic_first_appts import GenericFirstAppointmentsMixin
from tlo.methods.symptommanager import Symptom
from tlo.util import read_csv_files

if TYPE_CHECKING:
    from tlo.methods.hsi_generic_first_appts import HSIEventScheduler
    from tlo.population import IndividualProperties

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Influenza(Module):
    """Influenza module for simulating influenza events in a population."""

    def __init__(self, name=None):
        super().__init__(name)

    INIT_DEPENDENCIES = {'Demography', 'HealthSystem', 'SymptomManager'}

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden'}

    # Declare Causes of Death
    CAUSES = {
        "Influenza": Cause(gbd_causes="Influenza", label="Influenza" )
    }

    PARAMETERS = {
        "r_infection": Parameter(Types.REAL, "Probability per month of infection"),
        "r_symptomatic": Parameter(Types.REAL, "Probability infection → symptoms"),
        "r_recovery": Parameter(Types.REAL, "Probability recovery from symptoms"),
        "r_death": Parameter(Types.REAL, "Probability death if symptomatic"),
    }

    PROPERTIES = {
        "flu_status": Property(
            Types.CATEGORICAL,
            "Stage of influenza disease",
            categories=["none", "infected", "symptomatic", "recovered"],
        ),
        "flu_date_infection": Property(Types.DATE, "Date of infection"),
        "flu_date_symptoms": Property(Types.DATE, "Date symptoms started"),
        "flu_date_recovery": Property(Types.DATE, "Date recovered"),
        "flu_date_death": Property(Types.DATE, "Date of influenza death"),
    }

    def read_parameters(self, resourcefilepath: Optional[Path] = None):
        # Register fever symptom
        self.sim.modules["SymptomManager"].register_symptom(
            Symptom(name="fever", odds_ratio_health_seeking_in_adults=2.0)
        )
    pass

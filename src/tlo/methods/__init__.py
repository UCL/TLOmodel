from enum import Enum, auto


class Metadata(Enum):
    DISEASE_MODULE = (
        auto()
    )  # Disease modules: Any disease module should carry this label.
    USES_SYMPTOMMANAGER = (
        auto()
    )  # The 'Symptom Manager' recognises modules with this label.
    USES_HEALTHSYSTEM = auto()  # The 'HealthSystem' recognises modules with this label.
    USES_HEALTHBURDEN = (
        auto()
    )  # The 'HealthBurden' module recognises modules with this label.

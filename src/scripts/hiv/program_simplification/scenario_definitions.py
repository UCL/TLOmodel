"""The file contains all the definitions of scenarios used the HIV program simplification analyses """

from typing import Dict

class ScenarioDefinitions:

    def status_quo(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return {
            'Hiv': {
                'type_of_scaleup': 'none',
            }
        }

    def reduce_HIV_test(self) -> Dict:
        return {
            'Hiv': {
                'type_of_scaleup': 'reduce_HIV_test',
            }
        }

    def remove_VL(self) -> Dict:
        return {
            'Hiv': {
                'type_of_scaleup': 'remove_VL',
            }
        }

    def target_VL(self) -> Dict:
        return {
            'Hiv': {
                'type_of_scaleup': 'target_VL',
            }
        }

    def replace_VL_with_TDF(self) -> Dict:
        return {
            'Hiv': {
                'type_of_scaleup': 'replace_VL_with_TDF',
            }
        }

    def remove_prep_fsw(self) -> Dict:
        return {
            'Hiv': {
                'type_of_scaleup': 'remove_prep_fsw',
            }
        }

    def remove_prep_agyw(self) -> Dict:
        return {
            'Hiv': {
                'type_of_scaleup': 'remove_prep_agyw',
            }
        }

    def remove_IPT(self) -> Dict:
        return {
            'Hiv': {
                'type_of_scaleup': 'remove_IPT',
            }
        }

    def target_IPT(self) -> Dict:
        return {
            'Hiv': {
                'type_of_scaleup': 'target_IPT',
            }
        }

    def remove_vmmc(self) -> Dict:
        return {
            'Hiv': {
                'type_of_scaleup': 'remove_vmmc',
            }
        }

    def increase_6MMD(self) -> Dict:
        return {
            'Hiv': {
                'type_of_scaleup': 'increase_6MMD',
            }
        }

    def remove_all(self) -> Dict:
        return {
            'Hiv': {
                'type_of_scaleup': 'remove_all',
            }
        }

    def scaleup(self) -> Dict:
        return {
            'Hiv': {
                'type_of_scaleup': 'target',
            }
        }

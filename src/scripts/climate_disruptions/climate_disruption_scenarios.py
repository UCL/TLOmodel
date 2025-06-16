from typing import Dict

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.methods.scenario_switcher import ImprovedHealthSystemAndCareSeekingScenarioSwitcher
from tlo.scenario import BaseScenario


class ClimateDisruptionScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2026, 1, 12)
        self.pop_size = 500
        self.runs_per_draw = 1
        self.YEAR_OF_CHANGE = 2020
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)

    def log_configuration(self):
        return {
            'filename': 'climate_scenario_runs',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.INFO,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
                'tlo.methods.population': logging.INFO,
                "tlo.methods.enhanced_lifestyle": logging.INFO
            }
        }

    def modules(self):
        return (
            fullmodel(resourcefilepath=self.resources)
            + [ImprovedHealthSystemAndCareSeekingScenarioSwitcher(resourcefilepath=self.resources)]
        )

    def draw_parameters(self, draw_number, rng):
        if draw_number < len(self._scenarios):

            return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """Return the Dict with values for the parameters that are changed, keyed by a name for the scenario."""
        return {#'Baseline': self._baseline(),
                'SSP 1.26 High': self._ssp126_high(),
                #'SSP 1.26 Low': self._ssp126_low(),
                #'SSP 1.26 Mean': self._ssp126_mean(),
                #'SSP 2.45 High': self._ssp245_high(),
                #'SSP 2.45 Low': self._ssp245_low(),
                #'SSP 2.45 Mean': self._ssp245_mean(),
                #'SSP 5.85 High': self._ssp585_high(),
                #'SSP 5.85 Low': self._ssp585_low(),
                #'SSP 5.85 Mean': self._ssp585_mean()
                }

    def _baseline(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                'max_healthsystem_function': [False, True],
                'max_healthcare_seeking': [False, True],
                'year_of_switch': self.YEAR_OF_CHANGE
                },
             "HealthSystem": {
                "mode_appt_constraints": 1,
                "cons_availability": "default",
                "cons_availability_postSwitch": "all",
                "year_cons_availability_switch": self.YEAR_OF_CHANGE,
                "beds_availability": "all",
                "equip_availability": "default",
                "equip_availability_postSwitch": "all",
                "year_equip_availability_switch": self.YEAR_OF_CHANGE,
                "use_funded_or_actual_staffing": "funded_plus",
                 "climate_ssp": 'ssp245', #status quo
                 "climate_model_ensemble_model": 'mean',
                 "services_affected_precip": 'none'
                },
                "Malaria": {
                    'type_of_scaleup': 'max',
                    'scaleup_start_year': self.YEAR_OF_CHANGE,
                },
                "Tb": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    },
                "Hiv": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    }
             },
        )

    def _ssp126_high(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                'max_healthsystem_function': [False, True],
                'max_healthcare_seeking': [False, True],
                'year_of_switch': self.YEAR_OF_CHANGE
                },
             "HealthSystem": {
                "mode_appt_constraints": 1,
                "cons_availability": "default",
                "cons_availability_postSwitch": "all",
                "year_cons_availability_switch": self.YEAR_OF_CHANGE,
                "beds_availability": "all",
                "equip_availability": "default",
                "equip_availability_postSwitch": "all",
                "year_equip_availability_switch": self.YEAR_OF_CHANGE,
                "use_funded_or_actual_staffing": "funded_plus",
                "climate_ssp":'ssp126',
                "climate_model_ensemble_model":'highest',
                 "services_affected_precip":'all'
                },
                "Malaria": {
                    'type_of_scaleup': 'max',
                    'scaleup_start_year': self.YEAR_OF_CHANGE,
                },
                "Tb": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    },
                "Hiv": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    }
             },
        )

    def _ssp126_low(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                'max_healthsystem_function': [False, True],
                'max_healthcare_seeking': [False, True],
                'year_of_switch': self.YEAR_OF_CHANGE
                },
             "HealthSystem": {
                "mode_appt_constraints": 1,
                "cons_availability": "default",
                "cons_availability_postSwitch": "all",
                "year_cons_availability_switch": self.YEAR_OF_CHANGE,
                "beds_availability": "all",
                "equip_availability": "default",
                "equip_availability_postSwitch": "all",
                "year_equip_availability_switch": self.YEAR_OF_CHANGE,
                "use_funded_or_actual_staffing": "funded_plus",
                "climate_ssp":'ssp126',
                "climate_model_ensemble_model":'lowest',
                 "services_affected_precip":'all'
                },
                "Malaria": {
                    'type_of_scaleup': 'max',
                    'scaleup_start_year': self.YEAR_OF_CHANGE,
                },
                "Tb": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    },
                "Hiv": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    }
             },
        )

    def _ssp126_mean(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                'max_healthsystem_function': [False, True],
                'max_healthcare_seeking': [False, True],
                'year_of_switch': self.YEAR_OF_CHANGE
                },
             "HealthSystem": {
                "mode_appt_constraints": 1,
                "cons_availability": "default",
                "cons_availability_postSwitch": "all",
                "year_cons_availability_switch": self.YEAR_OF_CHANGE,
                "beds_availability": "all",
                "equip_availability": "default",
                "equip_availability_postSwitch": "all",
                "year_equip_availability_switch": self.YEAR_OF_CHANGE,
                "use_funded_or_actual_staffing": "funded_plus",
                "climate_ssp":'ssp126',
                "climate_model_ensemble_model":'mean',
                 "services_affected_precip":'all'
                },
                "Malaria": {
                    'type_of_scaleup': 'max',
                    'scaleup_start_year': self.YEAR_OF_CHANGE,
                },
                "Tb": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    },
                "Hiv": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    }
             },
        )

    def _ssp245_high(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                'max_healthsystem_function': [False, True],
                'max_healthcare_seeking': [False, True],
                'year_of_switch': self.YEAR_OF_CHANGE
                },
             "HealthSystem": {
                "mode_appt_constraints": 1,
                "cons_availability": "default",
                "cons_availability_postSwitch": "all",
                "year_cons_availability_switch": self.YEAR_OF_CHANGE,
                "beds_availability": "all",
                "equip_availability": "default",
                "equip_availability_postSwitch": "all",
                "year_equip_availability_switch": self.YEAR_OF_CHANGE,
                "use_funded_or_actual_staffing": "funded_plus",
                "climate_ssp":'ssp245',
                "climate_model_ensemble_model":'highest',
                 "services_affected_precip":'all'
                },
                "Malaria": {
                    'type_of_scaleup': 'max',
                    'scaleup_start_year': self.YEAR_OF_CHANGE,
                },
                "Tb": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    },
                "Hiv": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    }
             },
        )

    def _ssp245_low(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                'max_healthsystem_function': [False, True],
                'max_healthcare_seeking': [False, True],
                'year_of_switch': self.YEAR_OF_CHANGE
                },
             "HealthSystem": {
                "mode_appt_constraints": 1,
                "cons_availability": "default",
                "cons_availability_postSwitch": "all",
                "year_cons_availability_switch": self.YEAR_OF_CHANGE,
                "beds_availability": "all",
                "equip_availability": "default",
                "equip_availability_postSwitch": "all",
                "year_equip_availability_switch": self.YEAR_OF_CHANGE,
                "use_funded_or_actual_staffing": "funded_plus",
                "climate_ssp":'ssp245',
                "climate_model_ensemble_model":'lowest',
                 "services_affected_precip":'all'
                },
                "Malaria": {
                    'type_of_scaleup': 'max',
                    'scaleup_start_year': self.YEAR_OF_CHANGE,
                },
                "Tb": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    },
                "Hiv": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    }
             },
        )

    def _ssp245_mean(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                'max_healthsystem_function': [False, True],
                'max_healthcare_seeking': [False, True],
                'year_of_switch': self.YEAR_OF_CHANGE
                },
             "HealthSystem": {
                "mode_appt_constraints": 1,
                "cons_availability": "default",
                "cons_availability_postSwitch": "all",
                "year_cons_availability_switch": self.YEAR_OF_CHANGE,
                "beds_availability": "all",
                "equip_availability": "default",
                "equip_availability_postSwitch": "all",
                "year_equip_availability_switch": self.YEAR_OF_CHANGE,
                "use_funded_or_actual_staffing": "funded_plus",
                "climate_ssp":'ssp245',
                "climate_model_ensemble_model":'mean',
                 "services_affected_precip":'all'
                },
                "Malaria": {
                    'type_of_scaleup': 'max',
                    'scaleup_start_year': self.YEAR_OF_CHANGE,
                },
                "Tb": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    },
                "Hiv": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    }
             },
        )

    def _ssp585_high(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                'max_healthsystem_function': [False, True],
                'max_healthcare_seeking': [False, True],
                'year_of_switch': self.YEAR_OF_CHANGE
                },
             "HealthSystem": {
                "mode_appt_constraints": 1,
                "cons_availability": "default",
                "cons_availability_postSwitch": "all",
                "year_cons_availability_switch": self.YEAR_OF_CHANGE,
                "beds_availability": "all",
                "equip_availability": "default",
                "equip_availability_postSwitch": "all",
                "year_equip_availability_switch": self.YEAR_OF_CHANGE,
                "use_funded_or_actual_staffing": "funded_plus",
                "climate_ssp":'ssp585',
                "climate_model_ensemble_model":'highest',
                 "services_affected_precip":'all'
                },
                "Malaria": {
                    'type_of_scaleup': 'max',
                    'scaleup_start_year': self.YEAR_OF_CHANGE,
                },
                "Tb": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    },
                "Hiv": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    }
             },
        )

    def _ssp585_low(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                'max_healthsystem_function': [False, True],
                'max_healthcare_seeking': [False, True],
                'year_of_switch': self.YEAR_OF_CHANGE
                },
             "HealthSystem": {
                "mode_appt_constraints": 1,
                "cons_availability": "default",
                "cons_availability_postSwitch": "all",
                "year_cons_availability_switch": self.YEAR_OF_CHANGE,
                "beds_availability": "all",
                "equip_availability": "default",
                "equip_availability_postSwitch": "all",
                "year_equip_availability_switch": self.YEAR_OF_CHANGE,
                "use_funded_or_actual_staffing": "funded_plus",
                "climate_ssp":'ssp585',
                "climate_model_ensemble_model":'lowest',
                 "services_affected_precip":'all'
                },
                "Malaria": {
                    'type_of_scaleup': 'max',
                    'scaleup_start_year': self.YEAR_OF_CHANGE,
                },
                "Tb": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    },
                "Hiv": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    }
             },
        )

    def _ssp585_mean(self) -> Dict:
        """Return the Dict with values for the parameter changes that define the baseline scenario. """
        return mix_scenarios(
            get_parameters_for_status_quo(),
            {'ImprovedHealthSystemAndCareSeekingScenarioSwitcher': {
                'max_healthsystem_function': [False, True],
                'max_healthcare_seeking': [False, True],
                'year_of_switch': self.YEAR_OF_CHANGE
                },
             "HealthSystem": {
                "mode_appt_constraints": 1,
                "cons_availability": "default",
                "cons_availability_postSwitch": "all",
                "year_cons_availability_switch": self.YEAR_OF_CHANGE,
                "beds_availability": "all",
                "equip_availability": "default",
                "equip_availability_postSwitch": "all",
                "year_equip_availability_switch": self.YEAR_OF_CHANGE,
                "use_funded_or_actual_staffing": "funded_plus",
                "climate_ssp":'ssp126',
                "climate_model_ensemble_model":'mean',
                 "services_affected_precip":'all'
                },
                "Malaria": {
                    'type_of_scaleup': 'max',
                    'scaleup_start_year': self.YEAR_OF_CHANGE,
                },
                "Tb": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    },
                "Hiv": {
                        'type_of_scaleup': 'max',
                        'scaleup_start_year': self.YEAR_OF_CHANGE,
                    }
             },
        )

if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])

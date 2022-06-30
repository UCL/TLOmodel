"""
This scenario runs the full model under a set of scenario in which each one TREATMENT_ID is excluded.

* No spurious symptoms
* Appts Contraints: Mode 0 (No Constraints)
* Consumables Availability: All
* Health care seeking forced to occur for every symptom

Run on the batch system using:
```tlo batch-submit src/scripts/healthsystem/finding_effects_of_each_treatment/scenario_force_healthcare_seeking.py```

or locally using:
    ```tlo scenario-run src/scripts/healthsystem/finding_effects_of_each_treatment/scenario_force_healthcare_seeking.py
    ```

"""
from pathlib import Path
from typing import Dict, List

from tlo import Date, logging
from tlo.analysis.utils import get_filtered_treatment_ids
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class EffectOfEachTreatment(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2014, 12, 31)
        self.pop_size = 50_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 3  # <- repeated this many times (per draw)

    def log_configuration(self):
        return {
            'filename': 'effect_of_each_treatment',
            'directory': Path('./outputs'),  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.demography.detail': logging.WARNING,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources, healthsystem_mode_appt_constraints=0)

    def draw_parameters(self, draw_number, rng):
        return {
            'HealthSystem': {
                'Service_Availability': list(self._scenarios.values())[draw_number],
                'cons_availability': 'all',
                },
            'HealthSeekingBehaviour': {
                'force_any_symptom_to_lead_to_healthcareseeking': True
                },

            # todo: Tim- all these params are registered as lists in my modules so need to be lists of the same values
            #  here (sorry thats quite messy)

            'PregnancySupervisor': {
                'odds_early_init_anc4': [20.0, 20.0],  # (prob 95%)
                'prob_anc1_months_2_to_4': [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                'prob_late_initiation_anc4': [0.0, 0.0],
                'prob_care_seeking_ectopic_pre_rupture': [1.0, 1.0],
                'prob_seek_care_pregnancy_complication': [1.0, 1.0],
                'prob_seek_care_pregnancy_loss': [1.0, 1.0],
            },

            'CareOfWomenDuringPregnancy': {
                'prob_intervention_delivered_urine_ds': [1.0, 1.0],
                'prob_intervention_delivered_bp': [1.0, 1.0],
                'prob_intervention_delivered_ifa': [1.0, 1.0],
                'prob_intervention_delivered_poct': [1.0, 1.0],
                'prob_intervention_delivered_syph_test': [1.0, 1.0],
                'prob_intervention_delivered_gdm_test': [1.0, 1.0],
                'squeeze_threshold_for_delay_three_an': [10_000, 10_000],
                'squeeze_factor_threshold_anc': [10_000, 10_000],
            },

            'Labour': {
                'odds_deliver_at_home': [0.06, 0.06],
                'prob_careseeking_for_complication': [1.0, 1.0],
                'prob_hcw_avail_iv_abx': [1.0, 1.0],
                'prob_hcw_avail_uterotonic': [1.0, 1.0],
                'prob_hcw_avail_man_r_placenta': [1.0, 1.0],
                'prob_hcw_avail_avd': [1.0, 1.0],
                'prob_hcw_avail_blood_tran': [1.0, 1.0],
                'prob_hcw_avail_surg': [1.0, 1.0],
                'prob_hcw_avail_retained_prod': [1.0, 1.0],
                'prob_hcw_avail_neo_resus': [1.0, 1.0],
                'mean_hcw_competence_hc': [[1.0, 1.0], [1.0, 1.0]],
                'mean_hcw_competence_hp': [[1.0, 1.0], [1.0, 1.0]],
                'odds_will_attend_pnc': [20.0, 20.0],  # (prob 95%)
                'prob_timings_pnc': [[1.0, 0.0], [1.0, 0.0]],
                'prob_delay_one_two_fd': [0.0, 0.0],
                'squeeze_threshold_for_delay_three_bemonc': [10_000, 10_000],
                'squeeze_threshold_for_delay_three_cemonc': [10_000, 10_000],
                'squeeze_threshold_for_delay_three_pn': [10_000, 10_000],
            },

            'NewbornOutcomes': {
                'prob_pnc_check_newborn': [1.0, 1.0],
                'prob_timings_pnc_newborns': [[1.0, 0.0], [1.0, 0.0]],
                'squeeze_threshold_for_delay_three_nb_care': [10_000, 10_000],
            },

            'PostnatalSupervisor': {
                'prob_care_seeking_postnatal_emergency’': [1.0, 1.0],
                'prob_care_seeking_postnatal_emergency_neonate': [1.0, 1.0],
            },

        }

    def _get_scenarios(self) -> Dict[str, List[str]]:
        """Return the Dict with values for the parameter `Service_Availability` keyed by a name for the scenario.
        The sequences of scenarios systematically omits one of the TREATMENT_ID's that is defined in the model. The
        complete list of TREATMENT_ID's is found by running `tlo_hsi_events.py`."""

        # Generate list of TREATMENT_IDs and filter to the resolution needed
        treatments = get_filtered_treatment_ids(depth=1)

        # Return 'Service_Availability' values, with scenarios for everything, nothing, and ones for which each
        # treatment is omitted
        service_availability = dict({"Everything": ["*"], "Nothing": []})
        service_availability.update(
            {f"No {t}": [x for x in treatments if x != t] for t in treatments}
        )

        return service_availability


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])

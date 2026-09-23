from tlo import Date, logging
from tlo.methods import mnh_cohort_module
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class InterventionLongScenario(BaseScenario):
    """Scenario for cohort model"""
    def __init__(self):
        super().__init__()
        self.seed = 120598
        self.start_date = Date(2025, 1, 1)
        self.end_date = Date(2026, 1, 2)
        self.pop_size = 90_000
        self.number_of_draws = 2
        self.runs_per_draw = 5

    def log_configuration(self):
        return {
            'filename': 'testing_bigger_pop_90K', 'directory': './outputs',
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.demography.detail": logging.INFO,
                "tlo.methods.contraception": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,  # TODO: will this work with new cons output
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.labour": logging.INFO,
                "tlo.methods.labour.detail": logging.INFO,
                "tlo.methods.newborn_outcomes": logging.INFO,
                "tlo.methods.care_of_women_during_pregnancy": logging.INFO,
                "tlo.methods.pregnancy_supervisor": logging.INFO,
                "tlo.methods.postnatal_supervisor": logging.INFO,
            }
        }

    def modules(self):
        return [*fullmodel(module_kwargs={'SymptomManager':{'always_refer_to_properties':True}}),
                 mnh_cohort_module.MaternalNewbornHealthCohort(stop_pregnancies=True)]

    def draw_parameters(self, draw_number, rng):

        if draw_number == 0:
            return {'PregnancySupervisor': {
                'analysis_year': 2025}}
        else:
            interventions_for_analysis = [
                # All interventions
                ["ectopic_pregnancy_treatment",
                 "post_abortion_care_core",
                 "sepsis_treatment",
                 "amtsl",
                 "pph_treatment_uterotonics",
                 "pph_treatment_mrrp",
                 "blood_transfusion_pph",
                 "blood_transfusion_aph",
                 "avd_ol",
                 "iv_anti_htns_ec",
                 "mgso4_spe",
                 "mgso4_ec",
                 "avd_spe_ec",
                 "caesarean_section_oth_surg_ip",
                 "caesarean_section_oth_surg_pp",
                 "neo_sepsis_treatment_term",
                 "neo_sepsis_treatment_preterm",
                 "kmc",
                 "neo_resus_term",
                 "neo_resus_preterm"]]

            return {'PregnancySupervisor': {
                'analysis_year': 2025,
                'interventions_analysis': True,
                'interventions_under_analysis': interventions_for_analysis[draw_number - 1],
                'intervention_analysis_availability': 1.0}}

if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])

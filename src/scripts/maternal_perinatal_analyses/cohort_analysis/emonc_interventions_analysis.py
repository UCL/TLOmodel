from tlo import Date, logging
from tlo.methods import mnh_cohort_module
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class EmoncScenario(BaseScenario):
    """Scenario for cohort model"""
    def __init__(self):
        super().__init__()
        self.seed = 537184
        self.start_date = Date(2025, 1, 1)
        self.end_date = Date(2026, 1, 2)
        self.pop_size = 20_000
        self.number_of_draws = 9
        self.runs_per_draw = 20

    def log_configuration(self):
        return {
            'filename': 'emonc_interventions', 'directory': './outputs',
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
        return [*fullmodel(module_kwargs={'SymptomManager':{'always_refer_to_properties':True},
                                          'Schisto': {'mda_execute': False}}),
                 mnh_cohort_module.MaternalNewbornHealthCohort()]

    def draw_parameters(self, draw_number, rng):
        if draw_number == 0:
            return {'PregnancySupervisor': {
                    'analysis_year': 2025}}        # TODO: 2025?

        else:
            interventions_for_analysis = [['abx_for_prom', 'sepsis_treatment', 'neo_sepsis_treatment'],   # TODO: PAC?
                                          'anti_htn_mgso4', # TODO: drop HTN?
                                          ['pph_treatment_uterotonics', 'amtsl'],
                                          ['pph_treatment_mrrp'],
                                          'post_abortion_care_core',   # TODO: retained products?
                                          'neo_resus'
                                          'blood_transfusion',
                                          'caesarean_section_oth_surg']

        return {'PregnancySupervisor': {
                'analysis_year': 2025,
                'interventions_analysis': True,
                'interventions_under_analysis': [interventions_for_analysis[draw_number]],
                'intervention_analysis_availability': 1.0}}


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])

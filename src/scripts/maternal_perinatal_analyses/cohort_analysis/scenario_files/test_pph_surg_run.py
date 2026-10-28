from tlo import Date, logging
from tlo.methods import mnh_cohort_module
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class EHPScaleUpScenario(BaseScenario):
    """Scenario using the cohort model in which the effective coverage of maternal and newborn health interventions
    is increased"""
    def __init__(self):
        super().__init__()
        self.seed = 120589
        self.start_date = Date(2026, 1, 1)
        self.end_date = Date(2027, 1, 2)
        self.pop_size = 40_000
        self.number_of_draws = 2
        self.runs_per_draw = 20

    def log_configuration(self):
        return {
            'filename': 'ehp_scale_up_scenario', 'directory': './outputs',
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.demography.detail": logging.INFO,
                "tlo.methods.contraception": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
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
                 mnh_cohort_module.MaternalNewbornHealthCohort()]

    def draw_parameters(self, draw_number, rng):

        if draw_number == 0:
            return {'PregnancySupervisor': {
                    'analysis_year': 2026}}
        else:

             interventions_for_analysis = ["caesarean_section_oth_surg_ip",
                                            "caesarean_section_oth_surg_pp"]

             return {'PregnancySupervisor': {
                        'analysis_year': 2026,
                        'interventions_analysis': True,
                        'interventions_under_analysis': interventions_for_analysis,
                        'intervention_analysis_availability': 1.0}}


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])

from tlo import Date, logging
from tlo.methods import mnh_cohort_module
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class EmoncScenario(BaseScenario):
    """Scenario for cohort model"""
    def __init__(self):
        super().__init__()
        self.seed = 7969672
        self.start_date = Date(2025, 1, 1)
        self.end_date = Date(2026, 1, 2)
        self.pop_size = 15_000
        self.number_of_draws = 10
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
        return [*fullmodel(module_kwargs={'SymptomManager':{'always_refer_to_properties':True}}),
                 mnh_cohort_module.MaternalNewbornHealthCohort()]

    def draw_parameters(self, draw_number, rng):
        if draw_number == 1:
            return {'PregnancySupervisor': {
                    'analysis_year': 2025}}
        else:

             # TODO : UPDATE THESE NOW WEVE ADDED NEW INTERVENTIONS

             interventions_for_analysis = [# Ectopic case management & post - abortion case management
                                           ["ectopic_pregnancy_treatment",
                                            "post_abortion_care_core"],

                                           # Maternal sepsis case management
                                           ["sepsis_treatment"],

                                           # Treatment of antepartum and postpartum hemorrhage
                                           ["pph_treatment_uterotonics",
                                            "pph_treatment_mrrp",
                                            "blood_transfusion_pph",
                                            "blood_transfusion_aph"],

                                           # Management of obstructed labor
                                           ["avd_ol"],

                                           # Management of pre-eclampsia and eclampsia
                                           ["iv_anti_htns",
                                            "mgso4",
                                            "avd_spe_ec"],

                                           # Caesarean section (uncomplicated and complicated) & other surgery
                                           ["caesarean_section_oth_surg"],

                                           # Newborn sepsis case management
                                           ["neo_sepsis_treatment_all"],

                                           # Essential care of preterm of sick newborn including KMC
                                           ["kmc", "neo_resus_preterm", "neo_sepsis_treatment_preterm"],

                                           # Newborn resuscitation
                                           ["neo_resus_all"]]


             return {'PregnancySupervisor': {
                        'analysis_year': 2025,
                        'interventions_analysis': True,
                        'interventions_under_analysis': interventions_for_analysis[draw_number-1],
                        'intervention_analysis_availability': 1.0}}

        # if draw_number == 1:
        #     return {'PregnancySupervisor': {
        #             'analysis_year': 2025}}
        #
        # else:
        #     interventions_for_analysis = [['sepsis_treatment', 'neo_sepsis_treatment'],   # TODO: abx for prom HTN?PAC?
        #                                   ['anti_htn_mgso4'], # TODO: drop HTN?
        #                                   ['pph_treatment_uterotonics', 'amtsl'],
        #                                   ['pph_treatment_mrrp'],
        #                                   ['post_abortion_care_core'],   # TODO: retained products?
        #                                   ['neo_resus'],
        #                                   ['blood_transfusion'],
        #                                   ['caesarean_section_oth_surg']]
        #
        #     return {'PregnancySupervisor': {
        #             'analysis_year': 2025,
        #             'interventions_analysis': True,
        #             'interventions_under_analysis': interventions_for_analysis[draw_number-2],
        #             'intervention_analysis_availability': 1.0}}


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])

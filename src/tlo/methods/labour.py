"""
Documentation:
First draft of Labour module (Natural history)
"""
import logging

import pandas as pd

import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

from tlo.methods import demography, healthsystem, healthburden


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Labour (Module):

    """
    This module models labour and delivery and generates the properties for "complications" of delivery """

    PARAMETERS = {

     #  ===================================  NATURAL HISTORY PARAMETERS ===============================================
        'prob_pregnancy': Parameter(
            Types.REAL, 'baseline probability of pregnancy'),  # DUMMY PARAMETER
        'prob_miscarriage': Parameter(
            Types.REAL, 'baseline probability of pregnancy loss before 28 weeks gestation'),
        'rr_miscarriage_prevmiscarriage': Parameter(
            Types.REAL, 'relative risk of pregnancy loss for women who have previously miscarried'),
        'rr_miscarriage_35': Parameter(
            Types.REAL, 'relative risk of pregnancy loss for women who is over 35 years old'),
        'rr_miscarriage_3134': Parameter(
            Types.REAL, 'relative risk of pregnancy loss for women who is between 31 and 34 years old'),
        'rr_miscarriage_grav4': Parameter(
            Types.REAL, 'relative risk of pregnancy loss for women who has a gravidity of greater than 4'),
        'prob_pl_ol': Parameter(
            Types.REAL, 'probability of a woman entering prolonged/obstructed labour'),
        'rr_PL_OL_nuliparity': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if they are nuliparous'),
        'rr_PL_OL_parity_3': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her parity is >3'),
        'rr_PL_OL_age_less20': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her age is less'
                        'than 20 years'),
        'prob_ptl': Parameter (
            Types.REAL, 'probability of a woman entering labour at <37 weeks gestation'),
        'rr_ptl_age20': Parameter(
            Types.REAL,'relative risk of preterm labour for women younger than 20'),
        'rr_ptl_pptb': Parameter(
            Types.REAL, 'relative risk of preterm labour for women younger than 20'),
        'prob_an_eclampsia': Parameter(
            Types.REAL, 'probability of an eclamptic seizure during labour'),
        'prob_an_aph': Parameter(
            Types.REAL, 'probability of an antepartum haemorrhage during labour'),
        'prob_an_sepsis': Parameter(
            Types.REAL, 'probability of sepsis in labour'),
        'rr_an_sepsis_pl_ol': Parameter(
            Types.REAL, 'relative risk of developing sepsis following obstructed labour'),
        'rr_an_eclampsia_30_34': Parameter(
            Types.REAL, 'relative risk of eclampsia for women ages between 30 and 34'),
        'rr_an_eclampsia_35': Parameter(
            Types.REAL, 'relative risk of eclampsia for women ages older than 35'),
        'rr_an_eclampsia_nullip': Parameter(
            Types.REAL, 'relative risk of eclampsia for women who have not previously delivered a child'),
        'rr_an_sepsis_anc_4': Parameter(
            Types.REAL, 'relative risk of sepsis for women who have attended greater than 4 ANC visits'),
        'rr_an_aph_noedu': Parameter(
            Types.REAL, 'relative risk of antepartum haemorrhage for women with education of primary level or lower'),
        'rr_an_aph_pl_ol': Parameter(
            Types.REAL, 'relative risk of antepartum haemorrhage following obstructed labour'),
        'cfr_aph': Parameter(
            Types.REAL, 'case fatality rate for antepartum haemorrhage during labour'),
        'cfr_eclampsia': Parameter(
            Types.REAL, 'case fatality rate for eclampsia during labours'),
        'cfr_sepsis': Parameter(
            Types.REAL, 'case fatality rate for sepsis during labour'),
        'cfr_uterine_rupture': Parameter(
            Types.REAL, 'case fatality rate for uterine rupture in labour'),
        'prob_still_birth_aph': Parameter(
            Types.REAL, 'probability of a still birth following antepartum haemorrhage where the mother survives'),
        'prob_still_birth_aph_md': Parameter(
            Types.REAL, 'probability of a still birth following antepartum haemorrhage where the mother dies'),
        'prob_still_birth_sepsis': Parameter(
            Types.REAL, 'probability of a still birth following sepsis in labour where the mother survives'),
        'prob_still_birth_sepsis_md': Parameter(
            Types.REAL, 'probability of a still birth following sepsis in labour where the mother dies'),
        'prob_still_birth_ur': Parameter(
            Types.REAL, 'probability of a still birth following uterine rupture in labour where the mother survives'),
        'prob_still_birth_ur_md': Parameter(
            Types.REAL, 'probability of a still birth following uterine rupture in labour where the mother dies'),
        'prob_still_birth_eclampsia': Parameter(
            Types.REAL, 'probability of still birth following eclampsia in labour where the mother survives'),
        'prob_still_birth_eclampsia_md': Parameter(
            Types.REAL, 'probability of still birth following eclampsia in labour where the mother dies'),
        'prob_pn_eclampsia': Parameter(
            Types.REAL, 'probability of eclampsia following delivery for women who were in spotaneous unobstructed '
                        'labour'),
        'prob_pn_pph': Parameter(
            Types.REAL, 'probability of an postpartum haemorrhage following labour'),
        'rr_pn_pph_pl_ol': Parameter(
            Types.REAL, 'relative risk of postpartum haemorrhage following obstructed labour'),
        'prob_pn_sepsis': Parameter(
            Types.REAL, 'probability of sepsis following delivery'),
        'prob_sa_pph': Parameter(
            Types.REAL, 'probability of a postpartum haemorrhage following a spontaneous abortion before 28 weeks'
                        ' gestation'),
        'prob_sa_sepsis': Parameter(
            Types.REAL, 'probability of a sepsis  following a spontaneous abortion before 28 weeks gestation'),
        'cfr_pn_pph': Parameter(
            Types.REAL, 'case fatality rate for postpartum haemorrhages'),
        'cfr_pn_eclampsia': Parameter(
            Types.REAL, 'case fatality rate for eclampsia following delivery'),
        'cfr_pn_sepsis': Parameter(
            Types.REAL, 'case fatality rate for sepsis following delivery'),
        'daly_wt_haemorrhage_moderate': Parameter(
            Types.REAL, 'DALY weight for a moderate maternal haemorrhage (<1 litre)'),
        'daly_wt_haemorrhage_severe': Parameter(
            Types.REAL, 'DALY weight for a severe maternal haemorrhage (>1 litre)'),
        'daly_wt_maternal_sepsis': Parameter(
            Types.REAL, 'DALY weight for maternal sepsis'),
        'daly_wt_eclampsia': Parameter(
            Types.REAL, 'DALY weight for eclampsia'),
        'daly_wt_obstructed_labour': Parameter(
            Types.REAL, 'DALY weight for obstructed labour'),
        'prob_neonatal_sepsis': Parameter(
            Types.REAL, 'baseline probability of a child developing sepsis following birth'),
        'prob_neonatal_enceph': Parameter(
            Types.REAL, 'baseline probability of a child developing neonatal encephalopathy following delivery'),

        # ================================= TREATMENT PARAMETERS =====================================================

        'rr_maternal_sepsis_clean_delivery': Parameter(
            Types.REAL, 'relative risk of maternal sepsis following clean birth practices employed in a facility'),
        'rr_newborn_sepsis_clean_delivery': Parameter(
            Types.REAL, 'relative risk of newborn sepsis following clean birth practices employed in a facility'),
        'prob_cure_antibiotics': Parameter(
            Types.REAL, 'Probability of sepsis resolving following the administration of antibiotics'),
        'prob_cure_mgso4': Parameter(
            Types.REAL, 'relative risk of additional seizures following of administration of magnesium sulphate'),
        'prob_prevent_mgso4': Parameter(
            Types.REAL, 'relative risk of eclampsia following administration of magnesium sulphate in women '
                        'with severe preeclampsia'),
        'prob_cure_diazepam': Parameter(
            Types.REAL, 'relative risk of additional seizures following of administration of diazepam'),
        'prob_cure_blood_transfusion': Parameter(
            Types.REAL, '...'),
        'prob_cure_oxytocin': Parameter(
            Types.REAL, 'probability of intravenous oxytocin arresting post-partum haemorrhage'),
        'prob_cure_misoprostol': Parameter(
            Types.REAL, 'probability of rectal misoprostol arresting post-partum haemorrhage'),
        'prob_cure_uterine_massage': Parameter(
            Types.REAL, 'probability of uterine massage arresting post-partum haemorrhage'),
        'prob_cure_uterine_tamponade': Parameter(
            Types.REAL, 'probability of uterine tamponade arresting post-partum haemorrhage'),
        'prob_cure_uterine_ligation': Parameter(
            Types.REAL, 'probability of laparotomy and uterine ligation arresting post-partum haemorrhage'),
        'prob_cure_b_lych': Parameter(
            Types.REAL, 'probability of laparotomy and B-lynch sutures arresting post-partum haemorrhage'),
        'prob_cure_hysterectomy': Parameter(
            Types.REAL, 'probability of total hysterectomy arresting post-partum haemorrhage'),
        'prob_cure_manual_removal': Parameter(
            Types.REAL, 'probability of manual removal of retained products arresting a post partum haemorrhage'),
        'prob_cure_uterine_repair': Parameter(
            Types.REAL, 'probability repairing a ruptured uterus surgically'),
        'prob_deliver_ventouse': Parameter(
            Types.REAL, 'probability of successful delivery with ventouse'),
        'prob_deliver_forceps': Parameter(
            Types.REAL, 'probability of successful delivery with forceps'),
    }

    PROPERTIES = {

        # TODO: ensure property names are consistant (use of last/this) should conform across names
        'la_due_date': Property(Types.DATE, 'The predicted date of delivery for a newly pregnant woman'),
        'la_labour': Property(Types.CATEGORICAL, 'not in labour, Term labour, Early Preterm Labour, '
                                                 'Late Preterm Labour, Post term labour',
                              categories=['not_in_labour', 'term_labour', 'early_preterm_labour', 'late_preterm_labour',
                                          'post_term_labour']),
        'la_gestation_at_labour':Property(Types.INT, 'length of gestation in weeks at the point the woman enters '
                                                     'labour'),
        'la_still_birth_this_delivery': Property(Types.BOOL,'whether this womans most recent pregnancy has ended in a '
                                                            'stillbirth'),
        'la_miscarriage': Property(Types.INT, 'the number of miscarriages a woman has experienced'),
        'la_miscarriage_date': Property(Types.DATE, 'the date this woman has last experienced spontaneous miscarriage'),
        'la_parity': Property(Types.INT, 'total number of previous deliveries'),
        'la_previous_cs': Property(Types.INT, 'number of previous deliveries by caesarean section'),
        'la_previous_ptb': Property(Types.BOOL, 'whether the woman has had a previous preterm delivery for any of her'
                                                'previous deliveries'),
        'la_obstructed_labour':Property(Types.BOOL, 'whether this womans labour has become obstructed'),
        'la_aph': Property(Types.BOOL, 'whether the woman has experienced an antepartum haemorrhage in this delivery'),
        'la_uterine_rupture': Property(Types.BOOL, 'whether the woman has experienced uterine rupture in this delivery'),
        'la_sepsis': Property(Types.BOOL, 'whether the woman has developed sepsis associated with in this delivery'),
        'la_eclampsia': Property(Types.BOOL, 'whether the woman has experienced an eclamptic seizure in this delivery'),
        'la_pph': Property(Types.BOOL, 'whether the woman has experienced an postpartum haemorrhage in this delivery'),
        'la_complications_of_labour': Property(Types.LIST, 'list recording complications experienced by a woman'
                                                                 ' during her current labour, that is not edited by '
                                                                 'interventions. Key 0=No, 1=Yes. Order '
                                                                 '1.)Obstructed Labour, 2.)Eclampsia, 3.)APH,'
                                                                 ' 4.)Sepsis,'' 5.)Uterine Rupture, 6.)PPH, '
                                                                 '7.)Postpartum Sepsis, '
                                                                 '8.) Postpartum eclampsia'),
        'la_delivery_mode': Property(Types.CATEGORICAL, 'Vaginal Delivery, Assisted Vaginal Delivery,'
                                                        'Emergency Caesarean Section, Elective Caesarean Section'
                                     , categories=['VD', 'AVD', 'EmCS', 'ElCS']),
        'la_died_in_labour': Property(Types.BOOL, 'whether the woman has died during this labour'),


        # ==================================== PROPERTIES TO LIVE IN THE NEW MATERNAL DATAFRAME ========================

        'mdf_location_of_delivery': Property(Types.CATEGORICAL, 'Facility Delivery, Home Birth',
                                         categories=['facility_delivery', 'home_birth']),
        'mdf_indication_for_attendance': Property(Types.CATEGORICAL, 'Attending in Labour, Attending for Induction, '
                                                                     'Attending for Caesarean',
                                                  categories=['attending_in_labour', 'attending_for_induction',
                                                              'attending_for_caesarean']),
        'mdf_risk_of_obstructed_labour': Property(Types.REAL, 'Individual risk of a womans labour being obstructed'),
        'mdf_obstructed_labour': Property(Types.BOOL, 'Whether, during current labour, a woman has developed obstructed'
                                                     ' labour'),
        'mdf_obstructed_labour_treated': Property(Types.BOOL, 'Whether obstructed labour has been successful treated'),
        # TODO: consider if we want to record the treatment received? what would be the benifit
        'mdf_risk_of_maternal_sepsis': Property(Types.REAL, 'Individual risk of a woman developing sepsis during '
                                                           'labour or postpartum'),
        # todo: should there be 2 seperate risk (intrapartum and post partum)
        'mdf_maternal_sepsis': Property(Types.BOOL, 'Whether, during current labour or post partum, a woman has '
                                                    'developed sepsis'),
        'mdf_source_of_sepsis': Property(Types.CATEGORICAL, 'Obstetric Sepsis, Non-Obstetric Sepsis',
                                                  categories=['obstetric_sepsis', 'non_obstetric_sepsis']),
        'mdf_timing_of_sepsis': Property(Types.CATEGORICAL, 'Intrapartum, Postpartum',
                                         categories=['intrapartum', 'postpartum']),
        'mdf_maternal_sepsis_treated': Property(Types.BOOL, 'Whether maternal sepsis has been successful treated'),
        'mdf_risk_of_aph': Property(Types.REAL, 'Individual risk of a woman developing an antepartum haemorrhage '
                                               'during labour'),
        'mdf_aph': Property(Types.BOOL, 'Whether, during current labour, a woman has developed an antepartum '
                                        'haemorrhage labour'),
        # todo: should include severity of bleed?
        'mdf_aph_treated': Property(Types.BOOL, 'Whether antepartum haemorrhage has been successful treated'),
        'mdf_risk_of_eclampsia': Property(Types.REAL, 'Individual risk of a woman developing eclampsia '
                                               'during labour or postpartum'),
        # todo: should there be 2 seperate risk (intrapartum and post partum)
        'mdf_eclampsia': Property(Types.BOOL, 'Whether, during current labour, a woman has developed eclampsia'),
        'mdf_timing_of_eclampsia': Property(Types.CATEGORICAL, 'Intrapartum, Postpartum',
                                         categories=['intrapartum', 'postpartum']),
        'mdf_eclampsia_treated': Property(Types.BOOL, 'Whether eclampsia has been successful treated'),
        'mdf_risk_of_uterine_rupture': Property(Types.REAL, 'Individual risk of a woman developing a uterine rupture'),
        'mdf_uterine_rupture': Property(Types.BOOL, 'Whether, during current labour, a woman has developed an '
                                                    'uterine rupture'),
        'mdf_uterine_rupture_treated': Property(Types.BOOL, 'Whether uterine rupture has been successful treated'),
        'mdf_risk_of_pph': Property(Types.REAL, 'Individual risk of a woman developing an postpartum haemorrhage '
                                               'during labour'),
        'mdf_pph': Property(Types.BOOL, 'Whether, during current labour, a woman has developed an postpartum '
                                        'haemorrhage labour'),
        'mdf_pph_treated': Property(Types.BOOL, 'Whether postpartum haemorrhage has been successful treated'),
        'mdf_still_birth_current_pregnancy': Property(Types.BOOL, 'Whether a pregnant womans pregnancy ends in '
                                                                  'stillbirth'),
        'mdf_died_in_labour': Property(Types.BOOL, 'whether the woman has died during this labour'),

    }

    def __init__(self):
        super().__init__()

        self.mother_and_newborn_info = dict()

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters

        #  ===================================  NATURAL HISTORY PARAMETERS ============================================

        params['prob_pregnancy'] = 0.083  # Calculated from DHS 2010
        params['prob_miscarriage'] = 0.053
        params['rr_miscarriage_prevmiscarriage'] = 2.23
        params['rr_miscarriage_35'] = 4.02
        params['rr_miscarriage_3134'] = 2.13
        params['rr_miscarriage_grav4'] = 0.49
        params['prob_pl_ol'] = 0.058
        params['rr_PL_OL_nuliparity'] = 1.47
        params['rr_PL_OL_para1'] = 1.57
        params['rr_PL_OL_age_less20'] = 1.3
        params['prob_ptl'] = 0.09
        params['rr_ptl_pptb'] = 2.13
        params['prob_an_eclampsia'] = 0.01
        params['prob_an_aph'] = 0.012
        params['prob_an_sepsis'] = 0.005
        params['rr_an_sepsis_pl_ol'] = 3.4
        params['prob_an_ur'] = 0.001
        params['rr_an_ur_grand_multip'] = 7.57
        params['rr_an_ur_prevcs'] = 2.02
        params['rr_an_ur_ref_ol'] = 23.65  # REVIEW "obstructed but not referred"
        params['rr_an_eclampsia_30_34'] = 1.4
        params['rr_an_eclampsia_35'] = 1.95
        params['rr_an_eclampsia_nullip'] = 2.04
        params['rr_an_sepsis_anc_4'] = 0.5
        params['rr_an_aph_noedu'] = 1.72
        params['rr_an_aph_pl_ol'] = 9.1
        params['cfr_aph'] = 0.02
        params['cfr_eclampsia'] = 0.184
        params['cfr_sepsis'] = 0.33
        params['cfr_uterine_rupture'] = 0.345
        params['prob_still_birth_aph'] = 0.38
        params['prob_still_birth_aph_md'] = 0.90
        params['prob_still_birth_sepsis'] = 0.25
        params['prob_still_birth_sepsis_md'] = 0.90
        params['prob_still_birth_ur'] = 0.93
        params['prob_still_birth_ur_md'] = 0.98
        params['prob_still_birth_eclampsia'] = 0.03
        params['prob_still_birth_eclampsia_md'] = 0.90
        params['prob_pn_eclampsia'] = 0.01
        params['prob_pn_pph'] = 0.03
        params['rr_pn_pph_pl_ol'] = 5.0
        params['prob_pn_sepsis'] = 0.05
        params['prob_sa_pph'] = 0.12
        params['prob_sa_sepsis'] = 0.20
        params['cfr_pn_pph'] = 0.1
        params['cfr_pn_eclampsia'] = 0.184
        params['cfr_pn_sepsis'] = 0.33
        params['prob_neonatal_sepsis'] = 0.15
        params['prob_neonatal_enceph'] = 0.16

        # ================================= TREATMENT PARAMETERS =====================================================

        params['rr_maternal_sepsis_clean_delivery'] = 0.7  # dummy
        params['rr_newborn_sepsis_clean_delivery'] = 0.7 # dummy
        params['prob_cure_antibiotics'] = 0.5  # dummy
        params['prob_cure_mgso4'] = 0.57  # probability taken from RR of 0.43for additional seizures (vs diazepam alone)
        params['prob_prevent_mgso4'] = 0.41  # Risk reduction of eclampsia in women who have pre-eclampsia
        params['prob_cure_diazepam'] = 0.8
        params['prob_cure_blood_transfusion'] = 0.4  # dummy
        params['prob_cure_oxytocin'] = 0.5  # dummy
        params['prob_cure_misoprostol'] = 0.3  # dummy
        params['prob_cure_uterine_massage'] = 0.15  # dummy
        params['prob_cure_uterine_tamponade'] = 0.6  # dummy
        params['prob_cure_uterine_ligation'] = 0.8  # dummy
        params['prob_cure_b_lych'] = 0.8  # dummy
        params['prob_cure_hysterectomy'] = 0.95  # dummy
        params['prob_cure_manual_removal'] = 0.75  # dummy
        params['prob_cure_uterine_repair'] = 0.7 # dummy
        params['prob_deliver_ventouse'] = 0.7 # dummy
        params['prob_deliver_forceps'] = 0.7 # dummy

        # Here we will include DALY weights if applicable...

        if 'HealthBurden' in self.sim.modules.keys():
            params['daly_wt_haemorrhage_moderate'] = self.sim.modules['HealthBurden'].get_daly_weight(339)
            params['daly_wt_haemorrhage_severe'] = self.sim.modules['HealthBurden'].get_daly_weight(338)
            params['daly_wt_maternal_sepsis'] = self.sim.modules['HealthBurden'].get_daly_weight(340)
            params['daly_wt_eclampsia'] = self.sim.modules['HealthBurden'].get_daly_weight(347)
            params['daly_wt_obstructed_labour'] = self.sim.modules['HealthBurden'].get_daly_weight(348)

        # DALYs not included - Uterine Rupture, c section?

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props
        m = self
        rng = m.rng
        params = self.parameters

    # ----------------------------------------- DEFAULTS ---------------------------------------------------------------

        df.loc[df.sex == 'F', 'la_labour'] = 'not_in_labour'
        df.loc[df.sex == 'F', 'la_gestation_at_labour'] = 0
        df.loc[df.sex == 'F', 'la_miscarriage'] = 0
        df.loc[df.sex == 'F', 'la_miscarriage_date'] = pd.NaT
        df.loc[df.sex == 'F', 'la_still_birth_this_delivery'] = False
        df.loc[df.sex == 'F', 'la_parity'] = 0
        df.loc[df.sex == 'F', 'la_previous_cs'] = 0
        df.loc[df.sex == 'F', 'la_previous_ptb'] = False
        df.loc[df.sex == 'F', 'la_due_date'] = pd.NaT
        df.loc[df.sex == 'F', 'la_obstructed_labour'] = False
        df.loc[df.sex == 'F', 'la_aph'] =False
        df.loc[df.sex == 'F', 'la_uterine_rupture'] = False
        df.loc[df.sex == 'F', 'la_eclampsia'] = False
        df.loc[df.sex == 'F', 'la_pph'] = False
        df.loc[df.sex == 'F', 'la_delivery_mode'] = None
        df.loc[df.sex == 'F', 'la_died_in_labour'] = False
        df.loc[df.sex == 'F', 'la_complications_of_labour'] = None

# ======================================================================================================================
# ===============================DEFAULTS FOR PROPERTIES TO LIVE IN THE NEW MATERNAL DATAFRAME =========================
# ======================================================================================================================

        df.loc[df.is_alive, 'mdf_location_of_delivery'] = None
        df.loc[df.is_alive, 'mdf_indication_for_attendance'] = None
        df.loc[df.is_alive, 'mdf_risk_of_obstructed_labour'] = 0
        df.loc[df.is_alive, 'mdf_obstructed_labour'] = False
        df.loc[df.is_alive, 'mdf_obstructed_labour_treated'] = False
        df.loc[df.is_alive, 'mdf_risk_of_maternal_sepsis'] = 0
        df.loc[df.is_alive, 'mdf_maternal_sepsis'] = False
        df.loc[df.is_alive, 'mdf_source_of_sepsis'] = None
        df.loc[df.is_alive, 'mdf_timing_of_sepsis'] = None
        df.loc[df.is_alive, 'mdf_maternal_sepsis_treated'] = False
        df.loc[df.is_alive, 'mdf_risk_of_aph'] = 0
        df.loc[df.is_alive, 'mdf_aph'] = False
        df.loc[df.is_alive, 'mdf_aph_treated'] = False
        df.loc[df.is_alive, 'mdf_risk_of_eclampsia'] = 0
        df.loc[df.is_alive, 'mdf_eclampsia'] = False
        df.loc[df.is_alive, 'mdf_timing_of_eclampsia'] = None
        df.loc[df.is_alive, 'mdf_eclampsia_treated'] = False
        df.loc[df.is_alive, 'mdf_risk_of_uterine_rupture'] = 0
        df.loc[df.is_alive, 'mdf_uterine_rupture'] = False
        df.loc[df.is_alive, 'mdf_uterine_rupture_treated'] = False
        df.loc[df.is_alive, 'mdf_risk_of_pph'] = 0
        df.loc[df.is_alive, 'mdf_pph'] = False
        df.loc[df.is_alive, 'mdf_still_birth_current_pregnancy'] = False
        df.loc[df.is_alive, 'mdf_died_in_labour'] = False


# -----------------------------------ASSIGN PREGNANCY AND DUE DATE AT BASELINE (DUMMY) --------------------------------

        # TODO: Await contraception code

        # Get and hold all the women who are eligible to become pregnant at baseline
        women_idx = df.index[(df.age_years >= 15) & (df.age_years <= 49) & df.is_alive & (df.sex == 'F')]

        # Apply an effective probability of pregnancy at baseline and allocate these women to be pregnant
        eff_prob_preg = pd.Series(m.prob_pregnancy, index=women_idx)

        random_draw = pd.Series(rng.random_sample(size=len(women_idx)),
                                index=df.index[(df.age_years >= 15) & (df.age_years <= 49) & df.is_alive
                                               & (df.sex == 'F')])

        dfx = pd.concat([eff_prob_preg, random_draw], axis=1)
        dfx.columns = ['eff_prob_pregnancy', 'random_draw']
        idx_pregnant = dfx.index[dfx.eff_prob_pregnancy > dfx.random_draw]
        df.loc[idx_pregnant, 'is_pregnant'] = True

# ---------------------------------    GESTATION AND SCHEDULING BIRTH BASELINE  ---------------------------------------

        # TODO: ensure women at baseline can go into preterm labour or be overdue (how much does this matter?)

        # Get and hold all the women who are pregnant at baseline
        pregnant_idx = df.index[df.is_pregnant & df.is_alive]

        random_draw = pd.Series(rng.random_sample(size=len(pregnant_idx)),
                                index=df.index[df.is_pregnant & df.is_alive])

        # Randomly generate a number of weeks gestation between 1-39 for all pregnant women
        simdate = pd.Series(self.sim.date, index=pregnant_idx)
        dfx = pd.concat((simdate, random_draw), axis=1)
        dfx.columns = ['simdate', 'random_draw']
        dfx['gestational_age_in_weeks'] = (39 - 39 * dfx.random_draw)

        # Use this gestational age to calculate when the woman's baby was conceived
        dfx['la_conception_date'] = dfx['simdate'] - pd.to_timedelta(dfx['gestational_age_in_weeks'], unit='w')

        # Apply a due date of 9 months in the future from the date of conception for each woman
        dfx['due_date_mth'] = 39 - dfx['gestational_age_in_weeks']
        dfx['due_date'] = dfx['simdate'] + pd.to_timedelta(dfx['due_date_mth'], unit='w')
        df.loc[pregnant_idx, 'date_of_last_pregnancy'] = dfx.la_conception_date
        df.loc[pregnant_idx, 'la_due_date'] = dfx.due_date

        pregnant_baseline = df.index[(df.is_pregnant == True) & df.is_alive]

        for person in pregnant_baseline:
            scheduled_labour_date = df.at[person, 'la_due_date']
            labour = LabourEvent(self, individual_id=person, cause='Labour')
            self.sim.schedule_event(labour, scheduled_labour_date)

#  ----------------------------ASSIGNING PARITY AT BASELINE (DUMMY)-----------------------------------------------------

        # Get and hold all the women in the dataframe who between the ages of 15-24 years old
        women_parity_1524_idx = df.index[(df.age_years >= 15) & (df.age_years <= 24) & (df.is_alive == True)
                                         & (df.sex == 'F')]

        baseline_p = pd.Series(0, index=women_parity_1524_idx)

        # Probability weighted random draw is applied to each women to determine how many previous deliveries she has
        # had
        random_draw2 = pd.Series(self.rng.choice(range(0, 5), p=[0.40, 0.35, 0.15, 0.06, 0.04],
                                                 size=len(women_parity_1524_idx)),
                                 index=df.index[(df.age_years >= 15) & (df.age_years <= 24)
                                                & df.is_alive & (df.sex == 'F')])

        dfx = pd.concat([baseline_p, random_draw2], axis=1)
        dfx.columns = ['baseline_p', 'random_draw2']
        idx_parity = dfx.index[dfx.baseline_p < dfx.random_draw2]
        df.loc[idx_parity, 'la_parity'] = dfx.random_draw2

        # These steps are repeated with different weightings for the next two older age groups
        women_parity_2540_idx = df.index[(df.age_years >= 25) & (df.age_years <= 40) & (df.is_alive == True)
                                         & (df.sex == 'F')]

        baseline_p = pd.Series(0, index=women_parity_2540_idx)

        random_draw = pd.Series(self.rng.choice(range(0, 6), p=[0.05, 0.15, 0.30, 0.20, 0.2, 0.1],
                                                size=len(women_parity_2540_idx)), index=df.index[(df.age_years >= 25) &
                                                                                                 (df.age_years <= 40)
                                                                                                 & (df.is_alive == True)
                                                                                                 & (df.sex == 'F')])

        dfx = pd.concat([baseline_p, random_draw], axis=1)
        dfx.columns = ['baseline_p', 'random_draw']
        idx_parity = dfx.index[dfx.baseline_p < dfx.random_draw]
        df.loc[idx_parity, 'la_parity'] = dfx.random_draw

        women_parity_4149_idx = df.index[(df.age_years >= 41) & (df.age_years <= 49) & (df.is_alive == True)
                                         & (df.sex == 'F')]

        baseline_p = pd.Series(0, index=women_parity_4149_idx)

        random_draw = pd.Series(self.rng.choice(range(0, 7), p=[0.05, 0.10, 0.25, 0.30, 0.25, 0.03, 0.02],
                                                size=len(women_parity_4149_idx)), index=df.index[(df.age_years >= 41)
                                                                                                 & (df.age_years <= 49)
                                                                                                 & (df.is_alive == True)
                                                                                                 & (df.sex == 'F')])

        dfx = pd.concat([baseline_p, random_draw], axis=1)
        dfx.columns = ['baseline_p', 'random_draw']
        idx_parity = dfx.index[dfx.baseline_p < dfx.random_draw]
        df.loc[idx_parity, 'la_parity'] = dfx.random_draw

#   ------------------------------ ASSIGN PREVIOUS CS AT BASELINE -----------------------------------------------

        # Get and hold women who have delivered one child and those who have delivered 2 or more children
        women_para1_idx = df.index[(df.la_parity == 1)]
        women_para2_idx = df.index[(df.la_parity >= 2)]

        baseline_cs1 = pd.Series(0, index=women_para1_idx)
        baseline_cs2 = pd.Series(0, index=women_para2_idx)

        # A weighted random choice is used to determine whether women who are para1 had delivered via caesarean
        random_draw1 = pd.Series(self.rng.choice(range(0, 2), p=[0.91, 0.09], size=len(women_para1_idx)),
                               index=women_para1_idx)

        # A second weighted random choice is applied to women with greater than 2 deliveries, due to low section rates
        # a maximum of 2 previous deliveries by caesarean is allowed
        random_draw2 = pd.Series(self.rng.choice(range(0, 3), p=[0.90, 0.07, 0.03], size=len(women_para2_idx)),
                                 index=women_para2_idx)

        dfx = pd.concat([baseline_cs1, random_draw1], axis=1)
        dfx.columns = ['baseline_cs1', 'random_draw1']
        idx_prev_cs = dfx.index[dfx.random_draw1 >= 1]
#       df.loc[idx_prev_cs, 'la_previous_cs'] = dfx.random_draw1

        dfx = pd.concat([baseline_cs2, random_draw2], axis=1)
        dfx.columns = ['baseline_cs2','random_draw2']
        idx_prev_cs1 = dfx.index[dfx.random_draw2 >=1]
#        df.loc[idx_prev_cs1, 'la_previous_cs'] = dfx.random_draw2

        # ------------------------------ ASSIGN PREVIOUS PTB AT BASELINE ----------------------------------------------

        # Get and hold all women who have given birth previously, excluding those with previous caesarean section
        women_para1_nocs_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') & (df.la_parity == 1) &
                                        (df.la_previous_cs ==0)]

        # Get and hold all women with greater than 2 deliveries excluding those in which both deliveries were by
        # caesarean
        women_para2_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') & (df.la_parity >= 2) &
                                   (df.la_previous_cs < 2)]

        baseline_ptb = pd.Series(m.prob_ptl, index=women_para1_nocs_idx)
        baseline_ptb_p2 = pd.Series(m.prob_ptl, index=women_para2_idx)

        # Multiply baseline probability of preterm birth by the relative risk of preterm birth in under 20s
#        baseline_ptb.loc[(df.is_alive & (df.sex == 'F') & (df.age_years >= 15) & (df.age_years <= 20))] *= \
#            params['rr_ptl_age20']
#        baseline_ptb_p2.loc[(df.is_alive & (df.sex == 'F') & (df.age_years >= 15) & (df.age_years <= 20))] *= \
#            params['rr_ptl_age20']

        random_draw = pd.Series(rng.random_sample(size=len(women_para1_nocs_idx)),
                                index=df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') &
                                               (df.la_parity == 1) & (df.la_previous_cs == 0)])
        random_draw2 = pd.Series(rng.random_sample(size=len(women_para2_idx)),
                                index=df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') &
                                               (df.la_parity >= 2) & (df.la_previous_cs < 2)])

        # Use a random draw to determine if this woman's past deliveries have ever been preterm
        dfx = pd.concat([baseline_ptb, random_draw], axis=1)
        dfx.columns = ['baseline_ptb', 'random_draw']
        idx_prev_ptb = dfx.index[dfx.baseline_ptb > dfx.random_draw]
        df.loc[idx_prev_ptb, 'la_previous_ptb'] = True

        dfx = pd.concat([baseline_ptb_p2, random_draw2], axis=1)
        dfx.columns = ['baseline_ptb_p2', 'random_draw2']
        idx_prev_ptb = dfx.index[dfx.baseline_ptb_p2 > dfx.random_draw2]
        df.loc[idx_prev_ptb, 'la_previous_ptb'] = True

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def initialise_simulation(self, sim):

        event = LabourLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):

        df = self.sim.population.props
        if df.at[child_id, 'sex'] == 'F':
            df.at[child_id, 'la_due_date'] = pd.NaT
            df.at[child_id, 'la_labour'] = 'not_in_labour'
            df.at[child_id, 'la_gestation_at_labour'] = 0
            df.at[child_id, 'la_miscarriage'] = 0
            df.at[child_id, 'la_miscarriage_date'] = pd.NaT
            df.at[child_id, 'la_still_birth_this_delivery'] = False
            df.at[child_id, 'la_parity'] = 0
            df.at[child_id, 'la_previous_cs'] = 0
            df.at[child_id, 'la_previous_ptb'] = False
            df.at[child_id, 'la_obstructed_labour'] = False
            df.at[child_id, 'la_aph'] = False
            df.at[child_id, 'la_uterine_rupture'] = False
            df.at[child_id, 'la_sepsis'] = False
            df.at[child_id, 'la_eclampsia'] = False
            df.at[child_id, 'la_pph'] = False
            df.at[child_id, 'la_delivery_mode'] = None
            df.at[child_id, 'la_died_in_labour'] = False

# ======================================================================================================================
# ===============================DEFAULTS FOR PROPERTIES TO LIVE IN THE NEW MATERNAL DATAFRAME =========================
# ======================================================================================================================

            df.at[child_id, 'mdf_location_of_delivery'] = None
            df.at[child_id, 'mdf_indication_for_attendance'] = None
            df.at[child_id, 'mdf_risk_of_obstructed_labour'] = 0
            df.at[child_id, 'mdf_obstructed_labour'] = False
            df.at[child_id, 'mdf_obstructed_labour_treated'] = False
            df.at[child_id, 'mdf_risk_of_maternal_sepsis'] = 0
            df.at[child_id, 'mdf_maternal_sepsis'] = False
            df.at[child_id, 'mdf_source_of_sepsis'] = None
            df.at[child_id, 'mdf_timing_of_sepsis'] = None
            df.at[child_id, 'mdf_maternal_sepsis_treated'] = False
            df.at[child_id, 'mdf_risk_of_aph'] = 0
            df.at[child_id, 'mdf_aph'] = False
            df.at[child_id, 'mdf_aph_treated'] = False
            df.at[child_id, 'mdf_risk_of_eclampsia'] = 0
            df.at[child_id, 'mdf_eclampsia'] = False
            df.at[child_id, 'mdf_timing_of_eclampsia'] = None
            df.at[child_id, 'mdf_eclampsia_treated'] = False
            df.at[child_id, 'mdf_risk_of_uterine_rupture'] = 0
            df.at[child_id, 'mdf_uterine_rupture'] = False
            df.at[child_id, 'mdf_uterine_rupture_treated'] = False
            df.at[child_id, 'mdf_risk_of_pph'] = 0
            df.at[child_id, 'mdf_pph'] = False
            df.at[child_id, 'mdf_still_birth_current_pregnancy'] = False
            df.at[child_id, 'mdf_died_in_labour'] = False

        # If a mothers labour has resulted in a late term still birth her child is still generated by the simulation
        # but is_alive is reset to false to allow for monitoring of still birth rates

        # TODO: Review if we should continue to record still birth like this or just through logging
        if df.at[mother_id, 'la_still_birth_this_delivery']:
            death = demography.InstantaneousDeath(self.sim.modules['Demography'], child_id,
                                                  cause='Intrapartum Stillbirth')
            self.sim.schedule_event(death, self.sim.date)

            # Log the still birth
            logger.debug('@@@@ A Still Birth has occurred, to mother %s', mother_id)
            logger.info('%s|still_birth|%s', self.sim.date,
                        {'age': df.at[child_id, 'age_years'],
                            'person_id': child_id,
                            'mother_id': mother_id})

            # This property is then reset in case of future pregnancies/stillbirths
            df.loc[mother_id, 'la_still_birth_this_delivery'] = False

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.debug('This is Labour, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

    # TODO: Need to build DALY module to count DALYs associated with Pregnancy?(or just labour?)

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        logger.debug('This is Labour reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe

        p = self.parameters

#        health_values = df.loc[df.is_alive, 'mi_specific_symptoms'].map({
#            'none': 0,
#            'mild sneezing': p['daly_wt_mild_sneezing'],
#            'coughing and irritable': p['daly_wt_coughing'],
#            'extreme emergency': p['daly_wt_advanced']
#        })
#        health_values.name = 'Mockitis Symptoms'    # label the cause of this disability

#        return health_values.loc[df.is_alive]   # returns the series


class CheckIfNewlyPregnantWomanWillMiscarry(Event, IndividualScopeEventMixin):
    """This event checks if a woman who is newly pregnant will experience a miscarriage, and will record on what date
     this miscarriage has occured. Women who """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self


        # First we identify if this woman has any risk factors for early pregnancy loss
        if (df.at[individual_id, 'la_miscarriage'] >= 1) & (df.at[individual_id, 'age_years'] <= 30) & \
            (df.at[individual_id, 'la_parity'] < 1 > 3):
            rf1 = params['rr_miscarriage_prevmiscarriage']
        else:
            rf1 = 1

        if (df.at[individual_id, 'la_miscarriage'] == 0) & (df.at[individual_id, 'age_years'] >= 35) & \
            (df.at[individual_id,'la_parity'] < 1 > 3):
            rf2 = params['rr_miscarriage_35']
        else:
            rf2 = 1

        if (df.at[individual_id, 'la_miscarriage'] == 0) & (df.at[individual_id, 'age_years'] >= 31) & \
            (df.at[individual_id, 'age_years'] <= 34) & (df.at[individual_id,'la_parity'] < 1 > 3):
            rf3 = params['rr_miscarriage_3134']
        else:
            rf3 = 1

        if (df.at[individual_id, 'la_miscarriage'] == 0) & (df.at[individual_id, 'age_years'] <= 30) & \
            (df.at[individual_id,'la_parity'] >= 1 <= 3):
            rf4 = params['rr_miscarriage_grav4']
        else:
            rf4 = 1

        # Next we multiply the baseline rate of miscarriage in the reference population who are absent of riskfactors
        # by the product of the relative rates for any risk factors this mother may have
        riskfactors = rf1 * rf2 * rf3 * rf4

        if riskfactors == 1:
            eff_prob_miscarriage = params['prob_miscarriage']
        else:
            eff_prob_miscarriage = riskfactors * params['prob_miscarriage']

        # Finally a random draw is used to determine if this woman will experience a miscarriage for this pregnancy
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_miscarriage:

            # If a newly pregnant woman will miscarry her pregnancy, a random date within 20 weeks of conception is
            # generated, the date of miscarriage is scheduled for this day
            random_draw = self.sim.rng.exponential(scale=0.5, size=1)  # DETERMINE SCALE??
            random_days = pd.to_timedelta(random_draw[0], unit='d')  # TODO: Ask Tim C to explain this again
            miscarriage_date = self.sim.date + random_days
            df.at[individual_id, 'la_miscarriage_date'] = miscarriage_date

            self.sim.schedule_event(MiscarriageAndPostMiscarriageComplicationsEvent(self.module, individual_id,
                                                                                    cause='post miscarriage'),
                                    miscarriage_date)

            # And for women who do not have a miscarriage we move them to labour scheduler to determine at what
            # gestation they will go into labour
        else:
            assert df.at[individual_id, 'la_due_date'] != pd.NaT
            self.sim.schedule_event(LabourScheduler(self.module, individual_id, cause='pregnancy'), self.sim.date)


class LabourScheduler (Event, IndividualScopeEventMixin):
    """This event determines when pregnant women, who have not experienced a miscarriage, will going to labour"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # First we identify if this woman has any risk factors that predispose her to preterm labour
        if df.at[individual_id, 'la_previous_ptb']:
            rf1 = params['rr_ptl_pptb']
        else:
            rf1 = 1

        # Next we multiply the baseline risk of preterm birth in the reference population who are absent of riskfactors
        # by the product of the relative rates for any risk factors this mother may have

        riskfactors = rf1

        if riskfactors == 1:
            eff_prob_ptl = params['prob_ptl']
        else:
            eff_prob_ptl = riskfactors * params['prob_ptl']

        # Todo: review any additional risk factors for preterm birth

        # A random draw is used to determine if this woman will go into early preterm labour and she is randomly
        # allocated a gestation of between 24 and 33 weeks at which time she will be scheduled to go into labour

        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_ptl:
            early_late = ['early', 'late']
            # Here we apply the a probability that women will deliver early or late preterm
            # Todo: discuss with Tim C how best to incorporate risk factors for early/late preterm
            probabilities = [0.752, 0.248]
            random_choice = self.sim.rng.choice(early_late, size=1, p=probabilities)
            if random_choice == 'early':
                random = np.random.randint(24, 33, size=1)
                random = int(random)
                df.at[individual_id, 'la_due_date'] = df.at[individual_id, 'date_of_last_pregnancy'] + \
                                                  pd.Timedelta(random, unit='W')
                due_date = df.at[individual_id, 'la_due_date']
            else:
                random = np.random.randint(34, 37, size=1)
                random = int(random)
                df.at[individual_id, 'la_due_date'] = df.at[individual_id, 'date_of_last_pregnancy'] + \
                                                      pd.Timedelta(random, unit='W')
                due_date = df.at[individual_id, 'la_due_date']

        # If the woman will not go into preterm labour she is allocated a due date of between 37 and 44 weeks following
        # conception

        else:
            random = np.random.randint(37, 44, size=1)
            random = int(random)
            df.at[individual_id, 'la_due_date'] = df.at[individual_id, 'date_of_last_pregnancy'] +\
                                                  pd.Timedelta(random, unit='W')
            due_date= df.at[individual_id, 'la_due_date']

        # Labour is then scheduled on the newly generated due date along with the birth event 2 days after
        self.sim.schedule_event(LabourEvent(self.module, individual_id, cause='labour'), due_date)


class LabourEvent(Event, IndividualScopeEventMixin):

    """Moves a pregnant woman into labour/spontaneous abortion based on gestation distribution """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # Here we populate the maternal and newborn info dictionary with baseline values before the womans labour begins
        mni = self.module.mother_and_newborn_info
        mni[individual_id] = {'gestation_at_labour': 0,
                              'labour_state':'X',
                              'delivery_setting': 'X',
                              'risk_ol': params['prob_pl_ol'],
                              'labour_is_obstructed': 'N',
                              'risk_sepsis': params['prob_an_sepsis'],
                              'sepsis': 'N',
                              'source_sepsis': 'X',
                              'timing_sepsis': 'X',
                              'risk_aph': params['prob_an_aph'],
                              'APH': 'N',
                              'source_aph': 'X',
                              'risk_eclampsia': params['prob_an_eclampsia'],
                              'eclampsia': 'N',
                              'timing_eclampsia': 'X',
                              'risk_ur': params['prob_an_ur'],
                              'UR': 'N',
                              'risk_pph': params['prob_pn_pph'],
                              'PPH': 'N',
                              'source_pph': 'X',
                              'risk_newborn_sepsis': params['prob_neonatal_sepsis'],
                              'risk_newborn_enceph': params['prob_neonatal_enceph'],
                              'death_in_labour': 'N',
                              'stillbirth_in_labour': 'N',
                              'death_postpartum': 'N'}

# ===================================== LABOUR STATE  ==================================================================

        # Based on gestational age the woman in labour is allocated to either term, earl/late preterm or post term
        # labour

        # df.at[individual_id, 'la_complications_of_labour'] = [0, 0, 0, 0, 0, 0, 0, 0]
        gestation_date = df.at[individual_id, 'la_due_date'] - df.at[individual_id, 'date_of_last_pregnancy']
        gestation_weeks = gestation_date / np.timedelta64(1, 'W')
        gestation_weeks = int(gestation_weeks)
        mni[individual_id]['gestation_at_labour'] = gestation_weeks

        if df.at[individual_id, 'is_pregnant']:
            mni[individual_id]['gestation_at_labour'] = gestation_weeks
            df.at[individual_id, 'la_gestation_at_labour'] = gestation_weeks
            if 37 <= df.at[individual_id, 'la_gestation_at_labour'] < 42:
                df.at[individual_id, 'la_labour'] = "term_labour"

            elif 24 <= df.at[individual_id, 'la_gestation_at_labour'] < 34:
                df.at[individual_id, 'la_labour'] = "early_preterm_labour"
                df.at[individual_id, 'la_previous_ptb'] = True

            elif 37 > df.at[individual_id, 'la_gestation_at_labour'] >= 34:
                df.at[individual_id, 'la_previous_ptb'] = True

            elif df.at[individual_id, 'la_gestation_at_labour'] > 41:
                df.at[individual_id, 'la_labour'] = "post_term_labour"

    # ===================== PLACE HOLDER CARE SEEKING AND SCHEDULING (DUMMY) =====================================

        #if df.at[individual_id, 'is_pregnant']:
            # prob = self.sim.modules['HealthSystem'].get_prob_seek_care(individual_id, symptom_code=4)
            prob = 0.91  # DUMMY- will just generate 9% home birth rate
            rand = self.module.rng.rand()
            if rand < prob:
                df.at[individual_id, 'mdf_location_of_delivery'] = 'facility_delivery'
                logger.debug(
                        'This is LabourEvent, scheduling HSI_Labour_PresentsForCareInLabour for person %d',
                        individual_id)

                # TODO: check this allows complications to be allocated before the labour event
                event = HSI_Labour_PresentsForSkilledAttendanceInLabour(self.module, person_id=individual_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1)
                                                                        )
            else:
                df.at[individual_id, 'mdf_location_of_delivery'] = 'home_birth'

# ============================ INDIVIDUAL RISK OF COMPLICATIONS DURING LABOUR =========================================

        # Todo: Discuss with Helen allot impact of gestation on complications of the mother

# ===================================  OBSTRUCTED LABOUR =============================================================

        # First we identify if this woman's labour will become obstructed based on risk factors using the same
        # calculation as previous individual style events
        if (df.at[individual_id,'la_parity'] == 0) & (df.at[individual_id,'age_years'] >= 21):
            rf1 = params['rr_PL_OL_nuliparity']
        else:
            rf1 = 1

        if (df.at[individual_id,'la_parity'] == 1) & (df.at[individual_id,'age_years'] >= 21):
            rf2 = params['rr_PL_OL_para1']
        else:
            rf2 = 1

        if (df.at[individual_id,'la_parity'] > 1) & (df.at[individual_id,'la_parity'] < 3) & (df.at[individual_id, 'age_years'] < 20):
            rf3 = params['rr_PL_OL_age_less20']
        else:
            rf3 = 1

        # TODO: this formulation of the risk factors (which is used very often) might be more readble as: p_outcome =
        #  p_baseeline * (has_rf1*rr1) * (has_rf2*rr2)

        riskfactors = rf1*rf2*rf3
        eff_prob_ol = riskfactors * params['prob_pl_ol']

        if df.at[individual_id, 'mdf_location_of_delivery'] == 'facility_delivery':
            df.at[individual_id, 'mdf_risk_of_obstructed_labour'] = eff_prob_ol

        else:
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_ol:
                df.at[individual_id, 'la_obstructed_labour'] = True
                df.at[individual_id, 'la_complications_of_labour'][0] = 1

        # todo: consider applying here if we should calculate an incidence of wether this is obstruction late in labour
        #  or they have been obstructed for a while
        # Todo: Case fatality/ stillbirth for untreated obstructed labour
        # We then work through the next complications and assess if this woman will experience additional complications

# ==================================== ECLAMPSIA ======================================================================

        if (df.at[individual_id, 'la_parity'] == 0) & (df.at[individual_id, 'age_years'] <= 29):
            rf1 = params['rr_an_eclampsia_nullip']
        else:
            rf1 = 1

        if (df.at[individual_id, 'la_parity'] >= 1) & (df.at[individual_id, 'age_years'] >= 35):
            rf2 = params['rr_an_eclampsia_35']
        else:
            rf2 = 1

        if (df.at[individual_id, 'la_parity'] >= 1) & (df.at[individual_id, 'age_years'] >= 30) & \
                (df.at[individual_id, 'age_years'] <= 34):
            rf3 = params['rr_an_eclampsia_30_34']
        else:
            rf3 = 1

        riskfactors = rf1 * rf2 * rf3
        eff_prob_eclampsia = riskfactors * params['prob_an_eclampsia']

        if df.at[individual_id, 'mdf_location_of_delivery'] == 'facility_delivery':
            df.at[individual_id, 'mdf_risk_of_eclampsia'] = eff_prob_eclampsia

        else:
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_eclampsia:
                df.at[individual_id, 'la_eclampsia'] = True
                df.at[individual_id, 'la_complications_of_labour'][1] = 1

# ============================================ ANTEPARTUM HAEMORRHAGE =================================================

        if df.at[individual_id, 'li_ed_lev'] == 1:
                rf1 = params['rr_an_aph_noedu']
        else:
            rf1 = 1
#        if (df.at[individual_id, 'la_complications_of_labour'][0] == 1) & (df.at[individual_id, 'li_ed_lev'] >1):
#                rf2 = params['rr_an_aph_pl_ol']
                # Here we have applied an increased risk of APH for women whose labour was obstructed
#        else:
#            rf2 = 1

        riskfactors = rf1 #* rf2
        eff_prob_aph = riskfactors * params['prob_an_aph']

        if df.at[individual_id, 'mdf_location_of_delivery'] == 'facility_delivery':
            df.at[individual_id, 'mdf_risk_of_aph'] = eff_prob_aph
        else:
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_aph:
                df.at[individual_id, 'la_aph'] = True
                df.at[individual_id, 'la_complications_of_labour'][2] = 1

# ==========================================  SEPSIS =================================================================

        # Todo: include risk factors

      #  if df.at[individual_id, 'la_complications_of_labour'][0] == 1:
      #      rf1 = params['rr_an_sepsis_pl_ol']
      #  else:
            rf1 = 1

        riskfactors = rf1
        eff_prob_sepsis = riskfactors * params['prob_an_sepsis']

        if df.at[individual_id, 'mdf_location_of_delivery'] == 'facility_delivery':
            df.at[individual_id, 'mdf_risk_of_maternal_sepsis'] = eff_prob_sepsis

        else:
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_sepsis:
                df.at[individual_id, 'la_sepsis'] = True
                df.at[individual_id, 'la_complications_of_labour'][3] = 1

# ====================================== UTERINE RUPTURE  =============================================================

        if (df.at[individual_id, 'la_parity'] >= 5) & (df.at[individual_id, 'la_previous_cs'] == 0) & \
            ~df.at[individual_id, 'la_obstructed_labour']: # should this be standardised with la_comps_lab
            rf1 = params['rr_an_ur_grand_multip']
        else:
            rf1 = 1

        if (df.at[individual_id, 'la_parity'] < 5) & (df.at[individual_id, 'la_previous_cs'] >= 1) & \
            ~df.at[individual_id, 'la_obstructed_labour']:
            rf2 = params['rr_an_ur_prevcs']
        else:
            rf2 = 1

        if (df.at[individual_id, 'la_parity'] < 5) & (df.at[individual_id, 'la_previous_cs'] == 0) & \
            df.at[individual_id, 'la_obstructed_labour']:
            rf3 = params['rr_an_ur_ref_ol']
        else:
            rf3 = 1

        riskfactors = rf1 * rf2 * rf3
        eff_prob_ur = riskfactors * params['prob_an_ur']

        if df.at[individual_id, 'mdf_location_of_delivery'] == 'facility_delivery':
            df.at[individual_id, 'mdf_risk_of_uterine_rupture'] = eff_prob_ur

        else:
            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_ur:
                df.at[individual_id, 'la_uterine_rupture'] = True
                df.at[individual_id, 'la_complications_of_labour'][4] = 1

        # Here we schedule the birth event for 2 days after labour- we do this prior to the death event as women who
        # die but still deliver a live child will pass through birth event

        due_date = df.at[individual_id, 'la_due_date']
        self.sim.schedule_event(BirthEvent(self.module, individual_id), due_date + DateOffset(days=2))

        # For women who experience a complication we determine their risk of death in the LabourDeathEvent
        # if df.at[individual_id, 'la_uterine_rupture'] or df.at[individual_id, 'la_aph'] or \
        #    df.at[individual_id, 'la_sepsis'] or df.at[individual_id, 'la_eclampsia']:

        self.sim.schedule_event(LabourDeathEvent(self.module, individual_id, cause='labour'),
                                self.sim.date+ DateOffset(days=1))


class BirthEvent(Event, IndividualScopeEventMixin):
    """A one-off event in which a pregnant mother gives birth.
    """

    def __init__(self, module, mother_id):
        """Create a new birth event."""
        super().__init__(module, person_id=mother_id)

    def apply(self, mother_id):

        # This event tells the simulation that the woman's pregnancy is over and generates the new child in the
        # data frame
        logger.debug('@@@@ A Birth is now occuring, to mother %s', mother_id)
        df = self.sim.population.props

        # If the mother is alive and still pregnant we generate a live child and the woman is scheduled to move to the
        # postpartum event to determine if she experiences any additional complications
        if df.at[mother_id, 'is_alive'] and df.at[mother_id, 'is_pregnant']:
            self.sim.do_birth(mother_id)
            df.at[mother_id, 'la_parity'] += 1
            self.sim.schedule_event(PostpartumLabourEvent(self.module, mother_id, cause='post partum'),
                                    self.sim.date)

        # If the mother has died during childbirth the child is still generated with is_alive=false to monitor
        # stillbirth rates. She will not pass through the postpartum complication events
        if df.at[mother_id, 'is_alive'] == False & df.at[mother_id, 'is_pregnant'] == True & \
            df.at[mother_id,'la_died_in_labour'] == True:
            self.sim.do_birth(mother_id)


class PostpartumLabourEvent(Event, IndividualScopeEventMixin):

    """applies probability of postpartum complications to women who have just delivered """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

# ============================== POSTPARTUM COMPLICATIONS FOLLOWING LABOUR ============================================
        # Here we follow the same format as within the main labour event and we determine if women experience any
        # complications following delivery

# ============================================= RISK OF PPH ===========================================================

        if df.at[individual_id, 'is_alive']:

#        if df.at[individual_id, 'la_complications_of_labour'][0] == 1:
#            rf1 = params['rr_pn_pph_pl_ol']
#        else:

            rf1 = 1

            riskfactors = rf1
            eff_prob_pph = riskfactors * params['prob_pn_pph']

            if df.at[individual_id, 'mdf_location_of_delivery'] == 'facility_delivery':
                df.at[individual_id, 'mdf_risk_of_pph'] = eff_prob_pph

            else:
                random = self.sim.rng.random_sample(size=1)
                if random < eff_prob_pph:
                    df.at[individual_id, 'la_pph'] = True
                    df.at[individual_id, 'la_complications_of_labour'][5] = 1


# ============================================= RISK OF SEPSIS =========================================================

#        if df.at[individual_id, 'la_complications_of_labour'][0] == 1:
#            rf1 = params['rr_an_sepsis_pl_ol'] # should we have differnt an/pn rates?
#        else:
            rf1 = 1

            riskfactors = rf1
            eff_prob_pn_sepsis = riskfactors * params['prob_pn_sepsis']

            if df.at[individual_id, 'mdf_location_of_delivery'] == 'facility_delivery':
                df.at[individual_id, 'mdf_risk_of_maternal_sepsis'] = eff_prob_pn_sepsis

            random = self.sim.rng.random_sample(size=1)
            if random < eff_prob_pn_sepsis:
                df.at[individual_id, 'la_sepsis'] = True
                df.at[individual_id, 'la_complications_of_labour'][6] = 1


# ============================================= RISK OF ECLAMPSIA ====================================================

            if (df.at[individual_id, 'la_parity'] == 0) & (df.at[individual_id, 'age_years'] <= 29):
                rf1 = params['rr_an_eclampsia_nullip']
            else:
                rf1 = 1

            if (df.at[individual_id, 'la_parity'] >= 1) & (df.at[individual_id, 'age_years'] >= 35):
                rf2 = params['rr_an_eclampsia_35']
            else:
                rf2 = 1

            if (df.at[individual_id, 'la_parity'] >= 1) & (df.at[individual_id, 'age_years'] >= 30) & \
                (df.at[individual_id, 'age_years'] <= 34):
                rf3 = params['rr_an_eclampsia_30_34']
            else:
                rf3 = 1

            riskfactors = rf1 * rf2 * rf3
            eff_prob_eclampsia = riskfactors * params['prob_pn_eclampsia']

            if df.at[individual_id, 'mdf_location_of_delivery'] == 'facility_delivery':
                df.at[individual_id, 'mdf_risk_of_eclampsia'] = eff_prob_eclampsia

            else:
                random = self.sim.rng.random_sample(size=1)
                if random < eff_prob_eclampsia:
                    df.at[individual_id, 'la_eclampsia'] = True
                    df.at[individual_id, 'la_complications_of_labour'][7] = 1


            # For women who experience a complication we determine their risk of death in the LabourDeathEvent
#        if df.at[individual_id, 'la_eclampsia'] or df.at[individual_id, 'la_pph'] or \
#                df.at[individual_id, 'la_sepsis']:

            self.sim.schedule_event(PostPartumDeathEvent(self.module, individual_id, cause='labour'), self.sim.date)

#  =============================================== RESET LABOUR STATUS =================================================

        # Following the end of this event we reset properties related to labour to ensure future labours run correctly
        # (commented out so can view effects at this point)
    #    if df.at[individual_id, 'is_alive']:
    #        df.at[individual_id, 'la_labour'] = "not_in_labour"
    #        df.at[individual_id, 'la_due_date'] = pd.NaT
    #        df.at[individual_id, 'la_gestation_at_labour'] = 0


class MiscarriageAndPostMiscarriageComplicationsEvent(Event, IndividualScopeEventMixin):

    """applies probability of postpartum complications to women who have just experience a miscarriage """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # TODO: consider stage of pregnancy loss and its impact on likelihood of complications i.e retained product

        df.at[individual_id, 'is_pregnant'] = False
        df.at[individual_id, 'la_miscarriage'] = +1
        df.at[individual_id, 'la_due_date'] = pd.NaT

        # As with the other complication events here we determine if this woman will experience any complications
        # following her miscarriage
        riskfactors = 1  # rf1
        eff_prob_pph = riskfactors * params['prob_pn_pph']
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_pph:
            df.at[individual_id, 'la_pph'] = True

        riskfactors = 1  # rf1
        eff_prob_pn_sepsis = riskfactors * params['prob_pn_sepsis']
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_pn_sepsis:
                df.at[individual_id, 'la_sepsis'] = True

        # Currently if she has experienced any of these complications she is scheduled to pass through the DeathEvent
        # to determine if they are fatal
        if df.at[individual_id,'la_sepsis'] or df.at[individual_id, 'la_pph']:
            self.sim.schedule_event(MiscarriageDeathEvent(self.module, individual_id, cause='miscarriage'),
                                    self.sim.date)


class LabourDeathEvent (Event, IndividualScopeEventMixin):

    """handles death in labour"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # TODO: review and restructure as needed

        # Currently we apply an untreated case fatality ratio (dummy values presently) who have experienced a
        # complication
        # Similarly we apply a risk of still birth associated with each complication

        if df.at[individual_id, 'la_eclampsia']:
            random = self.sim.rng.random_sample()
            if random < params['cfr_eclampsia']:
                df.at[individual_id, 'la_died_in_labour'] = True
                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_eclampsia_md']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True
            else:
                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_eclampsia']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

        if df.at[individual_id, 'la_aph']:
            random = self.sim.rng.random_sample()
            if random < params['cfr_aph']:
                df.at[individual_id, 'la_died_in_labour'] = True
                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_aph_md']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True
            else:
                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_aph']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

        if df.at[individual_id, 'la_sepsis']:
            random = self.sim.rng.random_sample()
            if random < params['cfr_sepsis']:
                df.at[individual_id, 'la_died_in_labour'] = True
                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_sepsis_md']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True
            else:
                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_sepsis']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

        if df.at[individual_id, 'la_uterine_rupture']:
            random = self.sim.rng.random_sample()
            if random < params['cfr_uterine_rupture']:
                df.at[individual_id, 'la_died_in_labour'] = True
                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_ur_md']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True
            else:
                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_ur']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

            # Schedule death for women who die in labour

        if df.at[individual_id, 'la_died_in_labour']:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                  cause='labour'), self.sim.date)

            # Todo: consider double logging (here and demography)
            # Log the maternal death
            logger.info('%s|maternal_death|%s', self.sim.date,
                        { 'age': df.at[individual_id, 'age_years'],
                            'person_id': individual_id })

        # DUMMY- For women who survives failed treatment of complications we schedule post partum care
        if ~df.at[individual_id, 'la_died_in_labour']:
            event = HSI_Labour_ReceivesCareForPostpartumPeriod(self.module, person_id=individual_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date
                                                                )


class PostPartumDeathEvent (Event, IndividualScopeEventMixin):

    """handles death following labour"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # We apply the same structure as with the LabourDeathEvent to women who experience postpartum complications
        if df.at[individual_id, 'la_eclampsia']:
            random = self.sim.rng.random_sample()
            if random < params['cfr_pn_eclampsia']:
                df.at[individual_id, 'la_died_in_labour'] = True

        if df.at[individual_id, 'la_pph']:
            random = self.sim.rng.random_sample()
            if random < params['cfr_pn_pph']:
                df.at[individual_id, 'la_died_in_labour'] = True

        if df.at[individual_id, 'la_sepsis']:
            random = self.sim.rng.random_sample()
            if random < params['cfr_pn_sepsis']:
                df.at[individual_id, 'la_died_in_labour'] = True

        if df.at[individual_id, 'la_died_in_labour']:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                  cause='postpartum labour'), self.sim.date)
            logger.info('%s|maternal_death|%s', self.sim.date,
                        {'age': df.at[individual_id, 'age_years'],
                            'person_id': individual_id})


class MiscarriageDeathEvent (Event, IndividualScopeEventMixin):

    """handles death following miscarriage"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        if df.at[individual_id, 'la_pph']:
            random = self.sim.rng.random_sample()
            if random < params['cfr_pn_pph']:
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause=' complication of miscarriage'),
                                        self.sim.date)

        if df.at[individual_id, 'la_sepsis']:
            random = self.sim.rng.random_sample()
            if random < params['cfr_pn_sepsis']:
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='complication of miscarriage'),
                                        self.sim.date)

                logger.info('%s|maternal_death|%s', self.sim.date,
                        {
                            'age': df.at[individual_id, 'age_years'],
                            'person_id': individual_id
                        })

# ======================================================================================================================
# ================================ HEALTH SYSTEM INTERACTION EVENTS ====================================================
# ======================================================================================================================

class HSI_Labour_PresentsForSkilledAttendanceInLabour(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This interaction manges a woman's initial presentation to a health facility when in labour
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['NormalDelivery'] = 1  # TODO: reset to complicated delivery if complications occur

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_sba_uncomp = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] ==
                                              'Vaginal delivery - skilled attendance',
                                              'Intervention_Pkg_Code'])[0]

        # TODO: reset above if complicated delivery if complications occur
        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code_sba_uncomp],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_Attends_For_Delivery'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters

        logger.debug('This is HSI_Labour_PresentsForSkilledAttendanceInLabour, labour and delivery care for person %d',
                     person_id)

        # TODO: Who is attending this labour?- Will I determine this or will it link to capacity, need to apply effect
        # TODO: worth storing WHERE (what level) health facility each womans last labour took place?

# ====================================== BOOKED DELIVERY ==========================================================
    # Here we deal with interventions and scheduling for planned delivery such as induction/ elective CS

#   INDUCTION (post term) (check as has multiple indications)

# ====================================== UNBOOKED DELIVERY ==========================================================
    # Here we deal with interventions for women who present in labour
#
        #   CLEAN DELIVERY PRACTICES:

        adjusted_maternal_sepsis_risk = df.at[person_id, 'mdf_risk_of_maternal_sepsis'] * \
                                        params['rr_maternal_sepsis_clean_delivery']
        df.at[person_id, 'mdf_risk_of_maternal_sepsis'] = adjusted_maternal_sepsis_risk

        #   MATERNAL/FETAL SURVEILLANCE:
            # limited evidence of any effect (partograph)

        #   ?AUGMENTATION OF LABOUR (this may not be needed)

# ================================== INTERVENTIONS FOR PRE EXSISTING CONDITIONS =====================================

#   SEVERE PRE ECLAMPSIA (prophylactic MgSO4?/ Induction/ CS)
#   PRETERM INTERVENTIONS (abx?, mgso4?)
#   ANTIBIOTICS FOR PROMs- Is in guidelines

# ===================================  COMPLICATION OF LABOUR ========================================================
    # Here, using the adjusted risks calculated following 'in-labour' interventions to determine which complications a
    # woman may experience and store those in the dataframe

    # WILL NEED TO APPLY RISK OF COMPLICATION IN LIGHT OF OTHER COMPLICATIONS
       # TODO: link between obstructed labour and uterine rupture?

        random = self.sim.rng.random_sample()
        if random < df.at[person_id, 'mdf_risk_of_obstructed_labour']:
            df.at[person_id, 'la_obstructed_labour'] = True


# ==================================== SCHEDULE HEALTH SYSTEM INTERACTIONS ===========================================
    # Here, if a woman has developed a complication, she is scheduled to receive any care she may need

        if df.at[person_id,'la_obstructed_labour']:
            event = HSI_Labour_ReceivesCareForObstructedLabour(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                        priority=0,
                                                        topen=self.sim.date,
                                                        tclose=self.sim.date + DateOffset(days=1)
                                                        )

        if df.at[person_id,'la_sepsis']:
            event = HSI_Labour_ReceivesCareForMaternalSepsis(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                        priority=0,
                                                        topen=self.sim.date,
                                                        tclose=self.sim.date + DateOffset(days=1)
                                                        )

        if df.at[person_id, 'la_aph']:
            event = HSI_Labour_ReceivesCareForMaternalHaemorrhage(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1)
                                                                    )

        if df.at[person_id, 'la_eclampsia']:
            event = HSI_Labour_ReceivesCareForEclampsia(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1)
                                                                    )



class HSI_Labour_ReceivesCareForObstructedLabour(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It manages the treatment of obstructed labour and referral in the instance of failed treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1  # TODO: to be discussed with TH/TC best appt footprint to use

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_obstructed_labour = pd.unique(consumables.loc[consumables[
                                                                   'Intervention_Pkg'] ==
                                                               'Management of obstructed labour',
                                                               'Intervention_Pkg_Code'])[0]
        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code_obstructed_labour],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_Receives_Care_For_Obstructed_Labour'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters

        logger.debug('This is HSI_Labour_ReceivesCareForObstructedLabour, management of obstructed labour for '
                     'person %d',
                     person_id)

# =====================================  OBSTRUCTED LABOUR TREATMENT ==================================================

        # Differentiate between CPD and other?
        # todo: Record type of AVD, also await answer on non-consumable equipment

        if df.at[person_id, 'la_obstructed_labour']:
            treatment_effect = params['prob_deliver_ventouse']
            random = self.sim.rng.random_sample()
            if treatment_effect > random:
                df.at[person_id, 'la_obstructed_labour'] = False
                df.at[person_id, 'la_delivery_mode'] = 'AVD'
            else:
                treatment_effect = params['prob_deliver_forceps']
                random = self.sim.rng.random_sample()
                if treatment_effect > random:
                    df.at[person_id, 'la_obstructed_labour'] = False
                    df.at[person_id, 'la_delivery_mode'] = 'AVD'
            #   else:
                # schedule caesarean
                # GIVE ANTIBIOTICS
            # Symphysiotomy???

class HSI_Labour_ReceivesCareForMaternalSepsis(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It manages the
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1  # TODO: to be discussed with TH/TC best appt footprint to use

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_sepsis = pd.unique(consumables.loc[consumables[
                                                        'Intervention_Pkg'] ==
                                                    'Maternal sepsis case management',
                                                    'Intervention_Pkg_Code'])[0]
        the_cons_footprint = {
                    'Intervention_Package_Code': [pkg_code_sepsis],
                    'Item_Code': []
                }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_Receives_Care_For_Maternal_Sepsis'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters

        logger.debug('This is HSI_Labour_ReceivesCareForMaternalSepsis, management of obstructed labour for '
                             'person %d',
                             person_id)

# =====================================  MATERNAL SEPSIS TREATMENT ====================================================

        if df.at[person_id, 'la_sepsis']:
            treatment_effect = params['prob_cure_antibiotics']
            random = self.sim.rng.random_sample()
            if treatment_effect > random:
                df.at[person_id, 'la_sepsis'] = False


class HSI_Labour_ReceivesCareForEclampsia(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It manages the
        """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1  # TODO: to be discussed with TH/TC best appt footprint to use

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_eclampsia = pd.unique(consumables.loc[consumables[
                                                           'Intervention_Pkg'] ==
                                                       'Management of eclampsia',
                                                       'Intervention_Pkg_Code'])[0]
        the_cons_footprint = {
                    'Intervention_Package_Code': [pkg_code_eclampsia],
                    'Item_Code': []
                }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_Receives_Care_For_Eclampsia'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters

        logger.debug('This is HSI_Labour_ReceivesCareForEclampsia, management of obstructed labour for '
                             'person %d',
                             person_id)
# =======================================  ECLAMPSIA TREATMENT ========================================================

        if df.at[person_id, 'la_sepsis']:
            treatment_effect = params['prob_cure_mgso4']
            random = self.sim.rng.random_sample()
            if treatment_effect > random:
                df.at[person_id, 'la_eclampsia'] = False


class HSI_Labour_ReceivesCareForMaternalHaemorrhage(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It manages the
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1  # TODO: to be discussed with TH/TC best appt footprint to use

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_pph = pd.unique(consumables.loc[consumables[
                                                     'Intervention_Pkg'] ==
                                                 'Treatment of postpartum hemorrhage',
                                                 'Intervention_Pkg_Code'])[0]

        item_code_aph1 = pd.unique(consumables.loc[consumables['Items'] == 'Blood, one unit', 'Item_Code'])[0]
        item_code_aph2 = pd.unique(consumables.loc[consumables['Items'] == 'Lancet, blood, disposable', 'Item_Code'])[0]
        item_code_aph3 = pd.unique(consumables.loc[consumables['Items'] == 'Test, hemoglobin', 'Item_Code'])[0]
        item_code_aph4 = pd.unique(consumables.loc[consumables['Items'] == 'IV giving/infusion set, with needle',
                                                   'Item_Code'])[0]
        item_code_aph5 = pd.unique(consumables.loc[consumables['Items'] == 'Gloves, surgeon’s, latex, disposable,'
                                                                           ' sterile, pair', 'Item_Code'])[0]


        the_cons_footprint = {
                    'Intervention_Package_Code': [pkg_code_pph],
                    'Item_Code': [item_code_aph1, item_code_aph2, item_code_aph3, item_code_aph4, item_code_aph5]
                }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_Receives_Care_For_Maternal_Haemorrhage'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters

        logger.debug('This is HSI_Labour_ReceivesCareForMaternalHaemorrhag, management of obstructed labour for '
                         'person %d',
                         person_id)

# ===================================  ANTEPARTUM HAEMORRHAGE TREATMENT ===============================================
    # TODO: Do we apply a severity of PP/PA in which we can just give blood and deliver vaginally? (As per Helen A)
    # TODO: Here we'll schedule a caesarean for severe cases

# ======================================= POSTPARTUM HAEMORRHAGE ======================================================
        # First we use a probability weighted random draw to determine the underlying etiology of this womans PPH

        if df.at[person_id, 'la_pph']:
            etiology = ['uterine atony', 'retained products']
            probabilities = [0.67, 0.33]  # dummy
            random_choice = self.sim.rng.choice(etiology, size=1, p=probabilities)
            # Todo: also consider here if we should add a level of severity of PPH

            #       TREATMENT CASCADE FOR ATONIC UTERUS:

            # Here we use a treatment cascade adapted from Malawian Obs/Gynae guidelines
            # Women who are bleeding due to atonic uterus first undergo medical management, oxytocin IV, misoprostol PR
            # and uterine massage in an attempt to stop bleeding

            if random_choice == 'uterine atony':
                random = self.sim.rng.random_sample()
                if params['prob_cure_oxytocin'] > random:
                    df.at[person_id, 'la_pph'] = False
                else:
                    random = self.sim.rng.random_sample()
                    if params['prob_cure_misoprostol'] > random:
                        df.at[person_id, 'la_pph'] = False
                    else:
                        random = self.sim.rng.random_sample()
                        if params['prob_cure_uterine_massage'] > random:
                            df.at[person_id, 'la_pph'] = False
                        else:
                            random = self.sim.rng.random_sample()
                            if params['prob_cure_uterine_tamponade'] > random:
                                df.at[person_id, 'la_pph'] = False
                        # HERE WE WOULD SCHEDULE SURGICAL MANAGEMENT
            # Todo: consider the impact of oxy + miso + massage as ONE value, Discuss with expert

            #       TREATMENT CASCADE FOR RETAINED PRODUCTS/PLACENTA:
            if random_choice == 'retained products':
                random = self.sim.rng.random_sample()
                if params['prob_cure_manual_removal'] > random:
                    df.at[person_id, 'la_pph'] = False
                    # HERE WE WOULD SCHEDULE SURGICAL MANAGEMENT

# ===================================  UTERINE RUPTURE TREATMENT ======================================================
    # TODO: Do we apply a severity of UR in which we can just give blood and deliver vaginally? (As per Helen A)
    # TODO: Here we'll schedule a surgery +/-caesarean for severe cases

# ========================================SCHEDULING DEATH EVENT FOR FAILED TREATMENT =================================

#        if df.at[person_id, 'la_uterine_rupture'] or df.at[person_id, 'la_aph'] or \
#            df.at[person_id, 'la_sepsis'] or df.at[person_id, 'la_eclampsia']:
#            self.sim.schedule_event(LabourDeathEvent(self.module, person_id, cause='labour'), self.sim.date)

# ======================== SCHEDULING POSTPARTUM CARE EVENT FOR WOMEN FOLLOWING SUCCESSFUL TREATMENT ==================

# DUMMY- For women who survives failed treatment of complications we schedule post partum care
#        if ~df.at[person_id, 'la_uterine_rupture'] and ~df.at[person_id, 'la_aph'] and ~df.at[person_id, 'la_sepsis'] \
#            and ~df.at[person_id, 'la_eclampsia']:
#                event = HSI_Labour_ReceivesCareForPostpartumPeriod(self.module, person_id=person_id)
#                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
#                                                                priority=0,
#                                                                topen=self.sim.date,
#                                                                tclose=self.sim.date
#                                                                )


class HSI_Labour_ReceivesCareForPostpartumPeriod(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This event manages the Health System Interaction for a woman who needs to be referred to undergo an emergency
    casarean section
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['AccidentsandEmerg'] = 1  # TODO: currently this will count deliveries twice?

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # Todo: review:
        pkg_code_post_partum_care =pd.unique(consumables.loc[consumables[
                                                     'Intervention_Pkg'] ==
                                                 'Active management of the 3rd stage of labour',
                                                 'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code_post_partum_care],
            'Item_Code': []
        }

    # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_Attends_For_Postpartum_Care'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters

    #  ACTIVE MANAGEMENT OF THE THIRD STAGE
    # Cord care - YES is in treatment guidelines
    # chlorhexidine
    # start breast feeding?

        pass

class HSI_Labour_ReferredForSurgicalCareInLabour(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This event manages the Health System Interaction for a woman who needs to be referred to undergo an emergency
    surgical management of complications arising in labour, in the postpartum period or for caesarean section
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        the_appt_footprint['MajorSurg'] = 1  # this appt could be used for uterine repair/pph management
        the_appt_footprint['Csection'] = 1

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_cs = pd.unique(consumables.loc[consumables[
                                                        'Intervention_Pkg'] ==
                                                'Cesearian Section with indication (with complication)',  # or without?
                                                'Intervention_Pkg_Code'])[0]

        # pkg_code_uterine_repair
        # pkg_code_pph_surgery
        # +/- hysterectomy?

        the_cons_footprint = {
            'Intervention_Package_Code': [pkg_code_cs],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_Surgery_for_Labour_Complications'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [2, 3]  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props
        params = self.module.parameters

# ====================================== EMERGENCY CAESAREAN SECTION ==================================================
        if df.at[person_id, ' la_uterine_rupture'] or df.at[person_id,'la_aph'] or df.at[person_id,
                                                                                        'la_obstructed_labour']:
            # Consider all indications (elective)
            df.at[person_id, 'la_delivery_mode'] = 'EmCS'
            df.at[person_id, 'la_aph'] = False   # Should this be the case?
            df.at[person_id, 'la_obstructed_labour'] = False

# ====================================== UTERINE REPAIR ==============================================================
        # Next we determine if the uterus can be repaired surgically
        if df.at[person_id,' la_uterine_rupture']:
            random = self.sim.rng.random_sample()
            if params['prob_cure_uterine_repair'] > random:
                df.at[person_id, 'la_uterine_rupture'] = False
        # In the instance of failed surgical repair, the woman undergoes a hysterectomy
            else:
                random = self.sim.rng.random_sample()
                if params['prob_cure_hysterectomy'] > random:
                    df.at[person_id, 'la_uterine_rupture'] = False

# ================================== SURGERY FOR UNCONTROLLED POSTPARTUM HAEMORRHAGE ==================================

        if df.at[person_id, 'la_pph']:
            random = self.sim.rng.random_sample()
            if params['prob_cure_uterine_ligation'] > random:
                df.at[person_id, 'la_pph'] = False
                df.at[person_id, 'hm_pph_treat_received'] = True
            else:
                random = self.sim.rng.random_sample()
                if params['prob_cure_b_lych'] > random:
                    df.at[person_id, 'la_pph'] = False
                    df.at[person_id, 'hm_pph_treat_received'] = True
                else:
                    random = self.sim.rng.random_sample()
                    # Todo: similarly consider bunching surgical interventions
                    if params['prob_cure_hysterectomy'] > random:
                        df.at[person_id, 'la_pph'] = False

# ==================================  SCHEDULING DEATH EVENT FOR FAILED TREATMENT (INTRAPARTUM)  =======================

#       ? Currently its practically impossible to die following intrapartum treatment... which isnt true

#        if df.at[person_id, 'la_uterine_rupture'] or df.at[person_id, 'la_aph'] or \
#            df.at[person_id, 'la_sepsis'] or df.at[person_id, 'la_eclampsia']:
#            self.sim.schedule_event(LabourDeathEvent(self.module, person_id, cause='labour'), self.sim.date)

# ==================================  SCHEDULING DEATH EVENT FOR FAILED TREATMENT (POSTPARTUM)  =======================

        if df.at[person_id, 'la_pph']:
            self.sim.schedule_event(PostPartumDeathEvent(self.module, person_id, cause='labour'), self.sim.date)

# TODO: Need to be certain which women are coming to this event postpartum and those that are coming intrapartum to
#  make sure scheduling is right


class LabourLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles lifestyle logging"""
    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(days=self.repeat))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props

        logger.debug('%s|person_one|%s',
                          self.sim.date, df.loc[0].to_dict())

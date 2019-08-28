"""
A model of Labour and the Health System Interactions associated with Skilled Birth Attendance
"""
import logging
import os

import pandas as pd
import numpy as np
from pathlib import Path

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography, healthsystem, healthburden, antenatal_care

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
LOG_FILENAME = 'labour.log'
logging.basicConfig(filename=LOG_FILENAME,
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)


class Labour (Module):

    """
    This module models labour, delivery and the immediate postpartum period. It  generates the properties for a woman's
    obstetric history"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath
        self.mother_and_newborn_info = dict()

    PARAMETERS = {

     #  ===================================  NATURAL HISTORY PARAMETERS ===============================================

        'prob_pregnancy': Parameter(
            Types.REAL, 'baseline probability of pregnancy'),  # DUMMY PARAMETER
        'prob_prom': Parameter(
            Types.REAL, 'probability of a woman in term labour having had experience prolonged rupture of membranes'),
        'prob_pl_ol': Parameter(
            Types.REAL, 'probability of a woman entering prolonged/obstructed labour'),
        'rr_PL_OL_nuliparity': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if they are nuliparous'),
        'rr_PL_OL_age_less20': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her age is less'
                        'than 20 years'),
        'prob_ptl': Parameter (
            Types.REAL, 'probability of a woman entering labour at <37 weeks gestation'),
        'prob_early_ptb': Parameter(
            Types.REAL, 'probability of a woman going into preterm labour between 28-33 weeks gestation'),
        'rr_early_ptb_age<20': Parameter(
            Types.REAL, 'relative risk of early preterm labour for women younger than 20'),
        'rr_early_ptb_prev_ptb': Parameter(
            Types.REAL, 'relative risk of early preterm labour for women who have previously delivered preterm'),
        'rr_early_ptb_anaemia': Parameter(
            Types.REAL, 'relative risk of preterm labour for suffering from anaemia'),
        'prob_late_ptb': Parameter(
            Types.REAL, 'probability of a woman going into preterm labour between 33-36 weeks gestation'),
        'rr_late_ptb_prev_ptb': Parameter(
            Types.REAL, 'relative risk of preterm labour for women younger than 20'),
        'prob_potl': Parameter(
            Types.REAL, 'probability of a woman entering labour at >42 weeks gestation'),
        'prob_ip_eclampsia': Parameter(
            Types.REAL, 'probability of an eclamptic seizure during labour'),
        'prob_aph': Parameter(
            Types.REAL, 'probability of an antepartum haemorrhage during labour'),
        'prob_ip_sepsis': Parameter(
            Types.REAL, 'probability of sepsis in labour'),
        'prob_uterine_rupture': Parameter(
            Types.REAL, 'probablity of a uterine rupture during labour'),
        'rr_ur_grand_multip': Parameter(
            Types.REAL, 'relative risk of uterine rupture in women who have delivered >4 times previously'),
        'rr_ur_prev_cs': Parameter(
            Types.REAL, 'relative risk of uterine rupture in women who have previously delivered via caesarean section'),
        'rr_ur_ref_ol': Parameter(
            Types.REAL,
            'relative risk of uterine rupture in women who have been referred in obstructed labour'),
        'rr_ip_sepsis_pl_ol': Parameter(
            Types.REAL, 'relative risk of developing sepsis following obstructed labour'),
        'rr_ip_eclampsia_30_34': Parameter(
            Types.REAL, 'relative risk of eclampsia for women ages between 30 and 34'),
        'rr_ip_eclampsia_35': Parameter(
            Types.REAL, 'relative risk of eclampsia for women ages older than 35'),
        'rr_ip_eclampsia_nullip': Parameter(
            Types.REAL, 'relative risk of eclampsia for women who have not previously delivered a child'),
        'rr_ip_sepsis_anc_4': Parameter(
            Types.REAL, 'relative risk of sepsis for women who have attended greater than 4 ANC visits'),
        'rr_ip_aph_noedu': Parameter(
            Types.REAL, 'relative risk of antepartum haemorrhage for women with education of primary level or lower'),
        'rr_aph_pl_ol': Parameter(
            Types.REAL, 'relative risk of antepartum haemorrhage following obstructed labour'),
        'prob_cord_prolapse': Parameter(
            Types.REAL, 'probability of this woman experiencing a cord prolapse'),
        'cfr_obstructed_labour': Parameter(
            Types.REAL, 'case fatality rate for obstructed labour'),
        'cfr_aph': Parameter(
            Types.REAL, 'case fatality rate for antepartum haemorrhage during labour'),
        'cfr_eclampsia': Parameter(
            Types.REAL, 'case fatality rate for eclampsia during labours'),
        'cfr_sepsis': Parameter(
            Types.REAL, 'case fatality rate for sepsis during labour'),
        'cfr_uterine_rupture': Parameter(
            Types.REAL, 'case fatality rate for uterine rupture in labour'),
        'prob_still_birth_obstructed_labour': Parameter(
            Types.REAL, 'probability of a still birth following obstructed labour where the mother survives'),
        'prob_still_birth_obstructed_labour_md': Parameter(
            Types.REAL, 'probability of a still birth following obstructed labour where the mother dies'),
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
        'prob_pp_eclampsia': Parameter(
            Types.REAL, 'probability of eclampsia following delivery for women who were in spotaneous unobstructed '
                        'labour'),
        'prob_pph': Parameter(
            Types.REAL, 'probability of an postpartum haemorrhage following labour'),
        'rr_pph_pl_ol': Parameter(
            Types.REAL, 'relative risk of postpartum haemorrhage following obstructed labour'),
        'prob_pp_sepsis': Parameter(
            Types.REAL, 'probability of sepsis following delivery'),
        'cfr_pph': Parameter(
            Types.REAL, 'case fatality rate for postpartum haemorrhages'),
        'cfr_pp_eclampsia': Parameter(
            Types.REAL, 'case fatality rate for eclampsia following delivery'),
        'cfr_pp_sepsis': Parameter(
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
        'prob_neonatal_birth_asphyxia': Parameter(
            Types.REAL, 'baseline probability of a child developing neonatal encephalopathy following delivery'),

        # ================================= TREATMENT PARAMETERS =====================================================

        'prob_successful_induction': Parameter(
            Types.REAL, 'probability of that induction of labour will be successful'),
        'rr_maternal_sepsis_clean_delivery': Parameter(
            Types.REAL, 'relative risk of maternal sepsis following clean birth practices employed in a facility'),
        'rr_newborn_sepsis_clean_delivery': Parameter(
            Types.REAL, 'relative risk of newborn sepsis following clean birth practices employed in a facility'),
        'rr_sepsis_post_abx_prom': Parameter(
            Types.REAL, 'relative risk of maternal sepsis following prophylatic antibiotics for PROM in a facility'),
        'rr_newborn_sepsis_proph_abx': Parameter(
            Types.REAL, 'relative risk of newborn sepsis following prophylatic antibiotics for '
                        'premature labour in a facility'),
        'rr_pph_amtsl': Parameter(
            Types.REAL, 'relative risk of severe post partum haemorrhage following active management of the third '
                        'stage of labour'),
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

        'la_due_date_current_pregnancy': Property(Types.DATE, 'The date on which a newly pregnant woman is scheduled to'
                                                              ' go into labour'),
        'la_labour_current_pregnancy': Property(Types.CATEGORICAL, 'not in labour, Term labour, Early Preterm Labour, '
                                                                   'Late Preterm Labour, Post term labour',
                              categories=['not_in_labour', 'term_labour', 'early_preterm_labour', 'late_preterm_labour',
                                          'post_term_labour']),
        'la_current_labour_successful_induction': Property(Types.CATEGORICAL, 'Not Induced, Successful Induction, '
                                                                              'Failed Induction',
                                                     categories=['not_induced', 'successful_induction',
                                                                 'failed_induction']),
        'la_still_birth_current_pregnancy': Property(Types.BOOL,'whether this womans most recent pregnancy has ended '
                                                                'in a stillbirth'),
        #  TODO: work out if we need this property in main DF, or we could store number of still births per woman?
        #   (and log still births by type)
        'la_parity': Property(Types.INT, 'total number of previous deliveries'),
        'la_total_deliveries_by_cs': Property(Types.INT, 'number of previous deliveries by caesarean section'),
        'la_has_previously_delivered_preterm': Property(Types.BOOL, 'whether the woman has had a previous preterm '
                                                                    'delivery for any of her previous deliveries'),
        'la_obstructed_labour':Property(Types.BOOL, 'whether this womans labour has become obstructed'),
        'la_aph': Property(Types.BOOL, 'whether the woman has experienced an antepartum haemorrhage in this delivery'),
        'la_uterine_rupture': Property(Types.BOOL, 'whether the woman has experienced uterine rupture in this delivery'),
        'la_sepsis': Property(Types.BOOL, 'whether the woman has developed sepsis associated with in this delivery'),
        'la_eclampsia': Property(Types.BOOL, 'whether the woman has experienced an eclamptic seizure in this delivery'),
        'la_pph': Property(Types.BOOL, 'whether the woman has experienced an postpartum haemorrhage in this delivery'),
        'la_maternal_death':Property(Types.BOOL,' whether the woman has died as a result of this pregnancy'), # DUMMY
        'la_maternal_death_date': Property(Types.DATE, 'date of death for a date in pregnancy')  # DUMMY


    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_LabourSkilledBirthAttendance.xlsx',
                          sheet_name='parameter_values')

        dfd.set_index('parameter_name', inplace=True)

        #  ===================================  NATURAL HISTORY PARAMETERS ============================================

        # TODO: rename so parameters reflect the actual measure being employed (incidence rates/prevelance etc)

        params['prob_pregnancy'] = dfd.loc['prob_pregnancy', 'value']
        params['prob_prom'] = dfd.loc['prob_prom', 'value']
        params['prob_pl_ol'] = dfd.loc['prob_pl_ol', 'value']
        params['rr_PL_OL_nuliparity'] = dfd.loc['rr_PL_OL_nuliparity', 'value']
        params['rr_PL_OL_para1'] = dfd.loc['rr_PL_OL_para1', 'value']
        params['rr_PL_OL_age_less20'] = dfd.loc['rr_PL_OL_age_less20', 'value']
        params['prob_ptl'] = dfd.loc['prob_ptl', 'value']
        params['prob_early_ptb'] = dfd.loc['prob_early_ptb', 'value']  # rough calc - should use 2014 global estimates?
        params['rr_early_ptb_age<20'] = dfd.loc['rr_early_ptb_age<20', 'value']
        params['rr_early_ptb_prev_ptb'] = dfd.loc['rr_early_ptb_prev_ptb', 'value']
        params['rr_early_ptb_anaemia'] = dfd.loc['rr_early_ptb_anaemia', 'value']
        params['prob_late_ptb'] = dfd.loc['prob_late_ptb', 'value'] # rough calc - should use 2014 global estimates?
        params['rr_late_ptb_prev_ptb'] = dfd.loc['rr_late_ptb_prev_ptb', 'value']
        params['rr_ptl_pptb'] = dfd.loc['rr_ptl_pptb', 'value']
        params['prob_potl'] = dfd.loc['prob_potl', 'value']  # (incidence 32/1000 LBs)
        params['prob_ip_eclampsia'] = dfd.loc['prob_ip_eclampsia', 'value']
        params['prob_aph'] = dfd.loc['prob_aph', 'value']
        params['prob_ip_sepsis'] = dfd.loc['prob_ip_sepsis', 'value']
        params['rr_ip_sepsis_pl_ol'] = dfd.loc['rr_ip_sepsis_pl_ol', 'value']
        params['prob_uterine_rupture'] = dfd.loc['prob_uterine_rupture', 'value']
        params['rr_ur_grand_multip'] = dfd.loc['rr_ur_grand_multip', 'value']
        params['rr_ur_prev_cs'] = dfd.loc['rr_ur_prev_cs', 'value']
        params['rr_ur_ref_ol'] = dfd.loc['rr_ur_ref_ol', 'value'] # REVIEW "obstructed but not referred"
        params['rr_ip_eclampsia_30_34'] = dfd.loc['rr_ip_eclampsia_30_34', 'value']
        params['rr_ip_eclampsia_35'] = dfd.loc['rr_ip_eclampsia_35', 'value']
        params['rr_ip_eclampsia_nullip'] = dfd.loc['rr_ip_eclampsia_nullip', 'value']
        params['rr_ip_sepsis_anc_4'] = dfd.loc['rr_ip_sepsis_anc_4', 'value']
        params['rr_ip_aph_noedu'] = dfd.loc['rr_ip_aph_noedu', 'value']
        params['rr_aph_pl_ol'] = dfd.loc['rr_aph_pl_ol', 'value']
        params['prob_cord_prolapse'] = dfd.loc['prob_cord_prolapse', 'value']
        params['cfr_obstructed_labour'] = dfd.loc['cfr_obstructed_labour', 'value']  # dummy
        params['cfr_aph'] = dfd.loc['cfr_aph', 'value']
        params['cfr_eclampsia'] = dfd.loc['cfr_eclampsia', 'value']
        params['cfr_sepsis'] = dfd.loc['cfr_sepsis', 'value']
        params['cfr_uterine_rupture'] = dfd.loc['cfr_uterine_rupture', 'value']
        params['prob_still_obstructed_labour'] = dfd.loc['prob_still_obstructed_labour', 'value']  # dummy
        params['prob_still_birth_obstructed_labour_md'] = dfd.loc['prob_still_birth_obstructed_labour_md', 'value']  # dummy
        params['prob_still_birth_aph'] = dfd.loc['prob_still_birth_aph', 'value']
        params['prob_still_birth_aph_md'] =dfd.loc['prob_still_birth_aph_md', 'value']
        params['prob_still_birth_sepsis'] = dfd.loc['prob_still_birth_sepsis', 'value']
        params['prob_still_birth_sepsis_md'] = dfd.loc['prob_still_birth_sepsis_md', 'value']
        params['prob_still_birth_ur'] = dfd.loc['prob_still_birth_ur', 'value']
        params['prob_still_birth_ur_md'] = dfd.loc['prob_still_birth_ur_md', 'value']
        params['prob_still_birth_eclampsia'] = dfd.loc['prob_still_birth_eclampsia', 'value']
        params['prob_still_birth_eclampsia_md'] = dfd.loc['prob_still_birth_eclampsia_md', 'value']
        params['prob_pp_eclampsia'] =dfd.loc['prob_pp_eclampsia', 'value']
        params['prob_pph'] = dfd.loc['prob_pph', 'value']
        params['rr_pph_pl_ol'] = dfd.loc['rr_pph_pl_ol', 'value']
        params['prob_pp_sepsis'] = dfd.loc['prob_pp_sepsis', 'value']
        params['cfr_pph'] = dfd.loc['cfr_pph', 'value']
        params['cfr_pp_eclampsia'] = dfd.loc['cfr_pp_eclampsia', 'value']
        params['cfr_pp_sepsis'] = dfd.loc['cfr_pp_sepsis', 'value']
        params['prob_neonatal_sepsis'] = dfd.loc['prob_neonatal_sepsis', 'value']
        params['prob_neonatal_birth_asphyxia'] = dfd.loc['prob_neonatal_birth_asphyxia', 'value']

        # ================================= TREATMENT PARAMETERS =====================================================

        params['prob_successful_induction'] = dfd.loc['prob_successful_induction', 'value']  # norwegien study
        params['rr_maternal_sepsis_clean_delivery'] = dfd.loc['rr_maternal_sepsis_clean_delivery', 'value']  # dummy
        params['rr_newborn_sepsis_clean_delivery'] = dfd.loc['rr_newborn_sepsis_clean_delivery', 'value'] # dummy
        params['rr_sepsis_post_abx_prom'] = dfd.loc['rr_sepsis_post_abx_prom', 'value'] # dummy
        params['rr_newborn_sepsis_proph_abx'] = dfd.loc['rr_newborn_sepsis_proph_abx', 'value'] # dummy
        params['rr_pph_amtsl'] = dfd.loc['rr_pph_amtsl', 'value']  # cochrane (for SEVERE pph)
        params['prob_cure_antibiotics'] = dfd.loc['prob_cure_antibiotics', 'value']  # dummy
        params['prob_cure_mgso4'] = dfd.loc['prob_cure_mgso4', 'value']
        # probability taken from RR of 0.43for additional seizures (vs diazepam alone)
        params['prob_prevent_mgso4'] = dfd.loc['prob_prevent_mgso4', 'value']
        # Risk reduction of eclampsia in women who have pre-eclampsia
        params['prob_cure_diazepam'] = dfd.loc['prob_cure_diazepam', 'value']
        params['prob_cure_blood_transfusion'] = dfd.loc['prob_cure_blood_transfusion', 'value']  # dummy
        params['prob_cure_oxytocin'] = dfd.loc['prob_cure_oxytocin', 'value'] # dummy
        params['prob_cure_misoprostol'] = dfd.loc['prob_cure_misoprostol', 'value']  # dummy
        params['prob_cure_uterine_massage'] = dfd.loc['prob_cure_uterine_massage', 'value']  # dummy
        params['prob_cure_uterine_tamponade'] = dfd.loc['prob_cure_uterine_tamponade', 'value'] # dummy
        params['prob_cure_uterine_ligation'] = dfd.loc['prob_cure_uterine_ligation', 'value'] # dummy
        params['prob_cure_b_lych'] = dfd.loc['prob_cure_b_lych', 'value'] # dummy
        params['prob_cure_hysterectomy'] = dfd.loc['prob_cure_hysterectomy', 'value']  # dummy
        params['prob_cure_manual_removal'] = dfd.loc['prob_cure_manual_removal', 'value']  # dummy
        params['prob_cure_uterine_repair'] = dfd.loc['prob_cure_uterine_repair', 'value']  # dummy
        params['prob_deliver_ventouse'] = dfd.loc['prob_deliver_ventouse', 'value']  # dummy
        params['prob_deliver_forceps'] = dfd.loc['prob_deliver_forceps', 'value']  # dummy

        # Here we will include DALY weights if applicable...

        if 'HealthBurden' in self.sim.modules.keys():
            params['daly_wt_haemorrhage_moderate'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=339)
            params['daly_wt_haemorrhage_severe'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=338)
            params['daly_wt_maternal_sepsis'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=340)
            params['daly_wt_eclampsia'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=347)
            params['daly_wt_obstructed_labour'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=348)

        # TODO: determine DALY weights for uterine rupture?

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

        df.loc[df.sex == 'F', 'la_labour_current_pregnancy'] = 'not_in_labour'
        df.loc[df.sex == 'F', 'la_current_labour_successful_induction'] = 'not_induced'
        df.loc[df.sex == 'F', 'la_still_birth_current_pregnancy'] = False
        df.loc[df.sex == 'F', 'la_parity'] = 0
        df.loc[df.sex == 'F', 'la_total_deliveries_by_cs'] = 0
        df.loc[df.sex == 'F', 'la_has_previously_delivered_preterm'] = False
        df.loc[df.sex == 'F', 'la_due_date_current_pregnancy'] = pd.NaT
        df.loc[df.sex == 'F', 'la_obstructed_labour'] = False
        df.loc[df.sex == 'F', 'la_aph'] =False
        df.loc[df.sex == 'F', 'la_uterine_rupture'] = False
        df.loc[df.sex == 'F', 'la_eclampsia'] = False
        df.loc[df.sex == 'F', 'la_pph'] = False
        df.loc[df.sex == 'F', 'la_maternal_death'] = False
        df.loc[df.sex == 'F', 'la_maternal_death_date'] = pd.NaT

# -----------------------------------ASSIGN PREGNANCY AND DUE DATE AT BASELINE (DUMMY) --------------------------------

        # !!!!!!!!!!!!!!!!1 THIS WILL BE REPLACED BY CONTRACEPTION CODE (TC) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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
        df.loc[pregnant_idx, 'la_due_date_current_pregnancy'] = dfx.due_date

        # For women who are less than term gestation at baseline, we determine if they will go into preterm/post term
        # or term labour

        # First we apply the risk of preterm birth to these women
        non_term_women = dfx.index[dfx.gestational_age_in_weeks < 36]
        eff_prob_ptl = pd.Series(params['prob_ptl'], index=non_term_women)
        random_draw = pd.Series(rng.random_sample(size=len(non_term_women)),
                                index=non_term_women)

        dfx = pd.concat((eff_prob_ptl,random_draw), axis=1)
        dfx.columns = ['eff_prob_ptl', 'random_draw']
        idx_ptb = dfx.index[dfx.eff_prob_ptl > dfx.random_draw]
        idx_no_ptb = dfx.index[dfx.eff_prob_ptl < dfx.random_draw]

        # Todo: consider if we should apply risk factors to women at baseline (anaemia and age)

        # For those who will be preterm we then determine if this will be early (24-33 weeks) or late (34-36)
        random = pd.Series(self.rng.choice(('late', 'early'), p=[0.752, 0.248], size=len(idx_ptb)), index=idx_ptb)
        conception = pd.Series(df.date_of_last_pregnancy, index=idx_ptb)
        dfx = pd.concat((conception, random), axis=1)
        dfx.columns = ['conception', 'random']

        idx_e_ptl = dfx.index[dfx.random == 'early']
        idx_l_ptl = dfx.index[dfx.random == 'late']

        # We then set there due date for somewhere between 24-33 weeks in the future from their contraception date
        random_e = pd.Series(self.rng.choice(range(24, 34), size=len(idx_e_ptl)), index=idx_e_ptl)
        idx_e_ptl_concep= pd.Series(df.date_of_last_pregnancy, index=idx_e_ptl)
        dfx = pd.concat((idx_e_ptl_concep, random_e), axis=1)
        dfx.columns = ['idx_e_ptl_concep', 'random_e']
        dfx['due_date'] = dfx['idx_e_ptl_concep'] + pd.to_timedelta(dfx['random_e'], unit='w')
        df.loc[idx_e_ptl, 'la_due_date_current_pregnancy'] = dfx.due_date

        # or for late preterm women we set the due date between 34-36 weeks
        random_l = pd.Series(self.rng.choice(range(34, 37), size=len(idx_l_ptl)), index=idx_l_ptl)
        idx_l_ptl_concep = pd.Series(df.date_of_last_pregnancy, index=idx_l_ptl)
        dfx = pd.concat((idx_l_ptl_concep, random_l), axis=1)
        dfx.columns = ['idx_l_ptl_concep', 'random_l']
        dfx['due_date'] = dfx['idx_l_ptl_concep'] + pd.to_timedelta(dfx['random_l'], unit='w')
        df.loc[idx_l_ptl, 'la_due_date_current_pregnancy'] = dfx.due_date

        # For women who wont go into preterm labour, we then apply the risk of post term labour
        eff_prob_potl = pd.Series(params['prob_potl'], index=idx_no_ptb)
        random_draw = pd.Series(rng.random_sample(size=len(idx_no_ptb)),
                                index=idx_no_ptb)

        dfx = pd.concat((eff_prob_potl, random_draw), axis=1)
        dfx.columns = ['eff_prob_potl', 'random_draw']
        idx_potl = dfx.index[dfx.eff_prob_potl > dfx.random_draw]

        # And schedule these women's due dates between 42-44 weeks from gestation
        random = pd.Series(self.rng.choice(range(42, 45), size=len(idx_potl)), index=idx_potl)
        conception = pd.Series(df.date_of_last_pregnancy, index=idx_potl)
        dfx = pd.concat((conception, random), axis=1)
        dfx.columns = ['conception', 'random']
        dfx['due_date'] = dfx['conception'] + pd.to_timedelta(dfx['random'], unit='w')
        df.loc[idx_potl, 'la_due_date_current_pregnancy'] = dfx.due_date

        # Then all women are scheduled to go into labour on this due date
        for person in pregnant_idx:
            scheduled_labour_date = df.at[person, 'la_due_date_current_pregnancy']
            labour = LabourEvent(self, individual_id=person, cause='Labour')
            self.sim.schedule_event(labour, scheduled_labour_date)

#  ----------------------------ASSIGNING PARITY AT BASELINE (DUMMY)-----------------------------------------------------

        # !!!!!!!!!!!!!!!!1 THIS WILL BE REPLACED BY CONTRACEPTION CODE (TC) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO: just check that TC isnt just doing the analysis but is also applying the parity at baseline

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
        random_draw1 = pd.Series(self.rng.choice(range(0, 2), p=[0.954, 0.046], size=len(women_para1_idx)),
                               index=women_para1_idx)

        # A second weighted random choice is applied to women with greater than 2 deliveries, due to low section rates
        # a maximum of 2 previous deliveries by caesarean is allowed
        random_draw2 = pd.Series(self.rng.choice(range(0, 3), p=[0.90, 0.07, 0.03], size=len(women_para2_idx)),
                                 index=women_para2_idx)

        dfx = pd.concat([baseline_cs1, random_draw1], axis=1)
        dfx.columns = ['baseline_cs1', 'random_draw1']
        idx_prev_cs = dfx.index[dfx.random_draw1 >= 0]
        df.loc[idx_prev_cs, 'la_total_deliveries_by_cs'] = dfx.random_draw1

        dfx = pd.concat([baseline_cs2, random_draw2], axis=1)
        dfx.columns = ['baseline_cs2','random_draw2']
        idx_prev_cs1 = dfx.index[dfx.random_draw2 >= 0]
        df.loc[idx_prev_cs1, 'la_total_deliveries_by_cs'] = dfx.random_draw2

        # ------------------------------ ASSIGN PREVIOUS PTB AT BASELINE ----------------------------------------------

        # Get and hold all women who have given birth previously, excluding those with previous caesarean section
        women_para1_nocs_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') & (df.la_parity == 1) &
                                        (df.la_total_deliveries_by_cs ==0)]

        # Get and hold all women with greater than 2 deliveries excluding those in which both deliveries were by
        # caesarean
        women_para2_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') & (df.la_parity >= 2) &
                                   (df.la_total_deliveries_by_cs < 2)]

        baseline_ptb = pd.Series(m.prob_ptl, index=women_para1_nocs_idx)
        baseline_ptb_p2 = pd.Series(m.prob_ptl, index=women_para2_idx)

        random_draw = pd.Series(rng.random_sample(size=len(women_para1_nocs_idx)),
                                index=df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') &
                                               (df.la_parity == 1) & (df.la_total_deliveries_by_cs == 0)])
        random_draw2 = pd.Series(rng.random_sample(size=len(women_para2_idx)),
                                index=df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') &
                                               (df.la_parity >= 2) & (df.la_total_deliveries_by_cs < 2)])

        # Use a random draw to determine if this woman's past deliveries have ever been preterm
        dfx = pd.concat([baseline_ptb, random_draw], axis=1)
        dfx.columns = ['baseline_ptb', 'random_draw']
        idx_prev_ptb = dfx.index[dfx.baseline_ptb > dfx.random_draw]
        df.loc[idx_prev_ptb, 'la_has_previously_delivered_preterm'] = True

        dfx = pd.concat([baseline_ptb_p2, random_draw2], axis=1)
        dfx.columns = ['baseline_ptb_p2', 'random_draw2']
        idx_prev_ptb = dfx.index[dfx.baseline_ptb_p2 > dfx.random_draw2]
        df.loc[idx_prev_ptb, 'la_has_previously_delivered_preterm'] = True

    def initialise_simulation(self, sim):

        event = LabourLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

        # Register this disease module with the health system
        self.sim.modules['HealthSystem'].register_disease_module(self)

    def on_birth(self, mother_id, child_id):

        df = self.sim.population.props

        if df.at[child_id, 'sex'] == 'F':
            df.at[child_id, 'la_due_date_current_pregnancy'] = pd.NaT
            df.at[child_id, 'la_current_labour_successful_induction'] = 'not_induced'
            df.at[child_id, 'la_labour_current_pregnancy'] = 'not_in_labour'
            df.at[child_id, 'la_still_birth_current_pregnancy'] = False
            df.at[child_id, 'la_parity'] = 0
            df.at[child_id, 'la_total_deliveries_by_cs'] = 0
            df.at[child_id, 'la_has_previously_delivered_preterm'] = False
            df.at[child_id, 'la_obstructed_labour'] = False
            df.at[child_id, 'la_aph'] = False
            df.at[child_id, 'la_uterine_rupture'] = False
            df.at[child_id, 'la_sepsis'] = False
            df.at[child_id, 'la_eclampsia'] = False
            df.at[child_id, 'la_pph'] = False
            df.at[child_id, 'la_maternal_death'] = False
            df.at[child_id, 'la_maternal_death_date'] = pd.NaT

        # If a mothers labour has resulted in a late term still birth her child is still generated by the simulation
        # but is_alive is reset to false to allow for monitoring of still birth rates

        if df.at[mother_id, 'la_still_birth_current_pregnancy']:
            death = demography.InstantaneousDeath(self.sim.modules['Demography'], child_id,
                                                  cause='Intrapartum Stillbirth')
            self.sim.schedule_event(death, self.sim.date)

            # Log the still birth
            logger.info('@@@@ A Still Birth has occurred, to mother %s', mother_id)
            logger.info('%s|still_birth|%s', self.sim.date,
                        {'age': df.at[child_id, 'age_years'],
                            'person_id': child_id,
                            'mother_id': mother_id})

            # This property is then reset in case of future pregnancies/stillbirths
            df.loc[mother_id, 'la_still_birth_current_pregnancy'] = False

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.info('This is Labour, being alerted about a health system interaction '
                     'person %d for: %s', person_id, treatment_id)

        # TODO: do i need to utilise this functionality for anything?

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        # TODO: Issues 1.) DALYS are hard coded 2.) how will monthly sum of DALYS work for labour comps

        logger.info('This is Labour reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe
        p = self.parameters

        health_values_1 = df.loc[df.is_alive, 'la_obstructed_labour'].map(
            {
                False: 0,
                True: 0.324  # self.parameters['daly_wt_obstructed_labour']
            })
        health_values_1.name = 'Obstructed Labour'

        health_values_2 = df.loc[df.is_alive, 'la_eclampsia'].map(
            {
                False: 0,
                True: 0.5  # (dummy) self.parameters['daly_wt_eclampsia']
            })
        health_values_2.name = 'Eclampsia'

        health_values_3 = df.loc[df.is_alive, 'la_sepsis'].map(
            {
                False: 0,
                True: 0.133  # self.parameters['daly_wt_maternal_sepsis']
            })
        health_values_3.name = 'Maternal Sepsis'

        health_values_4 = df.loc[df.is_alive, 'la_aph'].map(  # todo: consider severity
            {
                False: 0,
                True: 0.324  # self.parameters['daly_wt_haemorrhage_severe']
            })
        health_values_4.name = 'Antepartum Haemorrhage'

        health_values_5 = df.loc[df.is_alive, 'la_pph'].map( # todo: consider severity
            {
                False: 0,
                True: 0.324 # self.parameters['daly_wt_haemorrhage_severe']
            })
        health_values_5.name = 'Postpartum Haemorrhage'

        health_values_6 = df.loc[df.is_alive, 'la_uterine_rupture'].map(  # todo: consider severity
            {
                False: 0,
                True: 0.5  # (dummy) self.parameters['daly_wt_haemorrhage_severe']
            })
        health_values_6.name = 'Uterine Rupture'

        health_values_df = pd.concat([health_values_1.loc[df.is_alive], health_values_2.loc[df.is_alive],
                                      health_values_3.loc[df.is_alive], health_values_4.loc[df.is_alive],
                                      health_values_5.loc[df.is_alive], health_values_6.loc[df.is_alive]], axis=1)

        return health_values_df  # return the dataframe

class LabourScheduler (Event, IndividualScopeEventMixin):
    """This event determines when pregnant women, who have not experienced a miscarriage, will going to labour"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # First we determine this woman's risk of early preterm birth based on independent risk factors
        if ~df.at[individual_id, 'la_has_previously_delivered_preterm'] & (df.at[individual_id, 'age_years'] <20):
            rf1 = params['rr_early_ptb_age<20']
        else:
            rf1 = 1

        if df.at[individual_id, 'la_has_previously_delivered_preterm'] & (df.at[individual_id, 'age_years'] > 20):
            rf2 = params['rr_early_ptb_prev_ptb']
        else:
            rf2 = 1

        # todo: include anaemia

        riskfactors = rf1 * rf2
        if riskfactors == 1:
            eff_prob_early_ptb = params['prob_early_ptb']
        else:
            eff_prob_early_ptb = riskfactors * params['prob_early_ptb']

        # Then we determine her risk of late preterm birth based on independant risk factors
        if df.at[individual_id, 'la_has_previously_delivered_preterm']:
            rf1 = params['rr_late_ptb_prev_ptb']
        else:
            rf1 = 1

        # todo: include persistant malaria

        riskfactors = rf1
        if riskfactors == 1:
            eff_prob_late_ptb = params['prob_late_ptb']
        else:
            eff_prob_late_ptb = riskfactors * params['prob_late_ptb']

        # We then use a random draw to determine if the woman will go into preterm labour and how early she will deliver
        random = self.sim.rng.random_sample(size=1)
        if (random < eff_prob_late_ptb) & (random > eff_prob_early_ptb):
            random = np.random.randint(34, 36, size=1)
            random = int(random)
            df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] + \
                                                                    pd.Timedelta(random, unit='W')
            due_date = df.at[individual_id, 'la_due_date_current_pregnancy']

        elif random < eff_prob_early_ptb:
            random = np.random.randint(24, 33, size=1)
            random = int(random)
            df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] + \
                                                                    pd.Timedelta(random, unit='W')
            due_date = df.at[individual_id, 'la_due_date_current_pregnancy']

        # For women who will deliver after term we apply a risk of post term birth
        elif random > eff_prob_late_ptb:
            random = self.sim.rng.random_sample(size=1)
            if random < params['prob_potl']:
                # Risk factors?!
                random = np.random.randint(42, 44, size=1)
                random = int(random)
                df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] + \
                                                                        pd.Timedelta(random, unit='W')
                due_date = df.at[individual_id, 'la_due_date_current_pregnancy']
                # TODO: should all of these women automatically go into labour- how will we account for induction
                #todo: we would just apply a higher risk of still birht to these women in these last few weeks if
                # theyre not induced in time
            else:
                random = np.random.randint(37, 41, size=1)
                random = int(random)
                df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] + \
                                                                        pd.Timedelta(random, unit='W')
                due_date = df.at[individual_id, 'la_due_date_current_pregnancy']

        # Labour is then scheduled on the newly generated due date
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
                              'labour_state': None,  # Term Labour (TL), Early Preterm (EPTL), Late Preterm (LPTL) or
                                                    # Post Term (POTL)
                              'delivery_setting': None,  # Facility Delivery (FD) or Home Birth (HB)
                              'induced_labour': False,
                              'referred_for': None,  # Induction (I) or Caesarean (CS)
                              'cord_prolapse': False,
                              'PROM': False,
                              'PPROM': False,
                              'risk_ol': params['prob_pl_ol'],
                              'labour_is_currently_obstructed': False,  # True (T) or False (F)
                              'labour_has_previously_been_obstructed': False,
                              'risk_ip_sepsis': params['prob_ip_sepsis'],
                              'risk_pp_sepsis': params['prob_pp_sepsis'],
                              'sepsis_ip': False,  # True (T) or False (F)
                              'sepsis_pp': False,  # True (T) or False (F)
                              'source_sepsis': None,  # Obstetric (O) or Non-Obstetric (NO)
                              'risk_aph': params['prob_aph'],
                              'APH': False,  # True (T) or False (F)
                              'source_aph': None,  # Placenta Praevia (PP) or Placental Abruption (PA) (Other?)
                              'units_transfused': 0,
                              'risk_ip_eclampsia': params['prob_ip_eclampsia'],
                              'risk_pp_eclampsia': params['prob_pp_eclampsia'],
                              'eclampsia_ip': False,  # True (T) or False (F)
                              'eclampsia_pp': False,  # True (T) or False (F)
                              'risk_ur': params['prob_uterine_rupture'],
                              'UR': False,   # True (T) or False (F)
                              'grade_of_UR': 'X', # Partial (P) or Complete (C)
                              'risk_pph': params['prob_pph'],
                              'PPH': False,   # True (T) or False (F)
                              'source_pph': None,  # Uterine Atony (UA) or Retained Products/Placenta (RPP)
                              'severity_pph': None,
                              'risk_newborn_sepsis': params['prob_neonatal_sepsis'],
                              'risk_newborn_ba': params['prob_neonatal_birth_asphyxia'],
                              #  Should this just be risk of asyphixa
                              'mode_of_delivery': None, # Vaginal Delivery (VD),Vaginal Delivery Induced (VDI),
                              # Assisted Vaginal Delivery Forceps (AVDF) Assisted Vaginal Delivery Ventouse (AVDV)
                              # Caesarean Section (CS)
                              'death_in_labour': False,  # True (T) or False (F)
                              'stillbirth_in_labour': False,  # True (T) or False (F)
                              'death_postpartum': False}  # True (T) or False (F)


# ===================================== LABOUR STATE  ==================================================================

        # Based on gestational age the woman in labour is allocated to either term, earl/late preterm or post term
        # labour

        if df.at[individual_id, 'la_due_date_current_pregnancy'] == pd.NaT:
            pass
            logger.info('This is LabourEvent, person %d has reached their previously allocated due date but is not '
                        'entering labour at this time', individual_id)

        elif df.at[individual_id, 'is_pregnant'] & df.at[individual_id, 'is_alive']:

            if (df.at[individual_id, 'la_current_labour_successful_induction'] == 'failed_induction') or \
               (df.at[individual_id, 'la_current_labour_successful_induction'] == 'successful_induction'):
                    mni[individual_id]['delivery_setting'] = 'FD'
            if df.at[individual_id, 'la_current_labour_successful_induction'] == 'successful_induction':
                    mni[individual_id]['induced_labour'] = True

            gestation_date = df.at[individual_id, 'la_due_date_current_pregnancy'] - df.at[individual_id,
                                                                                           'date_of_last_pregnancy']
            gestation_weeks = gestation_date / np.timedelta64(1, 'W')
            gestation_weeks = int(gestation_weeks)
            mni[individual_id]['gestation_at_labour'] = gestation_weeks

            if df.at[individual_id, 'is_pregnant'] & df.at[individual_id, 'is_alive']:
                if 37 <= mni[individual_id]['gestation_at_labour'] < 42:
                    mni[individual_id]['labour_state'] = 'TL'
                # df.at[individual_id, 'la_labour_current_pregnancy'] = "term_labour"

                    logger.info('This is LabourEvent, person %d has now gone into term labour on date %s',
                                individual_id, self.sim.date)

                elif 24 <= mni[individual_id]['gestation_at_labour'] < 34:
                    mni[individual_id]['labour_state'] = 'EPTL'
                # df.at[individual_id, 'la_labour_current_pregnancy'] = "early_preterm_labour"
                    df.at[individual_id, 'la_has_previously_delivered_preterm'] = True

                    logger.info('This is LabourEvent, person %d has now gone into early preterm labour on date %s',
                                individual_id, self.sim.date)

                elif 37 > mni[individual_id]['gestation_at_labour'] >= 34:
                    mni[individual_id]['labour_state'] = 'LPTL'
                    df.at[individual_id, 'la_has_previously_delivered_preterm'] = True

                    logger.info('This is LabourEvent, person %d has now gone into late preterm labour on date %s',
                                individual_id, self.sim.date)

                elif mni[individual_id]['gestation_at_labour'] > 41:
                    mni[individual_id]['labour_state'] = 'POTL'
                # df.at[individual_id, 'la_labour_current_pregnancy'] = "post_term_labour"

                    logger.info('This is LabourEvent, person %d is now overdue labour and is post-term  on date %s',
                                individual_id, self.sim.date)

# ===================== PLACE HOLDER CARE SEEKING AND SCHEDULING (DUMMY) =====================================

            # prob = self.sim.modules['HealthSystem'].get_prob_seek_care(individual_id, symptom_code=4)
                prob = 0 # 0.73  # DUMMY- will just generate 2010 home birth rate #TODO: incorporate care seeking equation
                random = self.sim.rng.random_sample(size=1)
                if (df.at[individual_id, 'la_current_labour_successful_induction'] == 'not_induced') & (random < prob):
                    mni[individual_id]['delivery_setting'] = 'FD'
                    logger.info(
                        'This is LabourEvent, scheduling HSI_Labour_PresentsForCareInLabour on date %s for person %d '
                        'as they have chosen to seek care for delivery', self.sim.date, individual_id)

                    event = HSI_Labour_PresentsForSkilledAttendanceInLabour(self.module, person_id=individual_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset (days=14)
                                                                        )  # DUMMY tclose --> change!

                    # TODO: ISSUE- if woman wants to seek care but cant, her complications will not be allocated
                    #  because she hasnt passed through the HSI. We need some logic that says- if seeking care, but not
                    #  availble, deliver at home

                elif (df.at[individual_id, 'la_current_labour_successful_induction'] == 'not_induced') & (random > prob):
                    mni[individual_id]['delivery_setting'] = 'HB'
                    logger.info(
                        'This is LabourEvent, doing nothing as person %d will not seek care in labour and will deliver '
                        'at home',individual_id)

                # Here we schedule delivery care for women who have already sought care for induction, whether or not
                # that induction was successful
                if (df.at[individual_id, 'la_current_labour_successful_induction'] == 'failed_induction') or \
                    (df.at[individual_id, 'la_current_labour_successful_induction'] == 'successful_induction'):
                    event = HSI_Labour_PresentsForSkilledAttendanceInLabour(self.module, person_id=individual_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=14)
                                                                        )  # DUMMY tclose --> change!

# ============================ INDIVIDUAL RISK OF COMPLICATIONS DURING LABOUR =========================================

# ====================================== STATUS OF MEMBRANES ==========================================================

        # Todo: consider risk factors
        # Todo: discuss with Tim C importance of modelling PPROM (would change structure)

        # Here we apply a risk that this woman's labour was preceded by premature rupture of membranes, in preterm
        # women this has likley predisposed their labour
                if (mni[individual_id]['labour_state'] == 'EPTL') or (mni[individual_id]['labour_state'] == 'LPTL'):
                    random = self.sim.rng.random_sample(size=1)
                    if random < params['prob_prom']:
                        mni[individual_id]['PROM'] = True

# ===================================  OBSTRUCTED LABOUR =============================================================

        # TODO: in the case of induction should we just skip obstructed labour? surely its possible

        # We identify if this woman's labour will become obstructed based on risk factors using the same
        # calculation as previous individual style events

                if (df.at[individual_id, 'la_parity'] == 0) & (df.at[individual_id, 'age_years'] >= 21):
                    rf1 = params['rr_PL_OL_nuliparity']
                    # rf1= 1
                else:
                    # rf1= 0
                    rf1 = 1

                if (df.at[individual_id, 'la_parity'] == 1) & (df.at[individual_id, 'age_years'] >= 21):
                    rf2 = params['rr_PL_OL_para1']
                    # rf2=1
                else:
                    rf2 = 1
                    # rf2 = 0
                if (df.at[individual_id, 'la_parity'] > 1) & (df.at[individual_id, 'la_parity'] < 3) &\
                    (df.at[individual_id, 'age_years'] < 20):
                    rf3 = params['rr_PL_OL_age_less20']
                    # rf3 = 1
                else:
                    rf3 = 1
                    # rf3 = 0

        # TODO: this formulation of the risk factors (which is used very often) might be more readble as: p_outcome =
        #  p_baseeline * (has_rf1*rr1) * (has_rf2*rr2) - discussed with TC may not work at individual level

                riskfactors = rf1*rf2*rf3
                eff_prob_ol = riskfactors * params['prob_pl_ol']

                if mni[individual_id]['delivery_setting'] == 'FD':
                    mni[individual_id]['risk_ol'] = eff_prob_ol
                else:
                    random = self.sim.rng.random_sample(size=1)
                    if random < eff_prob_ol:
                        mni[individual_id]['labour_is_currently_obstructed'] = True
                        mni[individual_id]['labour_has_previously_been_obstructed'] = True
                        df.at[individual_id, 'la_obstructed_labour'] = True
                        logger.info('person %d has developed obstructed labour in the community on date %s',
                                    individual_id, self.sim.date)
                        logger.info('%s|obstructed_labour|%s', self.sim.date,
                                    {'age': df.at[individual_id, 'age_years'],
                                     'person_id': individual_id})

# We then work through the next complications and assess if this woman will experience additional complications

# ==================================== ECLAMPSIA ======================================================================

                if (df.at[individual_id, 'la_parity'] == 0) & (df.at[individual_id, 'age_years'] <= 29):
                    rf1 = params['rr_ip_eclampsia_nullip']
                else:
                    rf1 = 1

                if (df.at[individual_id, 'la_parity'] >= 1) & (df.at[individual_id, 'age_years'] >= 35):
                    rf2 = params['rr_ip_eclampsia_35']
                else:
                    rf2 = 1

                if (df.at[individual_id, 'la_parity'] >= 1) & (df.at[individual_id, 'age_years'] >= 30) & \
                (df.at[individual_id, 'age_years'] <= 34):
                    rf3 = params['rr_ip_eclampsia_30_34']
                else:
                    rf3 = 1

                # To include: High BMI, Education, Chronic HTN, Gestational Diabetes, Anaemia, ANC visits,
                # previous PE/HTN (will need to be stored)

                riskfactors = rf1 * rf2 * rf3
                eff_prob_eclampsia = riskfactors * params['prob_ip_eclampsia']

                if mni[individual_id]['delivery_setting'] == 'FD':
                    mni[individual_id]['risk__ip_eclampsia'] = eff_prob_eclampsia

                else:
                    random = self.sim.rng.random_sample(size=1)
                    if random < eff_prob_eclampsia:
                        df.at[individual_id, 'la_eclampsia'] = True
                        mni[individual_id]['eclampsia_ip'] = True
                        logger.info('person %d is experiencing intrapartum eclampsia in the community on date %s',
                                    individual_id, self.sim.date)
                        logger.info('%s|eclampsia|%s', self.sim.date,
                                    {'age': df.at[individual_id, 'age_years'],
                                     'person_id': individual_id})


# ============================================ ANTEPARTUM HAEMORRHAGE =================================================

                if df.at[individual_id, 'li_ed_lev'] == 1:
                    rf1 = params['rr_ip_aph_noedu']
                else:
                    rf1 = 1
                if mni[individual_id]['labour_is_currently_obstructed'] & (df.at[individual_id, 'li_ed_lev'] >1):
                    rf2 = params['rr_aph_pl_ol']
                else:
                    rf2 = 1

                riskfactors = rf1 * rf2
                eff_prob_aph = riskfactors * params['prob_aph']

                # Todo: a/w review of evidence to determine strength of association between risk factors and
                #  praevia/abruption

                if mni[individual_id]['delivery_setting'] == 'FD':
                    mni[individual_id]['risk_aph'] = eff_prob_aph
                else:
                    random = self.sim.rng.random_sample(size=1)
                    if random < eff_prob_aph:
                        df.at[individual_id, 'la_aph'] = True
                        mni[individual_id]['APH'] = True
                        logger.info('person %d is experiencing an antepartum haemorrhage in the community on date %s',
                                    individual_id, self.sim.date)
                        logger.info('%s|antepartum_haemorrhage|%s', self.sim.date,
                                    {'age': df.at[individual_id, 'age_years'],
                                     'person_id': individual_id})


# ==========================================  SEPSIS =================================================================

        # Todo: We will include both obstetric and non obstetric sepsis, but will only apply here incidence on
        #  obstetric sepsis and use properties from the other modules to determine non obstetric infection

                if mni[individual_id]['labour_is_currently_obstructed']:
                    rf1 = params['rr_ip_sepsis_pl_ol']
                else:
                    rf1 = 1

                riskfactors = rf1
                eff_prob_sepsis = riskfactors * params['prob_ip_sepsis']

                if mni[individual_id]['delivery_setting'] == 'FD':
                    mni[individual_id]['risk_ip_sepsis'] = eff_prob_sepsis

                else:
                    random = self.sim.rng.random_sample(size=1)
                    if random < eff_prob_sepsis:
                        df.at[individual_id, 'la_sepsis'] = True
                        mni[individual_id]['sepsis_ip'] = True
                        logger.info('person %d is experiencing intrapartum maternal sepsis in the community on date %s',
                                    individual_id, self.sim.date)
                        logger.info('%s|maternal_sepsis|%s', self.sim.date,
                                    {'age': df.at[individual_id, 'age_years'],
                                     'person_id': individual_id})

                        # TODO modify newborn risk of sepsis for septic women

# ====================================== UTERINE RUPTURE  =============================================================

                if (df.at[individual_id, 'la_parity'] >= 5) & (df.at[individual_id, 'la_total_deliveries_by_cs'] == 0)\
                    & ~df.at[individual_id, 'la_obstructed_labour']:
                     rf1 = params['rr_ur_grand_multip']
                else:
                    rf1 = 1

                if (df.at[individual_id, 'la_parity'] < 5) & (df.at[individual_id, 'la_total_deliveries_by_cs'] >= 1) & \
                    ~df.at[individual_id, 'la_obstructed_labour']:
                    rf2 = params['rr_ur_prev_cs']
                else:
                    rf2 = 1

                if (df.at[individual_id, 'la_parity'] < 5) & (df.at[individual_id, 'la_total_deliveries_by_cs'] == 0) & \
                    (mni[individual_id]['labour_is_currently_obstructed'] == True):  # only HBs can already be in OL
                        rf3 = params['rr_ur_ref_ol']
                else:
                    rf3 = 1

                riskfactors = rf1 * rf2 * rf3
                eff_prob_ur = riskfactors * params['prob_uterine_rupture']

                if mni[individual_id]['delivery_setting'] == 'FD':
                    df.at[individual_id, 'risk_ur'] = eff_prob_ur

                else:
                    random = self.sim.rng.random_sample(size=1)
                    if random < eff_prob_ur:
                        df.at[individual_id, 'la_uterine_rupture'] = True
                        mni[individual_id]['UR'] = True
                        logger.info('person %d is experiencing uterine rupture in the community on date %s',
                                    individual_id, self.sim.date)
                        logger.info('%s|uterine_rupture|%s', self.sim.date,
                                    {'age': df.at[individual_id, 'age_years'],
                                     'person_id': individual_id})

        # Here we schedule the birth event for 2 days after labour- we do this prior to the death event as women who
        # die but still deliver a live child will pass through birth event

                due_date = df.at[individual_id, 'la_due_date_current_pregnancy']
                logger.info('This is LabourEvent scheduling a birth on date %s'
                             ' to mother %d',due_date, individual_id)
                self.sim.schedule_event(BirthEvent(self.module, individual_id), due_date + DateOffset(days=3))

        # We schedule all women to move through the death event where those who have developed a complication that
        # hasn't been treated or treatment has failed will have a case fatality rate applied

                self.sim.schedule_event(LabourDeathEvent(self.module, individual_id, cause='labour'),
                                self.sim.date + DateOffset(days=2))
                logger.info('This is LabourEvent scheduling a potential death on date %s for mother %d',
                             self.sim.date, individual_id)

                # Here we set the due date of women who have been induced to pd.NaT so they dont go into labour twice
                if df.at[individual_id, 'la_current_labour_successful_induction']:
                    df.at[individual_id, 'la_due_date_current_pregnancy'] = pd.NaT


class BirthEvent(Event, IndividualScopeEventMixin):
    """A one-off event in which a pregnant mother gives birth.
    """

    def __init__(self, module, mother_id):
        """Create a new birth event."""
        super().__init__(module, person_id=mother_id)

    def apply(self, mother_id):

        # This event tells the simulation that the woman's pregnancy is over and generates the new child in the
        # data frame
        logger.info('@@@@ A Birth is now occuring, to mother %d', mother_id)
        df = self.sim.population.props
        mni = self.module.mother_and_newborn_info

        # If the mother is alive and still pregnant we generate a live child and the woman is scheduled to move to the
        # postpartum event to determine if she experiences any additional complications
        if df.at[mother_id, 'is_alive'] and df.at[mother_id, 'is_pregnant']:
            self.sim.do_birth(mother_id)
            df.at[mother_id, 'la_parity'] += 1
            df.at[mother_id,'ac_gestational_age'] = 0
            logger.info('This is BirthEvent scheduling mother %d to undergo the PostPartumEvent following birth',
                         mother_id)
            self.sim.schedule_event(PostpartumLabourEvent(self.module, mother_id, cause='post partum'),
                                    self.sim.date)

        # If the mother has died during childbirth the child is still generated with is_alive=false to monitor
        # stillbirth rates. She will not pass through the postpartum complication events
        if df.at[mother_id, 'is_alive'] == False & df.at[mother_id, 'is_pregnant'] == True & \
            (mni[mother_id]['death_in_labour'] == True):
            self.sim.do_birth(mother_id)


class PostpartumLabourEvent(Event, IndividualScopeEventMixin):

    """applies probability of postpartum complications to women who have just delivered """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self
        mni = self.module.mother_and_newborn_info

# ============================== POSTPARTUM COMPLICATIONS FOLLOWING LABOUR ============================================
        # Here we follow the same format as within the main labour event and we determine if women experience any
        # complications following delivery

# ============================================= RISK OF PPH ===========================================================

        if df.at[individual_id, 'is_alive']:
            if mni[individual_id]['labour_has_previously_been_obstructed']:
                rf1 = params['rr_pph_pl_ol']
            else:
                rf1 = 1

            riskfactors = rf1
            eff_prob_pph = riskfactors * params['prob_pph']

            if mni[individual_id]['delivery_setting'] == 'FD':
                mni[individual_id]['risk_pph'] = eff_prob_pph

            else:
                random = self.sim.rng.random_sample(size=1)
                if random < eff_prob_pph:
                    df.at[individual_id, 'la_pph'] = True
                    mni[individual_id]['PPH'] = True
                    logger.info('person %d is experiencing a postpartum haemorrhage in the community on date %s',
                                individual_id, self.sim.date)
                    logger.info('%s|postpartum_haemorrhage|%s', self.sim.date,
                                {'age': df.at[individual_id, 'age_years'],
                                 'person_id': individual_id})

            #  todo: include severity

# ============================================= RISK OF SEPSIS =========================================================

            if mni[individual_id]['labour_has_previously_been_obstructed']:
                rf1 = params['rr_ip_sepsis_pl_ol'] # should we have differnt an/pn rates?
            else:
                rf1 = 1

            riskfactors = rf1
            eff_prob_pp_sepsis = riskfactors * params['prob_pp_sepsis']

            if mni[individual_id]['delivery_setting'] == 'FD':
                mni[individual_id]['risk_pp_sepsis'] = eff_prob_pp_sepsis

            else:
                random = self.sim.rng.random_sample(size=1)
                if random < eff_prob_pp_sepsis:
                    df.at[individual_id, 'la_sepsis'] = True
                    mni[individual_id]['sepsis_pp'] = True
                    logger.info('person %d is experiencing postpartum maternal sepsis in the community on date %s',
                            individual_id, self.sim.date)

                    logger.info('%s|maternal_sepsis|%s', self.sim.date,
                                {'age': df.at[individual_id, 'age_years'],
                                 'person_id': individual_id})

# ============================================= RISK OF ECLAMPSIA ====================================================

            if (df.at[individual_id, 'la_parity'] == 0) & (df.at[individual_id, 'age_years'] <= 29):
                rf1 = params['rr_ip_eclampsia_nullip']
            else:
                rf1 = 1

            if (df.at[individual_id, 'la_parity'] >= 1) & (df.at[individual_id, 'age_years'] >= 35):
                rf2 = params['rr_ip_eclampsia_35']
            else:
                rf2 = 1

            if (df.at[individual_id, 'la_parity'] >= 1) & (df.at[individual_id, 'age_years'] >= 30) & \
                (df.at[individual_id, 'age_years'] <= 34):
                rf3 = params['rr_ip_eclampsia_30_34']
            else:
                rf3 = 1

            riskfactors = rf1 * rf2 * rf3
            eff_prob_eclampsia = riskfactors * params['prob_pp_eclampsia']

            if mni[individual_id]['delivery_setting'] == 'FD':
                mni[individual_id]['risk_pp_eclampsia'] = eff_prob_eclampsia
            else:
                random = self.sim.rng.random_sample(size=1)
                if random < eff_prob_eclampsia:
                    df.at[individual_id, 'la_eclampsia'] = True
                    mni[individual_id]['eclampsia_pp'] = True
                    mni[individual_id]['timing_eclampsia'] = 'PP'
                    logger.info('person %d is experiencing postpartum eclampsia in the community on date %s',
                                individual_id, self.sim.date)
                    logger.info('%s|eclampsia|%s', self.sim.date,
                                {'age': df.at[individual_id, 'age_years'],
                                 'person_id': individual_id})

            # if in facility....
            if mni[individual_id]['delivery_setting'] == 'FD':
                logger.info('This is PostPartumEvent scheduling HSI_Labour_ReceivesCareForPostpartumPeriod for person '
                            '%d on date %s', individual_id,self.sim.date)
                event = HSI_Labour_ReceivesCareForPostpartumPeriod(self.module, person_id=individual_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date
                                                                )
                # todo: same issue for women who seek care but cant be seek- comps wont be allocated!

            self.sim.schedule_event(PostPartumDeathEvent(self.module, individual_id, cause='labour'), self.sim.date)
            logger.info('This is PostPartumEvent scheduling a potential death for person %d on date %s', individual_id,
                         self.sim.date + DateOffset(days=3))  # Date offsetted to allow for interventions


class LabourDeathEvent (Event, IndividualScopeEventMixin):

    """handles death in labour"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self
        mni = self.module.mother_and_newborn_info

        # Currently we apply an untreated case fatality ratio (dummy values presently) who have experienced a
        # complication
        # Similarly we apply a risk of still birth associated with each complication

        # if df.at[individual_id, 'la_eclampsia']:

        # TODO: for each intervention where appropriate apply reduction in CFR based on interventions?

        if mni[individual_id]['labour_is_currently_obstructed']:
            random = self.sim.rng.random_sample(size=1)
            if random < params['cfr_obstructed_labour']:
                mni[individual_id]['death_in_labour'] = True
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date

                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_obstructed_labour_md']:
                    df.at[individual_id, 'la_still_birth_current_pregnancy'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True
            else:
                random = self.sim.rng.random_sample(size=1)
                if random < params['prob_still_obstructed_labour']:
                    df.at[individual_id, 'la_still_birth_current_pregnancy'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True

        if mni[individual_id]['eclampsia_ip']:
            random = self.sim.rng.random_sample(size=1)
            if random < params['cfr_eclampsia']:
                mni[individual_id]['death_in_labour'] = True
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date

                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_eclampsia_md']:
                    df.at[individual_id, 'la_still_birth_current_pregnancy'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True
            else:
                random = self.sim.rng.random_sample(size=1)
                if random < params['prob_still_birth_eclampsia']:
                    df.at[individual_id, 'la_still_birth_current_pregnancy'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True

        # if df.at[individual_id, 'la_aph']:
        if mni[individual_id]['APH']:
            random = self.sim.rng.random_sample(size=1)
            if random < params['cfr_aph']:
                mni[individual_id]['death_in_labour'] = True
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date

                random = self.sim.rng.random_sample(size=1)
                if random < params['prob_still_birth_aph_md']:
                    df.at[individual_id, 'la_still_birth_current_pregnancy'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True
            else:
                random = self.sim.rng.random_sample(size=1)
                if random < params['prob_still_birth_aph']:
                    df.at[individual_id, 'la_still_birth_current_pregnancy'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True

        # if df.at[individual_id, 'la_sepsis']:
        if mni[individual_id]['sepsis_ip']:
            random = self.sim.rng.random_sample(size=1)
            if random < params['cfr_sepsis']:
                mni[individual_id]['death_in_labour'] = True
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date

                random = self.sim.rng.random_sample(size=1)
                if random < params['prob_still_birth_sepsis_md']:
                    df.at[individual_id, 'la_still_birth_current_pregnancy'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True
            else:
                random = self.sim.rng.random_sample(size=1)
                if random < params['prob_still_birth_sepsis']:
                    df.at[individual_id, 'la_still_birth_current_pregnancy'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True

        # if df.at[individual_id, 'la_uterine_rupture']:
        if mni[individual_id]['UR']:
            random = self.sim.rng.random_sample(size=1)
            if random < params['cfr_uterine_rupture']:
                mni[individual_id]['death_in_labour'] = True
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date

                random = self.sim.rng.random_sample(size=1)
                if random < params['prob_still_birth_ur_md']:
                    df.at[individual_id, 'la_still_birth_current_pregnancy'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True
            else:
                random = self.sim.rng.random_sample(size=1)
                if random < params['prob_still_birth_ur']:
                    df.at[individual_id, 'la_still_birth_current_pregnancy'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True

            # Schedule death for women who die in labour

        if mni[individual_id]['death_in_labour']:
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                  cause='labour'), self.sim.date)

            # Log the maternal death
            logger.info('This is LabourDeathEvent scheduling a death for person %d on date %s who died due to '
                        'intrapartum complications', individual_id, self.sim.date)

            complication_profile = mni[individual_id]
            logger.info('%s|labour_complications|%s', self.sim.date,
                        {'person_id': individual_id,
                         'labour_profile': complication_profile})

            logger.info('%s|maternal_death|%s', self.sim.date,
                        {'age': df.at[individual_id, 'age_years'],
                            'person_id': individual_id })


class PostPartumDeathEvent (Event, IndividualScopeEventMixin):

    """handles death following labour"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self
        mni = self.module.mother_and_newborn_info

        # We apply the same structure as with the LabourDeathEvent to women who experience postpartum complications
        # if df.at[individual_id, 'la_eclampsia']:
        if mni[individual_id]['eclampsia_pp']:
            random = self.sim.rng.random_sample(size=1)
            if random < params['cfr_pp_eclampsia']:
                mni[individual_id]['death_postpartum'] = True
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date


        # if df.at[individual_id, 'la_pph']:
        if mni[individual_id]['PPH']:
            random = self.sim.rng.random_sample(size=1)
            if random < params['cfr_pph']:
                mni[individual_id]['death_postpartum'] = True
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date

        # if df.at[individual_id, 'la_sepsis']:
        if mni[individual_id]['sepsis_pp']:
            random = self.sim.rng.random_sample(size=1)
            if random < params['cfr_pp_sepsis']:
                mni[individual_id]['death_postpartum'] = True
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date

        if mni[individual_id]['death_postpartum']:

            self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                  cause='postpartum labour'), self.sim.date)
            logger.info('This is PostPartumDeathEvent scheduling a death for person %d on date %s who died due to '
                        'postpartum complications', individual_id,
                         self.sim.date)

            logger.info('%s|maternal_death|%s', self.sim.date,
                        {'age': df.at[individual_id, 'age_years'],
                            'person_id': individual_id})

            complication_profile = mni[individual_id]
            logger.info('%s|labour_complications|%s', self.sim.date,
                        {'person_id': individual_id,
                         'labour_profile': complication_profile})
        else:
            complication_profile = mni[individual_id]
            logger.info('%s|labour_complications|%s', self.sim.date,
                        {'person_id': individual_id,
                         'labour_profile': complication_profile})

# ======================================================================================================================
# ================================ HEALTH SYSTEM INTERACTION EVENTS ====================================================
# ======================================================================================================================


class HSI_Labour_PresentsForInductionOfLabour(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This interaction manages induction of labour for women with indications identified antenatally such as severe
    pre-eclampsia and being post term
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1
    #   the_appt_footprint['NormalDelivery'] = 1  # THIS IS NOT WORKING

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        dummy_pkg_code = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] ==
                                              'HIV Testing Services',
                                              'Intervention_Pkg_Code'])[0]

    #   pkg_code_sba_uncomp = pd.unique(consumables.loc[consumables[
    #                                              'Intervention_Pkg'] ==
    #                                          'Induction of labour',
    #                                          'Intervention_Pkg_Code'])[0] # THIS IS NOT WORKING

        the_cons_footprint = {
            'Intervention_Package_Code': [dummy_pkg_code], #DUMMY
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_PresentsForInductionOfLabour'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        # TODO: await antenatal care module to schedule induction
        # TODO: women with praevia, obstructed labour, should be induced

        print('Induction working')
        logger.info('This is HSI_Labour_PresentsForInductionOfLabour, person %d is attending a health facility to have'
                    ' their labour induced on date %s', person_id, self.sim.date)

        # Indications: Post term, eclampsia, severe preeclampsia, mild preeclampsia at term, PROM > 24 hrs at term
        # or PPROM > 34 weeks EGA, and IUFD.

        # We use a random draw to determine if this womans labour will be successfully induced
        random = self.sim.rng.random_sample(size=1)
        if random < params['prob_successful_induction']:
            logger.info('Person %d has had her labour successfully induced', person_id)
            df.at[person_id, 'la_current_labour_successful_induction'] = 'successful_induction'

        # For women whose induction fails they will undergo caesarean section
        else:
            logger.info('Persons %d labour has been unsuccessful induced', person_id)
            df.at[person_id, 'la_current_labour_successful_induction'] = 'failed_induction'

        self.sim.schedule_event(LabourEvent(self.module, person_id, cause='labour'), self.sim.date)


class HSI_Labour_PresentsForSkilledAttendanceInLabour(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This interaction manages a woman's initial presentation to a health facility when in labour
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1
    #   the_appt_footprint['NormalDelivery'] = 1  # THIS IS NOT WORKING

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        dummy_pkg_code = pd.unique(consumables.loc[consumables[
                                                  'Intervention_Pkg'] ==
                                              'HIV Testing Services',
                                              'Intervention_Pkg_Code'])[0]

    #    pkg_code_sba_uncomp = pd.unique(consumables.loc[consumables[
    #                                              'Intervention_Pkg'] ==
    #                                          'Vaginal delivery - skilled attendance',
    #                                          'Intervention_Pkg_Code'])[0] # THIS IS NOT WORKING

        the_cons_footprint = {
            'Intervention_Package_Code': [dummy_pkg_code],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_PresentsForSkilledAttendanceInLabour'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: Providing initial skilled attendance '
                    'at birth for person %d on date %s', person_id, self.sim.date)

        # TODO: Discuss with Tim H the best way to capture which HCP will be attending this delivery and how that may
        #  affect outcomes?

    # ============================ CLEAN DELIVERY PRACTICES AT BIRTH ==================================================

        # First we apply the estimated impact of clean birth practices on maternal and newborn risk of sepsis

        adjusted_maternal_sepsis_risk = mni[person_id]['risk_ip_sepsis'] * \
            params['rr_maternal_sepsis_clean_delivery']
        mni[person_id]['risk_ip_sepsis'] = adjusted_maternal_sepsis_risk

        adjusted_newborn_sepsis_risk = mni[person_id]['risk_newborn_sepsis'] * \
            params['rr_newborn_sepsis_clean_delivery']
        mni[person_id]['risk_newborn_sepsis'] = adjusted_newborn_sepsis_risk

# =============================== SKILLED BIRTH ATTENDANCE EFFECT ======================================================
        # Then we apply the estimated effect of fetal surveillance methods on maternal/newborn outcomes

        # todo: Discuss again with Tim C if it is worth trying to quantify this, as many evaluative studies look at
        #  provision of BEmOC etc. so will include the impact of the interventions

# ================================== INTERVENTIONS FOR PRE-EXSISTING CONDITIONS =====================================
        # Here we apply the effect of any 'upstream' interventions that may reduce the risk of intrapartum
        # complications/poor newborn outcomes

        # ------------------------------------- PROM ------------------------------------------------------------------
        # First we apply a risk reduction in likelihood of sepsis for women with PROM as they have received prophylactic
        # antibiotics
        if mni[person_id]['PROM']:
            treatment_effect = params['rr_sepsis_post_abx_prom']
            new_sepsis_risk = mni[person_id]['risk_ip_sepsis'] * treatment_effect
            mni[person_id]['risk_ip_sepsis'] = new_sepsis_risk

        # ------------------------------------- PREMATURITY -----------------------------------------------------------
        # Here we apply the effect of interventions to improve outcomes of neonates born preterm

        # Antibiotics for group b strep prophylaxsis: (are we going to apply this to all cause sepsis?)
        if mni[person_id]['labour_state'] == 'EPTL' or mni[person_id]['labour_state'] == 'LPTL':
            treatment_effect = params['rr_newborn_sepsis_proph_abx']
            new_newborn_risk = mni[person_id]['risk_newborn_sepsis'] * treatment_effect
            mni[person_id]['risk_newborn_sepsis'] = new_newborn_risk

        # STEROIDS - effect applied directly to newborns (need consumables here
        # TOXOLYTICS !? - WHO advises against

# ===================================  COMPLICATIONS OF LABOUR ========================================================

    # Here, using the adjusted risks calculated following 'in-labour' interventions to determine which complications a
    # woman may experience and store those in the dataframe

        random = self.sim.rng.random_sample(size=1)
        if random < mni[person_id]['risk_ol']:
            df.at[person_id, 'la_obstructed_labour'] = True
            mni[person_id]['labour_is_currently_obstructed'] = True
            mni[person_id]['labour_has_previously_been_obstructed'] = True
            logger.info('person %d is experiencing obstructed labour in a health facility',
                        person_id)

            # TODO: issue- if we apply risk of both UR and OL here then we will negate the effect of OL treatment on
            #  reduction of incidence of UR

        random = self.sim.rng.random_sample(size=1)
        if random < mni[person_id]['risk_ip_eclampsia']:
            df.at[person_id, 'la_eclampsia'] = True
            mni[person_id]['eclampsia_ip'] = True
            logger.info('person %d is experiencing eclampsia in a health facility',
                        person_id)
            logger.info('%s|eclampsia|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

        if random < mni[person_id]['risk_aph']:
            df.at[person_id, 'la_aph'] = True
            mni[person_id]['APH'] = True
            logger.info('person %d is experiencing an antepartum haemorrhage in a health facility',
                        person_id)
            logger.info('%s|antepartum_haemorrhage|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

        random = self.sim.rng.random_sample(size=1)
        if random < mni[person_id]['risk_ip_sepsis']:
            df.at[person_id, 'la_sepsis'] = True
            mni[person_id]['sepsis'] = True
            mni[person_id]['source_sepsis'] = 'IP'
            logger.info('person %d has developed maternal sepsis in a health facility',
                        person_id)
            logger.info('%s|maternal_sepsis|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})
            # TODO modify newborn risk of sepsis for septic women

        random = self.sim.rng.random_sample(size=1)
        if random < mni[person_id]['risk_ur']:
            mni[person_id]['UR'] = True
            df.at[person_id, 'la_uterine_rupture'] = True
            logger.info('person %d is experiencing a uterine rupture in a health facility',
                        person_id)
            logger.info('%s|uterine_rupture|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

        # DUMMY ... (risk factors)
        random = self.sim.rng.random_sample(size=1)
        if random < params['prob_cord_prolapse']:
            mni[person_id]['cord_prolapse'] = True

# ==================================== SCHEDULE HEALTH SYSTEM INTERACTIONS ===========================================
    # Here, if a woman has developed a complication, she is scheduled to receive any care she may need

        if mni[person_id]['labour_is_currently_obstructed']:
            logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: scheduling immediate additional '
                        'treatment for obstructed labour for person %d', person_id)

            event = HSI_Labour_ReceivesCareForObstructedLabour(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=14))

        if mni[person_id]['sepsis_ip']:
            logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: scheduling immediate additional '
                        'treatment for maternal sepsis during labour for person %d', person_id)

            event = HSI_Labour_ReceivesCareForMaternalSepsis(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=14))

        if mni[person_id]['APH']:
            logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: scheduling immediate additional '
                        'treatment for antepartum haemorrhage during labour for person %d', person_id)

            event = HSI_Labour_ReceivesCareForMaternalHaemorrhage(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=14))

        if mni[person_id]['eclampsia_ip']:
            logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: scheduling immediate additional '
                        'treatment for eclampsia during labour for person %d', person_id)

            event = HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=14))

        if mni[person_id]['UR']:
            logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: scheduling immediate additional '
                        'treatment for uterine rupture during labour for person %d', person_id)

            event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=14)
                                                                )

        if df.at[person_id, 'la_current_labour_successful_induction'] == 'failed_induction':
            logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: scheduling a caesarean section'
                        ' for person %d', person_id)

            event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=14))


class HSI_Labour_ReceivesCareForObstructedLabour(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It manages the treatment of obstructed labour and referral in the instance of failed treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
    #    the_appt_footprint['InpatientDays'] = 1  # TODO: to be discussed with TH/TC best appt footprint to use
        the_appt_footprint['Over5OPD'] = 1

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

    #    pkg_code_obstructed_labour = pd.unique(consumables.loc[consumables[
    #                                                               'Intervention_Pkg'] ==
    #                                                           'Management of obstructed labour',
    #                                                           'Intervention_Pkg_Code'])[0]

        dummy_pkg_code = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] ==
                                                   'HIV Testing Services',
                                                   'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [dummy_pkg_code],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForObstructedLabour'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.info('This is HSI_Labour_ReceivesCareForObstructedLabour, management of obstructed labour for '
                     'person %d on date %s',
                     person_id, self.sim.date)

# =====================================  OBSTRUCTED LABOUR TREATMENT ==================================================

        # Differentiate between CPD and other?

        if mni[person_id]['labour_is_currently_obstructed']:
            treatment_effect = params['prob_deliver_ventouse']
            random = self.sim.rng.random_sample(size=1)
            if treatment_effect > random:
                # df.at[person_id, 'la_obstructed_labour'] = False
                mni[person_id]['labour_is_currently_obstructed'] = False
                mni[person_id]['mode_of_delivery'] = 'AVDV'
                print('Treatment success- delivery by Ventouse')
                # add here effect of antibiotics?
            else:
                print('Treatment failure- attempt forceps')
                treatment_effect = params['prob_deliver_forceps']
                random = self.sim.rng.random_sample(size=1)
                if treatment_effect > random:
                    # df.at[person_id, 'la_obstructed_labour'] = False
                    mni[person_id]['labour_is_currently_obstructed'] = False
                    mni[person_id]['mode_of_delivery'] = 'AVDF'
                    print('Treatment success- delivery by Forceps')
                    # add here effect of antibiotitcs?
                else:
                    print('Treatment Failure- Referral for CS ')
                    event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=14)
                                                                        )

                    logger.info(
                        'This is HSI_Labour_ReceivesCareForObstructedLabour referring person %d for a caesarean section'
                        ' following unsuccessful attempts at assisted vaginal delivery ',
                        person_id)

                    # Symphysiotomy??? Does this happen in malawi


class HSI_Labour_ReceivesCareForMaternalSepsis(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It manages the
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
    #   the_appt_footprint['InpatientDays'] = 1
        the_appt_footprint['Over5OPD'] = 1

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

    #   pkg_code_sepsis = pd.unique(consumables.loc[consumables[
    #                                                    'Intervention_Pkg'] ==
    #                                                'Maternal sepsis case management',
    #                                                'Intervention_Pkg_Code'])[0]

        dummy_pkg_code = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] ==
                                                   'HIV Testing Services',
                                                   'Intervention_Pkg_Code'])[0]
        the_cons_footprint = {
                    'Intervention_Package_Code': [dummy_pkg_code],
                    'Item_Code': []
                }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForMaternalSepsis'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.info('This is HSI_Labour_ReceivesCareForMaternalSepsis, management of maternal sepsis for '
                             'person %d on date %s',
                             person_id, self.sim.date)

# =====================================  MATERNAL SEPSIS TREATMENT ====================================================

        if mni[person_id]['sepsis']:
            treatment_effect = params['prob_cure_antibiotics']
            random = self.sim.rng.random_sample(size=1)
            if treatment_effect > random:
                mni[person_id]['sepsis'] = False
                print('Treatment success- antibiotics')


class HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy(Event, IndividualScopeEventMixin):
    # TODO: consider refactoring as HSI_Labour_ReceivesCareForHypertensiveDisorders??
    """
    This is a Health System Interaction Event.
    It manages the
        """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
    #   the_appt_footprint['InpatientDays'] = 1  # TODO: to be discussed with TH/TC best appt footprint to use
        the_appt_footprint['Over5OPD'] = 1

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

    #    pkg_code_eclampsia = pd.unique(consumables.loc[consumables[
    #                                                       'Intervention_Pkg'] ==
    #                                                   'Management of eclampsia',
    #                                                   'Intervention_Pkg_Code'])[0]

        dummy_pkg_code = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] ==
                                                   'HIV Testing Services',
                                                   'Intervention_Pkg_Code'])[0]


        the_cons_footprint = {
                    'Intervention_Package_Code': [dummy_pkg_code],
                    'Item_Code': []
                }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForEclampsia'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.info('This is HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy, management of hypertensive '
                    'disorders of pregnancy for person %d on date %s',
                             person_id, self.sim.date)

# =======================================  HYPERTENSION TREATMENT ======================================================

# =======================================  SEVERE PRE-ECLAMPSIA TREATMENT ==============================================

# =======================================  ECLAMPSIA TREATMENT ========================================================

        if mni[person_id]['eclampsia_ip'] or mni[person_id]['eclampsia_pp']:
            treatment_effect = params['prob_cure_mgso4']
            random = self.sim.rng.random_sample()
            if treatment_effect > random:
                mni[person_id]['eclampsia'] = False
                print('Treatment success- MgSO4')
            else:
                print('1st line treatment failure- additional MgSO4 required')
                random = self.sim.rng.random_sample()
                if treatment_effect > random:
                    mni[person_id]['eclampsia'] = False
                    print('Treatment success- MgSO4')
                else:
                    treatment_effect = params['prob_cure_diazepam']
                    random = self.sim.rng.random_sample()
                    if treatment_effect > random:
                        mni[person_id]['eclampsia'] = False
                        print('Treatment success- Diazepam')
                    else:
                        print('Seizures uncontolled - emergency caesarean & ICU required')
                        event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
                        self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                            priority=0,
                                                                            topen=self.sim.date,
                                                                            tclose=self.sim.date + DateOffset(days=14)
                                                                            )

                        logger.info(
                            'This is HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy referring person %d '
                            'for a caesarean section'
                            ' following uncontrolled eclampsia',
                            person_id)

        # Guidelines suggest assisting with second stage of labour via AVD - consider including this?


class HSI_Labour_ReceivesCareForMaternalHaemorrhage(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It manages the
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
    #   the_appt_footprint['InpatientDays'] = 1
        the_appt_footprint['Over5OPD'] = 1

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

    #    pkg_code_pph = pd.unique(consumables.loc[consumables[
    #                                                 'Intervention_Pkg'] ==
    #                                             'Treatment of postpartum hemorrhage',
    #                                             'Intervention_Pkg_Code'])[0]

    #    item_code_aph1 = pd.unique(consumables.loc[consumables['Items'] == 'Blood, one unit', 'Item_Code'])[0]
    #    item_code_aph2 = pd.unique(consumables.loc[consumables['Items'] == 'Lancet, blood, disposable', 'Item_Code'])
    #                                                                                                               [0]
    #    item_code_aph3 = pd.unique(consumables.loc[consumables['Items'] == 'Test, hemoglobin', 'Item_Code'])[0]
    #    item_code_aph4 = pd.unique(consumables.loc[consumables['Items'] == 'IV giving/infusion set, with needle',
    #                                               'Item_Code'])[0]
    #    item_code_aph5 = pd.unique(consumables.loc[consumables['Items'] == 'Gloves, surgeon’s, latex, disposable,'
    #                                                                       ' sterile, pair', 'Item_Code'])[0]

        dummy_pkg_code = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] ==
                                                   'HIV Testing Services',
                                                 'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
                    'Intervention_Package_Code': [dummy_pkg_code],
                    'Item_Code': []
                }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForMaternalHaemorrhage'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.info('This is HSI_Labour_ReceivesCareForMaternalHaemorrhage, management of obstructed labour for person '
                    '%d on date %s', person_id, self.sim.date)

# ===================================  ANTEPARTUM HAEMORRHAGE TREATMENT ===============================================
        # TODO: consider severity grading?

        etiology = ['PA', 'PP']  # May need to move this to allow for risk factors?
        probabilities = [0.67, 0.33]  # DUMMY
        random_choice = self.sim.rng.choice(etiology, size=1, p=probabilities)

        mni[person_id]['source_aph'] = random_choice  # Storing as high chance of SB in severe placental abruption
        mni[person_id]['units_transfused'] = 2

        event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
        self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                            priority=0,
                                                            topen=self.sim.date,
                                                            tclose=self.sim.date + DateOffset(days=14)
                                                            )

# ======================================= POSTPARTUM HAEMORRHAGE ======================================================
        # First we use a probability weighted random draw to determine the underlying etiology of this womans PPH

        if mni[person_id]['PPH']:
            etiology = ['UA', 'RPP']
            probabilities = [0.67, 0.33]  # dummy
            random_choice = self.sim.rng.choice(etiology, size=1, p=probabilities)
            mni[person_id]['source_pph'] = random_choice
            # Todo: add a level of severity of PPH -yes

            # ======================== TREATMENT CASCADE FOR ATONIC UTERUS:=============================================

            # Here we use a treatment cascade adapted from Malawian Obs/Gynae guidelines
            # Women who are bleeding due to atonic uterus first undergo medical management, oxytocin IV, misoprostol PR
            # and uterine massage in an attempt to stop bleeding

            if mni[person_id]['source_pph'] == 'UA':
                print('This PPH is due to Uterine Atony')
                random = self.sim.rng.random_sample(size=1)
                if params['prob_cure_oxytocin'] > random:
                    mni[person_id]['PPH'] = False
                    print('Treatment success- Oxytocin stopped the bleed')
                else:
                    random = self.sim.rng.random_sample(size=1)
                    if params['prob_cure_misoprostol'] > random:
                        mni[person_id]['PPH'] = False
                        print('Treatment success- Misoprostol stopped the bleed')
                    else:
                        random = self.sim.rng.random_sample(size=1)
                        if params['prob_cure_uterine_massage'] > random:
                            mni[person_id]['PPH'] = False
                            print('Treatment success- uterine massage stopped the bleed')
                        else:
                            random = self.sim.rng.random_sample(size=1)
                            if params['prob_cure_uterine_tamponade'] > random:
                                mni[person_id]['PPH'] = False
                                print('Treatment success- a uterine tamponade stopped the bleed')

            # Todo: consider the impact of oxy + miso + massage as ONE value, Discuss with expert

            # ===================TREATMENT CASCADE FOR RETAINED PRODUCTS/PLACENTA:====================================
            if mni[person_id]['source_pph'] == 'RPP':
                print('This PPH is due to retained products/placenta')
                random = self.sim.rng.random_sample(size=1)
                if params['prob_cure_manual_removal'] > random:
                    mni[person_id]['PPH'] = False
                    print('Treatment success-products removed manually')
                    # blood?

            if mni[person_id]['PPH']:
                print('Treatment failure-medically management of bleed has failed, referred for surgical care')
                event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=14)
                                                                    )


class HSI_Labour_ReceivesCareForPostpartumPeriod(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This event manages the Health System Interaction for women who receive post partum care following delivery
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
    #   the_appt_footprint['AccidentsandEmerg'] = 1
        the_appt_footprint['Over5OPD'] = 1

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_post_partum_care =pd.unique(consumables.loc[consumables[
                                                     'Intervention_Pkg'] ==
                                                 'Active management of the 3rd stage of labour',
                                                 'Intervention_Pkg_Code'])[0]

        dummy_pkg_code = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] ==
                                                   'HIV Testing Services',
                                                   'Intervention_Pkg_Code'])[0]

        the_cons_footprint = {
            'Intervention_Package_Code': [dummy_pkg_code],
            'Item_Code': []
        }

    # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForPostpartumPeriod'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [1, 2, 3]  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.info('This is HSI_Labour_ReceivesCareForPostpartumPeriod: Providing skilled attendance following birth '
                    'for person %d', person_id)

    #  =========================  ACTIVE MANAGEMENT OF THE THIRD STAGE  ===============================================

        # Here we apply a risk reduction of post partum bleeding following active management of the third stage of
        # labour (additional oxytocin, uterine massage and controlled cord traction)

        adjusted_maternal_pph_risk = mni[person_id]['risk_pph'] * params['rr_pph_amtsl']
        mni[person_id]['risk_pph'] = adjusted_maternal_pph_risk

    # ===============================  POSTPARTUM COMPLICATIONS ========================================================

        # TODO: link eclampsia/sepsis diagnosis in SBA and PPC
        random = self.sim.rng.random_sample(size=1)
        if random < mni[person_id]['risk_pp_eclampsia']:
            df.at[person_id, 'la_eclampsia'] = True
            mni[person_id]['eclampsia_pp'] = True
            logger.info('person %d is experiencing eclampsia in a health facility following birth',
                        person_id)
            logger.info('%s|eclampsia|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

        if random < mni[person_id]['risk_pph']: # increased liklihood of PPH based on whats happened so far? (APH, CS)
            df.at[person_id, 'la_pph'] = True
            mni[person_id]['PPH'] = True
            logger.info('person %d is experiencing an postpartum haemorrhage in a health facility following birth',
                        person_id)
            logger.info('%s|postpartum_haemorrhage|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

        random = self.sim.rng.random_sample(size=1)
        if random < mni[person_id]['risk_pp_sepsis']:
            df.at[person_id, 'la_sepsis'] = True
            mni[person_id]['sepsis_pp'] = True
            logger.info('person %d has developed maternal sepsis in a health facility following delivery',
                        person_id)
            logger.info('%s|maternal_sepsis|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

        # =============================  SCHEDULING ADDITIONAL TREATMENT ==================================================

        if mni[person_id]['sepsis']:
            logger.info('This is HSI_Labour_ReceivesCareForPostpartumPeriod: scheduling immediate additional '
                        'treatment for maternal sepsis during the postpartum period for person %d', person_id)

            event = HSI_Labour_ReceivesCareForMaternalSepsis(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                        priority=0,
                                                        topen=self.sim.date,
                                                        tclose=self.sim.date + DateOffset(days=14)
                                                        )


        if mni[person_id]['PPH']:
            logger.info('This is HSI_Labour_ReceivesCareForPostpartumPeriod: scheduling immediate additional '
                        'treatment for antepartum haemorrhage during the postpartum period for person %d', person_id)

            event = HSI_Labour_ReceivesCareForMaternalHaemorrhage(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=14)
                                                                    )

        if mni[person_id]['eclampsia_pp']:
            logger.info('This is HSI_Labour_ReceivesCareForPostpartumPeriod: scheduling immediate additional '
                        'treatment for eclampsia during the postpartum period for person %d', person_id)

            event = HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=14)
                                                                    )


class HSI_Labour_ReferredForSurgicalCareInLabour(Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This event manages the Health System Interaction for a woman who needs to be referred to undergo an emergency
    surgical management of complications arising in labour, in the postpartum period or for caesarean section
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()

    #   the_appt_footprint['MajorSurg'] = 1  # this appt could be used for uterine repair/pph management
    #   the_appt_footprint['Csection'] = 1
        the_appt_footprint['Over5OPD'] = 1

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_cs = pd.unique(consumables.loc[consumables[
                                                        'Intervention_Pkg'] ==
                                                'Cesearian Section with indication (with complication)',  # or without?
                                                'Intervention_Pkg_Code'])[0]

        dummy_pkg_code = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] ==
                                                   'HIV Testing Services',
                                                   'Intervention_Pkg_Code'])[0]

        # pkg_code_uterine_repair
        # pkg_code_pph_surgery
        # +/- hysterectomy?

        the_cons_footprint = {
            'Intervention_Package_Code': [dummy_pkg_code],
            'Item_Code': []
        }

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReferredForSurgicalCareInLabour'
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = [2, 3]  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.info('This is HSI_Labour_ReferredForSurgicalCareInLabour,providing surgical care during labour and the'
                    ' postpartum period for person %d on date %s', person_id, self.sim.date)

# ====================================== EMERGENCY CAESAREAN SECTION ==================================================

        if (mni[person_id]['UR']) or (mni[person_id]['APH']) or (mni[person_id]['ip_eclampsia']) or\
            (mni[person_id]['labour_is_currently_obstructed']) or \
            (df.at[person_id, 'la_current_labour_successful_induction'] == 'failed_induction'):
            # Consider all indications (elective)
            mni[person_id]['APH'] = False
            # reset eclampsia status?
            mni[person_id]['labour_is_currently_obstructed'] = False
            mni[person_id]['mode_of_delivery'] = 'CS'
            df.at[person_id, 'la_total_deliveries_by_cs'] = +1
            print('Treatment success- This person has delivered via caesarean')
            # apply risk of death from CS?

# ====================================== UTERINE REPAIR ==============================================================
        # Next we determine if the uterus can be repaired surgically
        # if df.at[person_id,' la_uterine_rupture']:

        if mni[person_id]['UR']:
            random = self.sim.rng.random_sample(size=1)
            if params['prob_cure_uterine_repair'] > random:
                # df.at[person_id, 'la_uterine_rupture'] = False
                mni[person_id]['UR'] = False
                print('Treatment success- this persons uterine rupture has been repaired surgically')
        # In the instance of failed surgical repair, the woman undergoes a hysterectomy
            else:
                print('Treatment Failure- this persons uterus couldnt be repaired, they are not undergoing '
                      'a hysterectomy')
                random = self.sim.rng.random_sample(size=1)
                if params['prob_cure_hysterectomy'] > random:
                    # df.at[person_id, 'la_uterine_rupture'] = False
                    mni[person_id]['UR'] = False

# ================================== SURGERY FOR UNCONTROLLED POSTPARTUM HAEMORRHAGE ==================================

        if mni[person_id]['PPH']:
            random = self.sim.rng.random_sample(size=1)
            if params['prob_cure_uterine_ligation'] > random:
                mni[person_id]['PPH'] = False
                print('Treatment success- this bleed has been stopped by uterine ligation')
            else:
                random = self.sim.rng.random_sample(size=1)
                if params['prob_cure_b_lych'] > random:
                    mni[person_id]['PPH'] = False
                    print('Treatment success- this bleed has been stopped by b-lynch suturing')
                else:
                    random = self.sim.rng.random_sample(size=1)
                    # Todo: similarly consider bunching surgical interventions
                    if params['prob_cure_hysterectomy'] > random:
                        mni[person_id]['PPH'] = False
                        print('Treatment success- this bleed has been stopped by a hysterectomy')


class LabourLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles lifestyle logging"""
    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
    #    self.repeat = 12
    #    super().__init__(module, frequency=DateOffset(days=self.repeat))
        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props

        #  MATERNAL MORTALITY RATIO:

        # live birth total
        one_year_prior = self.sim.date - np.timedelta64(1, 'Y')
        live_births = df.index[(df.date_of_birth > one_year_prior) & (df.date_of_birth < self.sim.date)]
        live_births_sum = len(live_births)
        print(live_births_sum)

        deaths = df.index[(df.la_maternal_death == True) & (df.la_maternal_death_date > one_year_prior) &
                          (df.la_maternal_death_date < self.sim.date)]

        cumm_deaths = len(deaths)
        print(cumm_deaths)

        mmr = cumm_deaths/live_births_sum * 100000
        print('The maternal mortality ratio for this year is', mmr)

        # Still Birth Rate

        # Perinatal Mortality
        #

        # Disease Incidence
        # Intervention incidence

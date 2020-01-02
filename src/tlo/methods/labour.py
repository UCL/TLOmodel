import logging

import pandas as pd
import numpy as np
from pathlib import Path

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.methods import demography
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
LOG_FILENAME = 'labour.log'
logging.basicConfig(filename=LOG_FILENAME,
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)


class Labour (Module):

    """
    This module models labour, delivery, immediate postpartum period and skilled birth attendance"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # Here we create a dictionary to store additional information around delivery and birth
        self.mother_and_newborn_info = dict()

    PARAMETERS = {

        #  ===================================  NATURAL HISTORY PARAMETERS =============================================
        'prob_pregnancy': Parameter(
            Types.REAL, 'baseline probability of pregnancy - currently included as a dummy parameter'),
        'prob_prom': Parameter(
            Types.REAL, 'probability of a woman in term labour having had experience prolonged rupture of membranes'),
        'prob_pl_ol': Parameter(
            Types.REAL, 'probability of a woman entering prolonged/obstructed labour'),
        'rr_PL_OL_nuliparity': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if they are nuliparous'),
        'rr_PL_OL_para1': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if they have a parity of 1'),
        'rr_PL_OL_age_less20': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her age is less'
                        'than 20 years'),
        'prob_ptl': Parameter(
            Types.REAL, 'baseline probability of a woman entering labour at less than 37 weeks gestation'),
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
            Types.REAL, 'probability of a uterine rupture during labour'),
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
            Types.REAL, 'case fatality rate for postpartum haemorrhage'),
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
            Types.REAL, 'relative risk of maternal sepsis following prophylactic antibiotics for PROM in a facility'),
        'rr_newborn_sepsis_proph_abx': Parameter(
            Types.REAL, 'relative risk of newborn sepsis following prophylactic antibiotics for '
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
                        'with severe pre-eclampsia'),
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
        'prob_cure_b_lynch': Parameter(
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
        'la_currently_in_labour': Property(Types.BOOL, 'whether this woman is currently in labour'),
        'la_current_labour_successful_induction': Property(Types.CATEGORICAL, 'Not Induced, Successful Induction, '
                                                                              'Failed Induction',
                                                           categories=['not_induced', 'successful_induction',
                                                                       'failed_induction']),
        'la_intrapartum_still_birth': Property(Types.BOOL, 'whether this womans most recent pregnancy has ended '
                                                           'in a stillbirth'),
        'la_parity': Property(Types.INT, 'total number of previous deliveries'),
        'la_total_deliveries_by_cs': Property(Types.INT, 'number of previous deliveries by caesarean section'),
        'la_has_previously_delivered_preterm': Property(Types.BOOL, 'whether the woman has had a previous preterm '
                                                                    'delivery for any of her previous deliveries'),
        'la_obstructed_labour': Property(Types.BOOL, 'whether this womans labour has become obstructed'),
        'la_ol_disability': Property(Types.BOOL, 'disability associated with obstructed labour'),
        'la_aph': Property(Types.BOOL, 'whether the woman has experienced an antepartum haemorrhage in this delivery'),
        'la_uterine_rupture': Property(Types.BOOL, 'whether the woman has experienced uterine rupture in this delivery'),
        'la_ur_disability': Property(Types.BOOL, 'disability associated with uterine rupture'),
        'la_sepsis': Property(Types.BOOL, 'whether the woman has developed sepsis associated with in this delivery'),
        'la_sepsis_disability': Property(Types.BOOL, 'disability associated with maternal sepsis'),
        'la_eclampsia': Property(Types.BOOL, 'whether the woman has experienced an eclamptic seizure in this delivery'),
        'la_eclampsia_disability': Property(Types.BOOL, 'disability associated with maternal haemorrhage'),
        'la_pph': Property(Types.BOOL, 'whether the woman has experienced an postpartum haemorrhage in this delivery'),
        'la_haemorrhage_disability': Property(Types.BOOL, 'disability associated with maternal haemorrhage'),
        # TODO: above property could be categorical to reflect severity of bleed and map better with DALY weights
        'la_maternal_death': Property(Types.BOOL, ' whether the woman has died as a result of this pregnancy'),  # DUMMY
        'la_maternal_death_date': Property(Types.DATE, 'date of death for a date in pregnancy')  # DUMMY
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_LabourSkilledBirthAttendance.xlsx',
                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)
        params = self.parameters

        # Here we will include DALY weights if applicable...

        if 'HealthBurden' in self.sim.modules.keys():
            params['daly_wt_haemorrhage_moderate'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=339)
            params['daly_wt_haemorrhage_severe'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=338)
            params['daly_wt_maternal_sepsis'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=340)
            params['daly_wt_eclampsia'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=347)
            params['daly_wt_obstructed_labour'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=348)
            # TODO: source DALY weight for Uterine Rupture

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

        df.loc[df.sex == 'F', 'la_current_labour_successful_induction'] = 'not_induced'
        df.loc[df.sex == 'F', 'la_currently_in_labour'] = False
        df.loc[df.sex == 'F', 'la_intrapartum_still_birth'] = False
        df.loc[df.sex == 'F', 'la_parity'] = 0
        df.loc[df.sex == 'F', 'la_total_deliveries_by_cs'] = 0
        df.loc[df.sex == 'F', 'la_has_previously_delivered_preterm'] = False
        df.loc[df.sex == 'F', 'la_due_date_current_pregnancy'] = pd.NaT
        df.loc[df.sex == 'F', 'la_obstructed_labour'] = False
        df.loc[df.sex == 'F', 'la_ol_disability'] = False
        df.loc[df.sex == 'F', 'la_aph'] = False
        df.loc[df.sex == 'F', 'la_uterine_rupture'] = False
        df.loc[df.sex == 'F', 'la_ur_disability'] = False
        df.loc[df.sex == 'F', 'la_eclampsia'] = False
        df.loc[df.sex == 'F', 'la_eclampsia_disability'] = False
        df.loc[df.sex == 'F', 'la_pph'] = False
        df.loc[df.sex == 'F', 'la_haemorrhage_disability'] = False
        df.loc[df.sex == 'F', 'la_maternal_death'] = False
        df.loc[df.sex == 'F', 'la_maternal_death_date'] = pd.NaT

# -----------------------------------ASSIGN PREGNANCY AND DUE DATE AT BASELINE (DUMMY) --------------------------------
        # TODO: Discuss with Tim Colbourn if he still plans to assign pregnancy at baseline

        # !!!!!!!!!!!!!!!!(DUMMY CODE) THIS WILL BE REPLACED BY CONTRACEPTION CODE (TC) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

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

        # Randomly generate a number of weeks gestation between 1-39 for all pregnant women at baseline
        simdate = pd.Series(self.sim.date, index=pregnant_idx)
        dfx = pd.concat((simdate, random_draw), axis=1)
        dfx.columns = ['simdate', 'random_draw']
        dfx['gestational_age_in_weeks'] = (39 - 39 * dfx.random_draw)
        df.loc[pregnant_idx, 'ps_gestational_age'] = dfx.gestational_age_in_weeks.astype(int)

        # Use gestational age to calculate when the woman's baby was conceived
        dfx['la_conception_date'] = dfx['simdate'] - pd.to_timedelta(dfx['gestational_age_in_weeks'], unit='w')

        # Apply a due date of 9 months in the future from the date of conception for each woman
        dfx['due_date_mth'] = 39 - df['ps_gestational_age']
        dfx['due_date'] = dfx['simdate'] + pd.to_timedelta(dfx['due_date_mth'], unit='w')
        df.loc[pregnant_idx, 'date_of_last_pregnancy'] = dfx.la_conception_date
        df.loc[pregnant_idx, 'la_due_date_current_pregnancy'] = dfx.due_date

        # For women whose gestation is less than 37 weeks at baseline, we determine if they will go into preterm/
        # post term or term labour:

        # First we apply the risk of preterm birth to these women
        non_term_women = dfx.index[dfx.gestational_age_in_weeks < 36]
        eff_prob_ptl = pd.Series(params['prob_ptl'], index=non_term_women)
        random_draw = pd.Series(rng.random_sample(size=len(non_term_women)),
                                index=non_term_women)

        dfx = pd.concat((eff_prob_ptl, random_draw), axis=1)
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
        idx_e_ptl_concep = pd.Series(df.date_of_last_pregnancy, index=idx_e_ptl)
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

        # !!!!!!!!!!!!!!!!(DUMMY CODE) THIS WILL BE REPLACED BY CONTRACEPTION CODE (TC) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
            df.at[child_id, 'la_currently_in_labour'] = pd.NaT
            df.at[child_id, 'la_current_labour_successful_induction'] = 'not_induced'
            df.at[child_id, 'la_intrapartum_still_birth'] = False
            df.at[child_id, 'la_parity'] = 0
            df.at[child_id, 'la_total_deliveries_by_cs'] = 0
            df.at[child_id, 'la_has_previously_delivered_preterm'] = False
            df.at[child_id, 'la_obstructed_labour'] = False
            df.at[child_id, 'la_ol_disability'] = False
            df.at[child_id, 'la_aph'] = False
            df.at[child_id, 'la_uterine_rupture'] = False
            df.at[child_id, 'la_ur_disability'] = False
            df.at[child_id, 'la_sepsis'] = False
            df.at[child_id, 'la_sepsis_disability'] = False
            df.at[child_id, 'la_eclampsia'] = False
            df.at[child_id, 'la_eclampsia_disability'] = False
            df.at[child_id, 'la_pph'] = False
            df.at[child_id, 'la_haemorrhage_disability'] = False
            df.at[child_id, 'la_maternal_death'] = False
            df.at[child_id, 'la_maternal_death_date'] = pd.NaT

        # If a mothers labour has resulted in a late term still birth her child is still generated by the simulation
        # but is_alive is reset to false to allow for monitoring of still birth rates

        # Log only live births:
        if ~df.at[mother_id, 'la_intrapartum_still_birth']:
            logger.info('%s|live_births|%s',
                        self.sim.date,
                        {
                            'mother': mother_id,
                            'child': child_id,
                            'mother_age': df.at[mother_id, 'age_years']
                        })

        if df.at[mother_id, 'la_intrapartum_still_birth']:
            #  N.B this will only record intrapartum stillbirth
            death = demography.InstantaneousDeath(self.sim.modules['Demography'], child_id,
                                                  cause='Stillbirth')
            self.sim.schedule_event(death, self.sim.date)

            # This property is then reset in case of future pregnancies/stillbirths
            df.loc[mother_id, 'la_intrapartum_still_birth'] = False


    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        logger.info('This is Labour, being alerted about a health system interaction '
                    'person %d for: %s', person_id, treatment_id)

        # TODO: Confirm when this function should be utilised.

    def report_daly_values(self):

        # TODO: work out why values have to be hard coded, wont read parameters?
        # TODO: Refine disability levels (could include severity) and explore a more elegant solution involving less
        #  properties

        logger.debug('This is Labour reporting my health values')

        df = self.sim.population.props  # shortcut to population properties dataframe
        p = self.parameters

        health_values_1 = df.loc[df.is_alive, 'la_ol_disability'].map(
            {False: 0, True: 0.324})  # p['daly_wt_obstructed_labour']
        health_values_1.name = 'Obstructed Labour'

        health_values_2 = df.loc[df.is_alive, 'la_eclampsia_disability'].map(
            {False: 0, True: 0.5})  # p['daly_wt_eclampsia']
        health_values_2.name = 'Eclampsia'

        health_values_3 = df.loc[df.is_alive, 'la_sepsis_disability'].map(
            {False: 0, True: 0.133})  # p['daly_wt_maternal_sepsis']
        health_values_3.name = 'Maternal Sepsis'

        health_values_4 = df.loc[df.is_alive, 'la_haemorrhage_disability'].map(  # TODO: consider severity
            {False: 0, True: 0.324})  # p['daly_wt_haemorrhage_severe']
        health_values_4.name = 'Antepartum Haemorrhage'

        health_values_5 = df.loc[df.is_alive, 'la_haemorrhage_disability'].map(  # TODO: consider severity
            {False: 0, True: 0.324})  # p['daly_wt_haemorrhage_severe']
        health_values_5.name = 'Postpartum Haemorrhage'

        health_values_6 = df.loc[df.is_alive, 'la_ur_disability'].map(  # TODO: consider severity
            {False: 0, True: 0.5})  # p['daly_wt_haemorrhage_severe']
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
        if ~df.at[individual_id, 'la_has_previously_delivered_preterm'] & (df.at[individual_id, 'age_years'] < 20):
            rf1 = params['rr_early_ptb_age<20']
        else:
            rf1 = 1

        if df.at[individual_id, 'la_has_previously_delivered_preterm'] & (df.at[individual_id, 'age_years'] > 20):
            rf2 = params['rr_early_ptb_prev_ptb']
        else:
            rf2 = 1

        riskfactors = rf1 * rf2
        if riskfactors == 1:
            eff_prob_early_ptb = params['prob_early_ptb']
        else:
            eff_prob_early_ptb = riskfactors * params['prob_early_ptb']

        # Then we determine her risk of late preterm birth based on independent risk factors
        if df.at[individual_id, 'la_has_previously_delivered_preterm']:
            rf1 = params['rr_late_ptb_prev_ptb']
        else:
            rf1 = 1

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
                # TODO: To explore causal influences on post term labour
                random = np.random.randint(42, 46, size=1)
                random = int(random)
                df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] + \
                                                                        pd.Timedelta(random, unit='W')
                due_date = df.at[individual_id, 'la_due_date_current_pregnancy']
            else:
                random = np.random.randint(37, 41, size=1)
                random = int(random)
                df.at[individual_id, 'la_due_date_current_pregnancy'] = df.at[individual_id, 'date_of_last_pregnancy'] + \
                                                                        pd.Timedelta(random, unit='W')
                due_date = df.at[individual_id, 'la_due_date_current_pregnancy']

        # Labour is then scheduled on the newly generated due date
        self.sim.schedule_event(LabourEvent(self.module, individual_id, cause='labour'), due_date)
        # TODO: 'Local variable 'due_date' might be referenced before assignment more' - i can't see how


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

        # TODO: review in context of properties- ensure what should be a property IS one, and what SHOULDN'T be isnt.
        mni[individual_id] = {'labour_state': None,  # Term Labour (TL), Early Preterm (EPTL), Late Preterm (LPTL) or
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

        # This check ensures only women whose due date is the date of this event go into labour
        if df.at[individual_id, 'la_due_date_current_pregnancy'] == pd.NaT:
            logger.info('This is LabourEvent, person %d has reached their previously allocated due date but is not '
                        'entering labour at this time', individual_id)

        if df.at[individual_id, 'is_pregnant'] & df.at[individual_id, 'is_alive'] & \
            (df.at[individual_id, 'la_due_date_current_pregnancy'] == self.sim.date):
            df.at[individual_id, 'la_currently_in_labour'] = True

            # If a woman has been induced/attempted induction she will already be in a facility therefore delivery_
            # setting is set to facility
            if (df.at[individual_id, 'la_current_labour_successful_induction'] == 'failed_induction') or \
               (df.at[individual_id, 'la_current_labour_successful_induction'] == 'successful_induction'):
                    mni[individual_id]['delivery_setting'] = 'FD'
            # We then store her induction status if successful
            if df.at[individual_id, 'la_current_labour_successful_induction'] == 'successful_induction':
                    mni[individual_id]['induced_labour'] = True

            # Now we use gestational age to categorise the 'labour_state'
            if df.at[individual_id, 'is_pregnant'] & df.at[individual_id, 'is_alive']:
                if 37 <= df.at[individual_id, 'ps_gestational_age'] < 42:
                    mni[individual_id]['labour_state'] = 'TL'

                    logger.debug('This is LabourEvent, person %d has now gone into term labour on date %s',
                                 individual_id, self.sim.date)

                elif 24 <= df.at[individual_id, 'ps_gestational_age'] < 34:
                    mni[individual_id]['labour_state'] = 'EPTL'
                    df.at[individual_id, 'la_has_previously_delivered_preterm'] = True

                    logger.debug('This is LabourEvent, person %d has now gone into early preterm labour on date %s',
                                 individual_id, self.sim.date)

                elif 37 > df.at[individual_id, 'ps_gestational_age'] >= 34:
                    mni[individual_id]['labour_state'] = 'LPTL'
                    df.at[individual_id, 'la_has_previously_delivered_preterm'] = True

                    logger.debug('This is LabourEvent, person %d has now gone into late preterm labour on date %s',
                                 individual_id, self.sim.date)

                elif df.at[individual_id, 'ps_gestational_age'] > 41:
                    mni[individual_id]['labour_state'] = 'POTL'

                    logger.debug('This is LabourEvent, person %d is now overdue labour and is post-term  on date %s',
                                 individual_id, self.sim.date)

# =============================== PLACE HOLDER CARE SEEKING AND SCHEDULING (DUMMY) =====================================

                # TODO: We are awaiting the equation to determine how we deal with care seeking in labour

                # Here we apply a probability that a woman will seek care in a facility as she is in labour
                prob = 0.73
                random = self.sim.rng.random_sample(size=1)

                # Again only women who have not undergone induction will be seeking care
                if (df.at[individual_id, 'la_current_labour_successful_induction'] == 'not_induced') & (random < prob):
                    mni[individual_id]['delivery_setting'] = 'FD'
                    logger.info(
                        'This is LabourEvent, scheduling HSI_Labour_PresentsForSkilledAttendanceInLabour on date %s for'
                        ' person %d as they have chosen to seek care for delivery', self.sim.date, individual_id)

                    event = HSI_Labour_PresentsForSkilledAttendanceInLabour(self.module, person_id=individual_id)
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1)
                                                                        )
                    # TODO: Consider a reasonable tclose date

                    # TODO: Line of logic that says if woman cannot access care because of contraints then they are
                    #  marked as having an out of facility birth and increased risk of death is applied (and increased
                    #  risk of comp?)

                elif (df.at[individual_id, 'la_current_labour_successful_induction'] == 'not_induced') & (random > prob):
                    mni[individual_id]['delivery_setting'] = 'HB'
                    logger.debug(
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
                                                                        tclose=self.sim.date + DateOffset(days=1)
                                                                        )

# ============================ INDIVIDUAL RISK OF COMPLICATIONS DURING LABOUR =========================================

# ====================================== STATUS OF MEMBRANES ==========================================================

        # TODO: discuss with Tim C importance of modelling PPROM (would change structure)

        # Here we apply a risk that this woman's labour was preceded by premature rupture of membranes, in preterm
        # women this has likley predisposed their labour
                if (mni[individual_id]['labour_state'] == 'EPTL') or (mni[individual_id]['labour_state'] == 'LPTL'):
                    random = self.sim.rng.random_sample(size=1)
                    if random < params['prob_prom']:
                        mni[individual_id]['PROM'] = True

# ===================================  OBSTRUCTED LABOUR =============================================================

        # TODO: As OL is a contraindication of induction, should we skip it here for induced women? late diagnosis of
        # obstruction is possible surely

        # We identify if this woman's labour will become obstructed based on risk factors using the same
        # calculation as previous individual style events

                if (df.at[individual_id, 'la_parity'] == 0) & (df.at[individual_id, 'age_years'] >= 21):
                    rf1 = params['rr_PL_OL_nuliparity']
                else:
                    rf1 = 1

                if (df.at[individual_id, 'la_parity'] == 1) & (df.at[individual_id, 'age_years'] >= 21):
                    rf2 = params['rr_PL_OL_para1']
                else:
                    rf2 = 1
                if (df.at[individual_id, 'la_parity'] > 1) & (df.at[individual_id, 'la_parity'] < 3) &\
                    (df.at[individual_id, 'age_years'] < 20):
                    rf3 = params['rr_PL_OL_age_less20']
                else:
                    rf3 = 1

        # TODO: this formulation of the risk factors (which is used very often) might be more readble as: p_outcome =
        #  p_baseeline * (has_rf1*rr1) * (has_rf2*rr2) - discussed with TC may not work at individual level

                riskfactors = rf1*rf2*rf3
                eff_prob_ol = riskfactors * params['prob_pl_ol']

                # For all the following complications, if a woman has delivered in a facility, we store their effective
                # probability of experiencing this complication as we recalculate this risk based on preventative
                # interventions during the firs SBA HSI

                if mni[individual_id]['delivery_setting'] == 'FD':
                    mni[individual_id]['risk_ol'] = eff_prob_ol
                else:
                    random = self.sim.rng.random_sample(size=1)
                    if random < eff_prob_ol:
                        mni[individual_id]['labour_is_currently_obstructed'] = True
                        mni[individual_id]['labour_has_previously_been_obstructed'] = True
                        df.at[individual_id, 'la_obstructed_labour'] = True
                        df.at[individual_id, 'la_ol_disability'] = True

                        logger.debug('person %d has developed obstructed labour in the community on date %s',
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
                        df.at[individual_id, 'la_eclampsia_disability'] = True
                        mni[individual_id]['eclampsia_ip'] = True

                        logger.debug('person %d is experiencing intrapartum eclampsia in the community on date %s',
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

                # TODO: a/w review of evidence to determine strength of association between risk factors and
                #  praevia/abruption

                if mni[individual_id]['delivery_setting'] == 'FD':
                    mni[individual_id]['risk_aph'] = eff_prob_aph
                else:
                    random = self.sim.rng.random_sample(size=1)
                    if random < eff_prob_aph:
                        df.at[individual_id, 'la_aph'] = True
                        df.at[individual_id, 'la_haemorrhage_disability'] = True
                        mni[individual_id]['APH'] = True

                        logger.debug('person %d is experiencing an antepartum haemorrhage in the community on date %s',
                                    individual_id, self.sim.date)

                        logger.info('%s|antepartum_haemorrhage|%s', self.sim.date,
                                    {'age': df.at[individual_id, 'age_years'],
                                     'person_id': individual_id})


# ======================================= PUERPERAL SEPSIS ============================================================

        # Here we apply the incidence of Puerperal sepsis (obstetric), we are not modelling here sepsis of other causes

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
                        df.at[individual_id]['sepsis_disability'] = True
                        mni[individual_id]['sepsis_ip'] = True

                        logger.debug('person %d is experiencing intrapartum maternal sepsis in the community on date %s'
                                     ,individual_id, self.sim.date)

                        logger.info('%s|maternal_sepsis|%s', self.sim.date,
                                    {'age': df.at[individual_id, 'age_years'],
                                     'person_id': individual_id})

            # TODO: Consider how to quantitfy increased risk of sepsis (mtct) in the newborns of septic mothers

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
                        df.at[individual_id, 'la_ur_disability'] = True
                        mni[individual_id]['UR'] = True

                        logger.debug('person %d is experiencing uterine rupture in the community on date %s',
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
                logger.debug('This is LabourEvent scheduling a potential death on date %s for mother %d',
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

        df.at[mother_id, 'la_currently_in_labour'] = False

        # If the mother is alive and still pregnant we generate a live child and the woman is scheduled to move to the
        # postpartum event to determine if she experiences any additional complications
        if df.at[mother_id, 'is_alive'] and df.at[mother_id, 'is_pregnant']:
            #  TODO: Ensure women who have had an IP still birth still move to postpartum event
            self.sim.do_birth(mother_id)
            df.at[mother_id, 'ps_gestational_age'] = 0
            df.at[mother_id, 'is_pregnant'] = False
            df.at[mother_id, 'date_of_last_pregnancy'] = pd.NaT

            logger.debug('This is BirthEvent scheduling mother %d to undergo the PostPartumEvent following birth',
                         mother_id)
            self.sim.schedule_event(PostpartumLabourEvent(self.module, mother_id, cause='post partum'),
                                    self.sim.date)

        # As only live births contribute to parity we exlcuded women who have had an IP stillbirth
        if df.at[mother_id, 'is_alive'] and df.at[mother_id, 'is_pregnant'] & \
            ~df.at[mother_id,'la_intrapartum_still_birth']:
            df.at[mother_id, 'la_parity'] += 1

        # If the mother has died during childbirth the child is still generated with is_alive=false to monitor
        # stillbirth rates. She will not pass through the postpartum complication events
        if df.at[mother_id, 'is_alive'] == False & df.at[mother_id, 'is_pregnant'] == True & \
            (mni[mother_id]['death_in_labour'] == True):
            self.sim.do_birth(mother_id)
            df.at[mother_id, 'is_pregnant'] = False


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
                    df.at[individual_id,'la_haemorrhage_disability'] = True

                    logger.debug('person %d is experiencing a postpartum haemorrhage in the community on date %s',
                                 individual_id, self.sim.date)
                    logger.info('%s|postpartum_haemorrhage|%s', self.sim.date,
                                {'age': df.at[individual_id, 'age_years'],
                                 'person_id': individual_id})

            #  TODO: consider including severity of bleeding to match with DALYs

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
                    df.at[individual_id, 'la_sepsis_disability'] = True
                    mni[individual_id]['sepsis_pp'] = True

                    logger.debug('person %d is experiencing postpartum maternal sepsis in the community on date %s',
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
                    df.at[individual_id, 'la_eclampsia_disability'] = True

                    logger.debug('person %d is experiencing postpartum eclampsia in the community on date %s',
                                 individual_id, self.sim.date)
                    logger.info('%s|eclampsia|%s', self.sim.date,
                                {'age': df.at[individual_id, 'age_years'],
                                 'person_id': individual_id})

            # If a woman has delivered in a facility we schedule her to now receive additional care following birth
            if mni[individual_id]['delivery_setting'] == 'FD':
                logger.info('This is PostPartumEvent scheduling HSI_Labour_ReceivesCareForPostpartumPeriod for person '
                            '%d on date %s', individual_id,self.sim.date)
                event = HSI_Labour_ReceivesCareForPostpartumPeriod(self.module, person_id=individual_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1)
                                                                    )
                # TODO: same issue for women who seek care but cant be seek- comps wont be allocated!

            # We schedule all women to then go through the death event where those with untreated/unsuccessfully treated
            # complications may experience death
            self.sim.schedule_event(PostPartumDeathEvent(self.module, individual_id, cause='labour'), self.sim.date)
            logger.info('This is PostPartumEvent scheduling a potential death for person %d on date %s', individual_id,
                        self.sim.date + DateOffset(days=3))  # Date offsetted to allow for interventions

            # Here we schedule women to an event which resets 'daly' disability associated with delivery complications
            self.sim.schedule_event(DisabilityResetEvent(self.module, individual_id, cause='reset'),
                                    self.sim.date + DateOffset(months=1))


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

        # TODO: Will we apply a reduced CFR in the instance of unsuccessful interventions

        # First we determine if the mother will die due to her complication
        if mni[individual_id]['labour_is_currently_obstructed']:
            random = self.sim.rng.random_sample(size=1)
            if random < params['cfr_obstructed_labour']:
                mni[individual_id]['death_in_labour'] = True
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date

                # then if she does die, we determine if the child will still survive
                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_obstructed_labour_md']:
                    df.at[individual_id, 'la_intrapartum_still_birth'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True
                    df.at[individual_id, 'ps_previous_stillbirth'] = True

            # Otherwise we just determine if this complication will lead to a stillbirth
            else:
                random = self.sim.rng.random_sample(size=1)
                if random < params['prob_still_birth_obstructed_labour']:
                    df.at[individual_id, 'la_intrapartum_still_birth'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True
                    df.at[individual_id, 'ps_previous_stillbirth'] = True

        if mni[individual_id]['eclampsia_ip']:
            random = self.sim.rng.random_sample(size=1)
            if random < params['cfr_eclampsia']:
                mni[individual_id]['death_in_labour'] = True
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date

                random = self.sim.rng.random_sample()
                if random < params['prob_still_birth_eclampsia_md']:
                    df.at[individual_id, 'la_intrapartum_still_birth'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True
                    df.at[individual_id, 'ps_previous_stillbirth'] = True
            else:
                random = self.sim.rng.random_sample(size=1)
                if random < params['prob_still_birth_eclampsia']:
                    df.at[individual_id, 'la_intrapartum_still_birth'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True
                    df.at[individual_id, 'ps_previous_stillbirth'] = True

        # if df.at[individual_id, 'la_aph']:
        if mni[individual_id]['APH']:
            random = self.sim.rng.random_sample(size=1)
            if random < params['cfr_aph']:
                mni[individual_id]['death_in_labour'] = True
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date

                random = self.sim.rng.random_sample(size=1)
                if random < params['prob_still_birth_aph_md']:
                    df.at[individual_id, 'la_intrapartum_still_birth'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True
                    df.at[individual_id, 'ps_previous_stillbirth'] = True
            else:
                random = self.sim.rng.random_sample(size=1)
                if random < params['prob_still_birth_aph']:
                    df.at[individual_id, 'la_intrapartum_still_birth'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True
                    df.at[individual_id, 'ps_previous_stillbirth'] = True

        # if df.at[individual_id, 'la_sepsis']:
        if mni[individual_id]['sepsis_ip']:
            random = self.sim.rng.random_sample(size=1)
            if random < params['cfr_sepsis']:
                mni[individual_id]['death_in_labour'] = True
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date

                random = self.sim.rng.random_sample(size=1)
                if random < params['prob_still_birth_sepsis_md']:
                    df.at[individual_id, 'la_intrapartum_still_birth'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True
                    df.at[individual_id, 'ps_previous_stillbirth'] = True
            else:
                random = self.sim.rng.random_sample(size=1)
                if random < params['prob_still_birth_sepsis']:
                    df.at[individual_id, 'la_intrapartum_still_birth'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True
                    df.at[individual_id, 'ps_previous_stillbirth'] = True

        # if df.at[individual_id, 'la_uterine_rupture']:
        if mni[individual_id]['UR']:
            random = self.sim.rng.random_sample(size=1)
            if random < params['cfr_uterine_rupture']:
                mni[individual_id]['death_in_labour'] = True
                df.at[individual_id, 'la_maternal_death'] = True
                df.at[individual_id, 'la_maternal_death_date'] = self.sim.date

                random = self.sim.rng.random_sample(size=1)
                if random < params['prob_still_birth_ur_md']:
                    df.at[individual_id, 'la_intrapartum_still_birth'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True
                    df.at[individual_id, 'ps_previous_stillbirth'] = True
            else:
                random = self.sim.rng.random_sample(size=1)
                if random < params['prob_still_birth_ur']:
                    df.at[individual_id, 'la_intrapartum_still_birth'] = True
                    mni[individual_id]['stillbirth_in_labour'] = True
                    df.at[individual_id, 'ps_previous_stillbirth'] = True

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

        if df.at[individual_id, 'la_intrapartum_still_birth']:
            logger.info('@@@@ A Still Birth has occurred, to mother %s', individual_id)
            logger.info('%s|still_birth|%s', self.sim.date,
                        {'mother_id': individual_id})


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
            logger.debug('This is PostPartumDeathEvent scheduling a death for person %d on date %s who died due to '
                        'postpartum complications', individual_id,
                         self.sim.date)

            logger.info('%s|maternal_death|%s', self.sim.date,
                        {'age': df.at[individual_id, 'age_years'],
                            'person_id': individual_id})

            complication_profile = mni[individual_id]
            logger.debug('%s|labour_complications|%s', self.sim.date,
                        {'person_id': individual_id,
                         'labour_profile': complication_profile})
        else:
            complication_profile = mni[individual_id]
            # Surviving women pass through the DiseaseResetEvent to ensure all complication variable are set to false
            # TODO: Consider how best to deal with complications that are long lasting.
            self.sim.schedule_event(DiseaseResetEvent(self.module, individual_id, cause='reset'),
                                    self.sim.date + DateOffset(weeks=1))

            logger.debug('%s|labour_complications|%s', self.sim.date,
                        {'person_id': individual_id,
                         'labour_profile': complication_profile})


class DisabilityResetEvent (Event, IndividualScopeEventMixin):
    """resets a woman's disability properties one month after labour"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props

        # TODO: Confirm this allows enough time for model to count DALYs before resetting

        # Here we turn off all the properties which are used to count DALYs
        if df.at[individual_id, 'is_alive']:
            logger.debug('person %d is having their disability status reset', individual_id)

            df.at[individual_id, 'la_sepsis_disability'] = False
            df.at[individual_id, 'la_ol_disability'] = False
            df.at[individual_id, 'la_ur_disability'] = False
            df.at[individual_id, 'la_eclampsia_disability'] = False
            df.at[individual_id, 'la_haemorrhage_disability'] = False


class DiseaseResetEvent (Event, IndividualScopeEventMixin):
    """resets a woman's disability properties one month after labour"""

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props

        # This event ensures that for women who have survived delivery but have suffered a complication in which
        # treatment was unsuccessful have their diseases variables reset

        if df.at[individual_id, 'is_alive']:
            logger.debug('person %d is having their maternal disease status reset', individual_id)

            df.at[individual_id, 'la_sepsis'] = False
            df.at[individual_id, 'la_obstructed_labour'] = False
            df.at[individual_id, 'la_aph'] = False
            df.at[individual_id, 'la_uterine_rupture'] = False
            df.at[individual_id, 'la_eclampsia'] = False
            df.at[individual_id, 'la_pph'] = False

# ======================================================================================================================
# ================================ HEALTH SYSTEM INTERACTION EVENTS ====================================================
# ======================================================================================================================
    # TODO: Discuss with the team the potential impact of deviating from guidelines- at present if a woman presents,
    #  staff time is available and resources are available she gets all needed interventions (this may not be the case,
    #  question of quality

class HSI_Labour_PresentsForInductionOfLabour(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This interaction manages induction of labour for women with indications identified antenatally such as severe
    pre-eclampsia and being post term
    """

    # TODO: await antenatal care module to schedule induction
    # TODO: women with praevia, obstructed labour, should be induced

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)
        
        self.TREATMENT_ID = 'Labour_PresentsForInductionOfLabour'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1
        # TODO: review appt footprint as this wont include midwife time?

        # Define the necessary information for an HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.debug('This is HSI_Labour_PresentsForInductionOfLabour, person %d is attending a health facility to have'
                     'their labour induced on date %s', person_id, self.sim.date)

        # TODO: discuss squeeze factors/ consumable back up with TC

        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        # Initial request for consumables needed
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_induction = pd.unique(consumables.loc[consumables[
                                                       'Intervention_Pkg'] ==
                                                       'Induction of labour (beyond 41 weeks)',
                                                       'Intervention_Pkg_Code'])[0]
        # TODO: review induction guidelines to confirm appropriate 1st line/2nd line consumable use

        consumables_needed = {'Intervention_Package_Code': [{pkg_code_induction: 1}],
                                  'Item_Code': [], }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self, cons_req_as_footprint=consumables_needed
            )
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_induction]:
            logger.debug('pkg_code_induction is available, so use it.')
            # TODO: reschedule if consumables aren't available at this point in time?
        else:
            logger.debug('PkgCode1 is not available, so can' 't use it.')

        # We use a random draw to determine if this womans labour will be successfully induced
        # Indications: Post term, eclampsia, severe preeclampsia, mild preeclampsia at term, PROM > 24 hrs at term
        # or PPROM > 34 weeks EGA, and IUFD.

        random = self.sim.rng.random_sample(size=1)
        if random < params['prob_successful_induction']:
            logger.info('Person %d has had her labour successfully induced', person_id)
            df.at[person_id, 'la_current_labour_successful_induction'] = 'successful_induction'
            self.sim.schedule_event(LabourEvent(self.module, person_id, cause='labour'), self.sim.date)

            # For women whose induction fails they will undergo caesarean section
        else:
            logger.info('Persons %d labour has been unsuccessful induced', person_id)
            df.at[person_id, 'la_current_labour_successful_induction'] = 'failed_induction'
            # TODO: schedule CS or second attempt induction? -- will need to lead to labour event

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
    #    actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_PresentsForInductionOfLabour: did not run')
        pass


class HSI_Labour_PresentsForSkilledAttendanceInLabour(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This interaction manages the care a woman receives when she is admitted to a facility in spontanous labour or in
    labour following an induction
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)
        
        self.TREATMENT_ID = 'Labour_PresentsForSkilledAttendanceInLabour'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['NormalDelivery'] = 1

        # Define the necessary information for an HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        # TODO: Squeeze factor, consumable conditions?

        logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: Providing initial skilled attendance '
                    'at birth for person %d on date %s', person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_sba_uncomp = pd.unique(consumables.loc[consumables[
                                                      'Intervention_Pkg'] ==
                                                  'Vaginal delivery - skilled attendance',
                                                  'Intervention_Pkg_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code_sba_uncomp: 1}],
            'Item_Code': [],
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sba_uncomp]:
            logger.debug('PkgCode1 is available, so use it.')
        else:
            logger.debug('PkgCode1 is not available, so can' 't use it.')
            # TODO: If delivery pack isnt availble then birth will still occur but should have risk of sepsis?

        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

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

        # TODO: Discuss again with Tim C if it is worth trying to quantify this, as many evaluative studies look at
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

        # TODO: STEROIDS - effect applied directly to newborns (need consumables here) store within maternal MNI
        # TOXOLYTICS !? - WHO advises against

# ===================================  COMPLICATIONS OF LABOUR ========================================================

    # Here, using the adjusted risks calculated following 'in-labour' interventions to determine which complications a
    # woman may experience and store those in the dataframe

        random = self.sim.rng.random_sample(size=1)
        if random < mni[person_id]['risk_ol']:
            df.at[person_id, 'la_obstructed_labour'] = True
            df.at[person_id, 'la_ol_disability'] = True
            mni[person_id]['labour_is_currently_obstructed'] = True
            mni[person_id]['labour_has_previously_been_obstructed'] = True

            logger.debug('person %d is experiencing obstructed labour in a health facility',
                        person_id)

            # TODO: issue- if we apply risk of both UR and OL here then we will negate the effect of OL treatment on
            #  reduction of incidence of UR

        random = self.sim.rng.random_sample(size=1)
        if random < mni[person_id]['risk_ip_eclampsia']:
            df.at[person_id, 'la_eclampsia'] = True
            df.at[person_id, 'la_eclampsia_disability'] = True
            mni[person_id]['eclampsia_ip'] = True

            logger.debug('person %d is experiencing eclampsia in a health facility',
                        person_id)
            logger.info('%s|eclampsia|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

        if random < mni[person_id]['risk_aph']:
            df.at[person_id, 'la_aph'] = True
            df.at[person_id, 'la_haemorrhage_disability'] = True
            mni[person_id]['APH'] = True

            logger.debug('person %d is experiencing an antepartum haemorrhage in a health facility',
                        person_id)
            logger.info('%s|antepartum_haemorrhage|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

        random = self.sim.rng.random_sample(size=1)
        if random < mni[person_id]['risk_ip_sepsis']:
            df.at[person_id, 'la_sepsis'] = True
            df.at[person_id, 'la_sepsis_disability'] = True
            mni[person_id]['sepsis_ip'] = True

            logger.debug('person %d has developed maternal sepsis in a health facility',
                        person_id)
            logger.info('%s|maternal_sepsis|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})
            # TODO modify newborn risk of sepsis for septic women

        random = self.sim.rng.random_sample(size=1)
        if random < mni[person_id]['risk_ur']:
            mni[person_id]['UR'] = True
            df.at[person_id, 'la_uterine_rupture'] = True
            df.at[person_id, 'la_ur_disability'] = True

            logger.debug('person %d is experiencing a uterine rupture in a health facility',
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
                                                                tclose=self.sim.date + DateOffset(days=1)) # TODO: check all these offsets

        if mni[person_id]['sepsis_ip']:
            logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: scheduling immediate additional '
                        'treatment for maternal sepsis during labour for person %d', person_id)

            event = HSI_Labour_ReceivesCareForMaternalSepsis(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        if mni[person_id]['APH']:
            logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: scheduling immediate additional '
                        'treatment for antepartum haemorrhage during labour for person %d', person_id)

            event = HSI_Labour_ReceivesCareForMaternalHaemorrhage(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        if mni[person_id]['eclampsia_ip']:
            logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: scheduling immediate additional '
                        'treatment for eclampsia during labour for person %d', person_id)

            event = HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        if mni[person_id]['UR']:
            logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: scheduling immediate additional '
                        'treatment for uterine rupture during labour for person %d', person_id)

            event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1)
                                                                )

        if df.at[person_id, 'la_current_labour_successful_induction'] == 'failed_induction':
            logger.info('This is HSI_Labour_PresentsForSkilledAttendanceInLabour: scheduling a caesarean section'
                        ' for person %d', person_id)

            event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT

        if mni[person_id]['UR'] or mni[person_id]['eclampsia_ip'] or mni[person_id]['APH'] or\
            mni[person_id]['sepsis_ip'] or mni[person_id]['labour_is_currently_obstructed']:
            actual_appt_footprint['NormalDelivery'] = actual_appt_footprint['CompDelivery'] * 1

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_PresentsForSkilledAttendanceInLabour: did not run')
        # TODO: What else needs to go in here?
        pass


class HSI_Labour_ReceivesCareForObstructedLabour(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It manages the treatment of obstructed labour and referral in the instance of failed treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForObstructedLabour'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1  # TODO: to be discussed with TH/TC best appt footprint to use

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2  # check this?
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        # TODO: apply squeeze factor

        logger.info('This is HSI_Labour_ReceivesCareForObstructedLabour, management of obstructed labour for '
                    'person %d on date %s',
                    person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_obst_lab = pd.unique(consumables.loc[consumables[
                                                            'Intervention_Pkg'] ==
                                                        'Antibiotics for pPRoM',  # TODO: Obs labour package not working
                                                        'Intervention_Pkg_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code_obst_lab: 1}],
            'Item_Code': [],
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_obst_lab]:
            logger.debug('pkg_code_obst_lab is available, so use it.')
        else:
            logger.debug('pkg_code_obst_lab is not available, so can' 't use it.')
            # TODO: This will need to be equipment by equipment- i.e. if no vacuum then forceps if none then caesarean?
            # TODO: add equipment to lines on consumable chart?

        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

# =====================================  OBSTRUCTED LABOUR TREATMENT ==================================================

        # TODO: Differentiate between CPD and other?

        # For women in obstructed labour delivery is first attempted by vacuum, delivery mode is stored
        if mni[person_id]['labour_is_currently_obstructed']:
            treatment_effect = params['prob_deliver_ventouse']
            random = self.sim.rng.random_sample(size=1)
            if treatment_effect > random:
                # df.at[person_id, 'la_obstructed_labour'] = False
                mni[person_id]['labour_is_currently_obstructed'] = False
                mni[person_id]['mode_of_delivery'] = 'AVDV'
                # add here effect of antibiotics?
            else:
                # If the vacuum is unsuccessful we apply the probability of successful forceps delivery
                treatment_effect = params['prob_deliver_forceps']
                random = self.sim.rng.random_sample(size=1)
                if treatment_effect > random:
                    # df.at[person_id, 'la_obstructed_labour'] = False
                    mni[person_id]['labour_is_currently_obstructed'] = False
                    mni[person_id]['mode_of_delivery'] = 'AVDF'
                    # add here effect of antibiotitcs?

                # Finally if assisted vaginal delivery fails then CS is scheduled
                else:
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

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        #    actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_PresentsForInductionOfLabour: did not run')
        pass


class HSI_Labour_ReceivesCareForMaternalSepsis(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It manages the
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForMaternalSepsis'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2  # check this?
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        # TODO: Apply squeeze factor

        logger.info('This is HSI_Labour_ReceivesCareForMaternalSepsis, management of maternal sepsis for '
                    'person %d on date %s',
                    person_id, self.sim.date)

        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_sepsis = pd.unique(consumables.loc[consumables[
                                                        'Intervention_Pkg'] ==
                                                    'Maternal sepsis case management',
                                                    'Intervention_Pkg_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code_sepsis: 1}],
            'Item_Code': [],
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        # TODO: consider 1st line/2nd line and efficacy etc etc

        # Treatment can only be delivered if appropriate antibiotics are available, if they're not the woman isn't
        # treated
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_sepsis]:
            logger.debug('pkg_code_sepsis is available, so use it.')
            treatment_effect = params['prob_cure_antibiotics']
            random = self.sim.rng.random_sample(size=1)
            if treatment_effect > random:
                mni[person_id]['sepsis'] = False
        else:
            logger.debug('pkg_code_sepsis is not available, so can' 't use it.')

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        #    actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReceivesCareForMaternalSepsis: did not run')
        pass


class HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It manages the
        """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForHypertensiveDisorder'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1  # TODO: to be discussed with TH/TC best appt footprint to use

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2  # check this?
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.info('This is HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy, management of hypertensive '
                    'disorders of pregnancy for person %d on date %s',
                             person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_eclampsia = pd.unique(consumables.loc[consumables[
                                                               'Intervention_Pkg'] ==
                                                           'Management of eclampsia',
                                                       'Intervention_Pkg_Code'])[0]

        item_code_nf = pd.unique(
            consumables.loc[consumables['Items'] == 'nifedipine retard 20 mg_100_IDA',
                            'Item_Code']
        )[0]
        item_code_hz = pd.unique(
            consumables.loc[consumables['Items'] == 'Hydralazine hydrochloride 20mg/ml, 1ml_each_CMST',
                            'Item_Code']
        )[0]
        item_code_md = pd.unique(
            consumables.loc[consumables['Items'] == 'Methyldopa 250mg_1000_CMS',
                            'Item_Code']
        )[0]
        item_code_hs = pd.unique(
            consumables.loc[consumables['Items'] ==  "Ringer's lactate (Hartmann's solution), 500 ml_20_IDA",
                            'Item_Code']
        )[0]

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code_eclampsia: 1}],
            'Item_Code': [{item_code_nf: 2}, {item_code_hz:  2}, {item_code_md: 2}, {item_code_hs: 1}],
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        # TODO: Again determine how to reflect 1st/2nd line choice
#        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_obst_lab]:
#            logger.debug('pkg_code_obst_lab is available, so use it.')
#        else:
#            logger.debug('pkg_code_obst_lab is not available, so can' 't use it.')


# =======================================  HYPERTENSION TREATMENT ======================================================
        # tbc

# =======================================  SEVERE PRE-ECLAMPSIA TREATMENT ==============================================
        # tbc

# =======================================  ECLAMPSIA TREATMENT ========================================================

        # Here we apply the treatment algorithm, unsuccessful control of seizures leads to caesarean delivery
        if mni[person_id]['eclampsia_ip'] or mni[person_id]['eclampsia_pp']:
            treatment_effect = params['prob_cure_mgso4']
            random = self.sim.rng.random_sample()
            if treatment_effect > random:
                mni[person_id]['eclampsia'] = False
            else:
                random = self.sim.rng.random_sample()
                if treatment_effect > random:
                    mni[person_id]['eclampsia'] = False
                else:
                    treatment_effect = params['prob_cure_diazepam']
                    random = self.sim.rng.random_sample()
                    if treatment_effect > random:
                        mni[person_id]['eclampsia'] = False
                    else:
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
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        #    actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReceivesCareForMaternalSepsis: did not run')
        pass


class HSI_Labour_ReceivesCareForMaternalHaemorrhage(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    It manages the
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForMaternalHaemorrhage'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2  # check this?
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):
        # TODO: squeeze factor AND consider having APH, PPH and RPP as separate treatment events??

        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.info('This is HSI_Labour_ReceivesCareForMaternalHaemorrhage, management of obstructed labour for person '
                    '%d on date %s', person_id, self.sim.date)

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

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code_pph: 1}],
            'Item_Code': [{item_code_aph1: 2}, {item_code_aph2: 1}, {item_code_aph3: 1}, {item_code_aph4: 1},],
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        # TODO: Again determine how to use outcome of consumable request

# ===================================  ANTEPARTUM HAEMORRHAGE TREATMENT ===============================================
        # TODO: consider severity grading?

        # Here we determine the etiology of the bleed, which will determine treatment algorithm
        etiology = ['PA', 'PP']  # May need to move this to allow for risk factors?
        probabilities = [0.67, 0.33]  # DUMMY
        random_choice = self.sim.rng.choice(etiology, size=1, p=probabilities)

        mni[person_id]['source_aph'] = random_choice  # Storing as high chance of SB in severe placental abruption
        # TODO: Needs to be dependent on blood availability and establish how we're quantifying effect
        mni[person_id]['units_transfused'] = 2

        event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
        self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                            priority=0,
                                                            topen=self.sim.date,
                                                            tclose=self.sim.date + DateOffset(days=1)
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
                random = self.sim.rng.random_sample(size=1)
                if params['prob_cure_oxytocin'] > random:
                    mni[person_id]['PPH'] = False
                else:
                    random = self.sim.rng.random_sample(size=1)
                    if params['prob_cure_misoprostol'] > random:
                        mni[person_id]['PPH'] = False
                    else:
                        random = self.sim.rng.random_sample(size=1)
                        if params['prob_cure_uterine_massage'] > random:
                            mni[person_id]['PPH'] = False
                        else:
                            random = self.sim.rng.random_sample(size=1)
                            if params['prob_cure_uterine_tamponade'] > random:
                                mni[person_id]['PPH'] = False

            # Todo: consider the impact of oxy + miso + massage as ONE value, Discuss with expert

            # ===================TREATMENT CASCADE FOR RETAINED PRODUCTS/PLACENTA:====================================
            if mni[person_id]['source_pph'] == 'RPP':
                random = self.sim.rng.random_sample(size=1)
                if params['prob_cure_manual_removal'] > random:
                    mni[person_id]['PPH'] = False
                    # blood?

            # In the instance of uncontrolled bleeding a woman is referred on for surgical care
            if mni[person_id]['PPH']:
                event = HSI_Labour_ReferredForSurgicalCareInLabour(self.module, person_id=person_id)
                self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1)
                                                                    )

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        #    actual_appt_footprint['ConWithDCSA'] = actual_appt_footprint['ConWithDCSA'] * 2

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReceivesCareForMaternalHaemorrhage: did not run')
        pass

class HSI_Labour_ReceivesCareForPostpartumPeriod(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This event manages the Health System Interaction for women who receive post partum care following delivery
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

    # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReceivesCareForPostpartumPeriod'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['InpatientDays'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2 # check this?
        self.ALERT_OTHER_DISEASES = ['*']

    def apply(self, person_id, squeeze_factor):

        # TODO: Squeeze factor
        logger.info('This is HSI_Labour_ReceivesCareForPostpartumPeriod: Providing skilled attendance following birth '
                    'for person %d', person_id)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        pkg_code_am = pd.unique(consumables.loc[consumables[
                                                            'Intervention_Pkg'] ==
                                                        'Active management of the 3rd stage of labour',
                                                        'Intervention_Pkg_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code_am: 1}],
            'Item_Code': [],
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )

        df = self.sim.population.props  # shortcut to the dataframe
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

    #  =========================  ACTIVE MANAGEMENT OF THE THIRD STAGE  ===============================================

        # Here we apply a risk reduction of post partum bleeding following active management of the third stage of
        # labour (additional oxytocin, uterine massage and controlled cord traction)
        if outcome_of_request_for_consumables['Intervention_Package_Code'][pkg_code_am]:
            logger.debug('pkg_code_am is available, so use it.')
            adjusted_maternal_pph_risk = mni[person_id]['risk_pph'] * params['rr_pph_amtsl']
            mni[person_id]['risk_pph'] = adjusted_maternal_pph_risk
        else:
            logger.debug('pkg_code_am is not available, so can' 't use it.')
            logger.debug('woman %d did not receive active managment of the third stage of labour due to resource '
                         'constraints')
    # ===============================  POSTPARTUM COMPLICATIONS ========================================================

        # TODO: link eclampsia/sepsis diagnosis in SBA and PPC

        # As with the SkilledBirthAttendance HSI we recalcualte risk of complications in light of preventative
        # interventions
        random = self.sim.rng.random_sample(size=1)
        if random < mni[person_id]['risk_pp_eclampsia']:
            df.at[person_id, 'la_eclampsia'] = True
            df.at[person_id, 'la_eclampsia_disability'] = True
            mni[person_id]['eclampsia_pp'] = True

            logger.debug('person %d is experiencing eclampsia in a health facility following birth',
                        person_id)
            logger.info('%s|eclampsia|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

        if random < mni[person_id]['risk_pph']: # increased liklihood of PPH based on whats happened so far? (APH, CS)
            df.at[person_id, 'la_pph'] = True
            df.at[person_id, 'la_haemorrhage_disability'] = True
            mni[person_id]['PPH'] = True

            logger.debug('person %d is experiencing an postpartum haemorrhage in a health facility following birth',
                        person_id)
            logger.info('%s|postpartum_haemorrhage|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

        random = self.sim.rng.random_sample(size=1)
        if random < mni[person_id]['risk_pp_sepsis']:
            df.at[person_id, 'la_sepsis'] = True
            df.at[person_id, 'la_sepsis_disability'] = True
            mni[person_id]['sepsis_pp'] = True

            logger.debug('person %d has developed maternal sepsis in a health facility following delivery',
                        person_id)
            logger.info('%s|maternal_sepsis|%s', self.sim.date,
                        {'age': df.at[person_id, 'age_years'],
                         'person_id': person_id})

        # =============================  SCHEDULING ADDITIONAL TREATMENT ==============================================

        if mni[person_id]['sepsis_pp']:
            logger.info('This is HSI_Labour_ReceivesCareForPostpartumPeriod: scheduling immediate additional '
                        'treatment for maternal sepsis during the postpartum period for person %d', person_id)

            event = HSI_Labour_ReceivesCareForMaternalSepsis(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                        priority=0,
                                                        topen=self.sim.date,
                                                        tclose=self.sim.date + DateOffset(days=1)
                                                        )

        if mni[person_id]['PPH']:
            logger.info('This is HSI_Labour_ReceivesCareForPostpartumPeriod: scheduling immediate additional '
                        'treatment for antepartum haemorrhage during the postpartum period for person %d', person_id)

            event = HSI_Labour_ReceivesCareForMaternalHaemorrhage(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1)
                                                                    )

        if mni[person_id]['eclampsia_pp']:
            logger.info('This is HSI_Labour_ReceivesCareForPostpartumPeriod: scheduling immediate additional '
                        'treatment for eclampsia during the postpartum period for person %d', person_id)

            event = HSI_Labour_ReceivesCareForHypertensiveDisordersOfPregnancy(self.module, person_id=person_id)
            self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                    priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1)
                                                                    )

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT #  TODO: modify based on complications?

        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReceivesCareForPostpartumPeriod: did not run')
        pass

class HSI_Labour_ReferredForSurgicalCareInLabour(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event.
    This event manages the Health System Interaction for a woman who needs to be referred to undergo an emergency
    surgical management of complications arising in labour, in the postpartum period or for caesarean section
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module,Labour)

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Labour_ReferredForSurgicalCareInLabour'

        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()

    #   the_appt_footprint['MajorSurg'] = 1  # this appt could be used for uterine repair/pph management
        the_appt_footprint['Csection'] = 1

        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 2  # check this?
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        #  TODO: squeeze factor and consider splitting HSIs for different surgeries

        df = self.sim.population.props
        params = self.module.parameters
        mni = self.module.mother_and_newborn_info

        logger.info('This is HSI_Labour_ReferredForSurgicalCareInLabour,providing surgical care during labour and the'
                    ' postpartum period for person %d on date %s', person_id, self.sim.date)

        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        pkg_code_cs = pd.unique(consumables.loc[consumables[
                                                    'Intervention_Pkg'] ==
                                                'Cesearian Section with indication (with complication)',  # or without?
                                                'Intervention_Pkg_Code'])[0]

        consumables_needed = {
            'Intervention_Package_Code': [{pkg_code_cs: 1}],
            'Item_Code': [],
        }

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self, cons_req_as_footprint=consumables_needed
        )
        # TODO: consumables

        # pkg_code_uterine_repair
        # pkg_code_pph_surgery
        # +/- hysterectomy?

# ====================================== EMERGENCY CAESAREAN SECTION ==================================================

        if (mni[person_id]['UR']) or (mni[person_id]['APH']) or (mni[person_id]['eclampsia_ip']) or\
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

        # For women with UR we determine if the uterus can be repaired surgically
        if mni[person_id]['UR']:
            random = self.sim.rng.random_sample(size=1)
            if params['prob_cure_uterine_repair'] > random:
                # df.at[person_id, 'la_uterine_rupture'] = False
                mni[person_id]['UR'] = False

        # In the instance of failed surgical repair, the woman undergoes a hysterectomy
            else:
                random = self.sim.rng.random_sample(size=1)
                if params['prob_cure_hysterectomy'] > random:
                    # df.at[person_id, 'la_uterine_rupture'] = False
                    mni[person_id]['UR'] = False

# ================================== SURGERY FOR UNCONTROLLED POSTPARTUM HAEMORRHAGE ==================================

        # If a woman has be referred for surgery for uncontrolled post partum bleeding we use the treatment alogrith to
        # determine if her bleeding can be controlled surgically
        if mni[person_id]['PPH']:
            random = self.sim.rng.random_sample(size=1)
            if params['prob_cure_uterine_ligation'] > random:
                mni[person_id]['PPH'] = False
                print('Treatment success- this bleed has been stopped by uterine ligation')
            else:
                random = self.sim.rng.random_sample(size=1)
                if params['prob_cure_b_lynch'] > random:
                    mni[person_id]['PPH'] = False
                    print('Treatment success- this bleed has been stopped by b-lynch suturing')
                else:
                    random = self.sim.rng.random_sample(size=1)
                    # Todo: similarly consider bunching surgical interventions
                    if params['prob_cure_hysterectomy'] > random:
                        mni[person_id]['PPH'] = False
                        print('Treatment success- this bleed has been stopped by a hysterectomy')

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT #  TODO: modify based on complications?
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug('HSI_Labour_ReferredForSurgicalCareInLabour: did not run')
        pass


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
        # Disease Incidence
        # Intervention incidence

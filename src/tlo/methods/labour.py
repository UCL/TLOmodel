"""
Documentation:
First draft of Labour module (Natural history)
"""
import logging

import pandas as pd

import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

from tlo.methods import demography, eclampsia_treatment, sepsis_treatment, haemorrhage_treatment


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Labour (Module):

    """
    This module models labour and delivery and generates the properties for "complications" of delivery """

    PARAMETERS = {

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
            Types.REAL, 'case fatality rate for sepsis following delivery')
      }

    PROPERTIES = {

        'la_due_date': Property(Types.DATE, 'The predicted date of delivery for a newly pregnant woman'),
        'la_labour': Property(Types.CATEGORICAL, 'not in labour, Term labour, Early Preterm Labour, '
                                                 'Late Preterm Labour, Post term labour',
                              categories=['not_in_labour', 'term_labour', 'early_preterm_labour', 'late_preterm_labour'
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
        'la_delivery_mode': Property(Types.CATEGORICAL, 'Vaginal Delivery, Assisted Vaginal Delivery,'
                                                        'Emergency Caesarean Section, Elective Caesarean Section'
                                     , categories=['VD', 'AVD', 'EmCS', 'ElCS']),
        'la_died_in_labour': Property(Types.BOOL, 'whether the woman has died during this labour')
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters

        params['prob_pregnancy'] = 0.083  # Calculated from DHS 2010
        params['prob_miscarriage'] = 0.053 #0.189
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
        params['prob_an_ur'] = 0.001
        params['rr_an_ur_grand_multip'] = 7.57
        params['rr_an_ur_prevcs'] = 2.02
        params['rr_an_ur_ref_ol'] = 23.65  # REVIEW "obstructed but not referred"
        params['rr_an_eclampsia_30_34'] = 1.4
        params['rr_an_eclampsia_35'] = 1.95
        params['rr_an_eclampsia_nullip'] = 2.04
        params['rr_an_sepsis_anc_4'] = 0.5
        params['rr_an_aph_noedu'] = 1.72
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
        params['prob_pn_sepsis'] = 0.05
        params['prob_sa_pph'] = 0.12
        params['prob_sa_sepsis'] = 0.20
        params['cfr_pn_pph'] = 0.1
        params['cfr_pn_eclampsia'] = 0.184
        params['cfr_pn_sepsis'] = 0.33

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

        df['la_labour'] = 'not_in_labour'
        df['la_gestation_at_labour'] = 0
        df['la_miscarriage'] = 0
        df['la_miscarriage_date'] =pd.NaT
        df['la_still_birth_this_delivery'] = False
        df['la_partiy'] = 0
        df['la_previous_cs'] = 0
        df['la_previous_ptb'] = False
        df['la_due_date'] = pd.NaT
        df['la_obstructed_labour'] = False
        df['la_aph'] = False
        df['la_uterine_rupture'] = False
        df['la_sepsis'] = False
        df['la_eclampsia'] = False
        df['la_pph'] = False
        df['la_delivery_mode'] = 'not_in_labour'
        df['la_died_in_labour'] = False

    # -----------------------------------ASSIGN PREGNANCY AND DUE DATE AT BASELINE (DUMMY) ----------------------------

        # TODO: code in effect of contraception on likelihood of pregnancy at baseline (copy demography code?)

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

        # TODO: ensure women at baseline can go into preterm labour or be overdue

        # Get and hold all the women who are pregnant at baseline
        pregnant_idx = df.index[df.is_pregnant & df.is_alive]

        random_draw = pd.Series(rng.random_sample(size=len(pregnant_idx)),
                                index=df.index[df.is_pregnant & df.is_alive])

        # Randomly generate a number of weeks gestation between 1-39 for all pregnant women
        simdate = pd.Series(self.sim.date, index=pregnant_idx)
        dfx = pd.concat((simdate, random_draw), axis=1)
        dfx.columns = ['simdate', 'random_draw']
        dfx['gestational_age'] = (39 - 39 * dfx.random_draw)

        # Use this gestational age to calculate when the woman's baby was conceived
        dfx['la_conception_date'] = dfx['simdate'] - pd.to_timedelta(dfx['gestational_age'], unit='w')

        # Apply a due date of 9 months in the future from the date of conception for each woman
        dfx['due_date_mth'] = 39 - dfx['gestational_age']
        dfx['due_date'] = dfx['simdate'] + pd.to_timedelta(dfx['due_date_mth'], unit='w')
        df.loc[pregnant_idx, 'date_of_last_pregnancy'] = dfx.la_conception_date
        df.loc[pregnant_idx, 'la_due_date'] = dfx.due_date

        pregnant_baseline = df.index[(df.is_pregnant == True) & df.is_alive]

        for person in pregnant_baseline:
            scheduled_labour_date = df.at[person, 'la_due_date']
            labour = LabourEvent(self, individual_id=person, cause='Labour')
            birth = BirthEvent(self, mother_id=person)
            self.sim.schedule_event(labour, scheduled_labour_date)
            self.sim.schedule_event(birth, scheduled_labour_date + DateOffset(days=2))

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
        idx_prev_cs = dfx.index[dfx.baseline_cs1 < dfx.random_draw1]
        df.loc[idx_prev_cs, 'la_previous_cs'] = dfx.random_draw1

        dfx = pd.concat([baseline_cs2, random_draw2], axis=1)
        dfx.columns = ['baseline_cs2', 'random_draw2']
        idx_prev_cs1 = dfx.index[dfx.baseline_cs2 < dfx.random_draw2]
        df.loc[idx_prev_cs1, 'la_previous_cs'] = dfx.random_draw2

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

    def initialise_simulation(self, sim):

        event = LabourLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):

        df = self.sim.population.props

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
        df.at[child_id, 'la_delivery_mode'] = 'not_in_labour'
        df.at[child_id, 'la_died_in_labour'] = False

        # If a mothers labour has resulted in a late term still birth her child is still generated by the simulation
        # but is_alive is reset to false to allow for monitoring of still birth rates
        if df.at[mother_id, 'la_still_birth_this_delivery']:
            death = demography.InstantaneousDeath(self.sim.modules['Demography'], child_id,
                                                  cause='Intrapartum Stillbirth')
            self.sim.schedule_event(death, self.sim.date)

            # This property is then reset in case of future pregnancies/stillbirths
            df.loc[mother_id, 'la_still_birth_this_delivery'] = False


class MiscarriageEvent(Event, IndividualScopeEventMixin):
    """On conception event that applies a cumulative risk of early pregnancy loss to women """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # First we identify if this woman has any risk factors for early pregnancy loss
        if (df.at[individual_id, 'la_miscarriage'] >= 1) & (df.at[individual_id, 'age_years'] <= 30) & \
            (df.at[individual_id,'la_parity'] < 1 > 3):
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
            df.at[individual_id, 'is_pregnant'] = False
            df.at[individual_id, 'la_due_date'] = pd.NaT
            df.at[individual_id, 'la_miscarriage_date'] = self.sim.date
            df.at[individual_id, 'la_miscarriage'] = +1

            # For women who have had a miscarriage we schedule the post miscarriage event to determine if they
            # experience any complications
            self.sim.schedule_event(PostMiscarriageEvent(self.module, individual_id, cause='post miscarriage'),
                                    self.sim.date)
            # And for women who do not have a miscarriage we move them to labour scheduler to determine at what
            # gestation they will go into labour
        else:
            assert df.at[individual_id, 'la_due_date'] != pd.NaT
            self.sim.schedule_event(LabourScheduler(self.module, individual_id, cause='pregnancy'), self.sim.date)


class LabourScheduler (Event, IndividualScopeEventMixin):
    """This event determines when newly pregnant women will going to labour"""

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

        # Labour is then scheduled on the newly generated due date along with the birth event 2 days after
            self.sim.schedule_event(LabourEvent(self.module, individual_id, cause='labour'), due_date)
            self.sim.schedule_event(BirthEvent(self.module, individual_id), due_date + DateOffset(days=2))

        # If the woman will not go into preterm labour she is allocated a due date of between 37 and 44 weeks following
        # conception

        # Todo: Discuss with Tim C/Helen A benifit of post- term labour state and potential risk factors
        else:
            random = np.random.randint(37, 44, size=1)
            random = int(random)
            df.at[individual_id, 'la_due_date'] = df.at[individual_id, 'date_of_last_pregnancy'] +\
                                                  pd.Timedelta(random, unit='W')
            due_date= df.at[individual_id, 'la_due_date']
            self.sim.schedule_event(LabourEvent(self.module, individual_id, cause='labour'), due_date)
            self.sim.schedule_event(BirthEvent(self.module, individual_id), due_date + DateOffset(days=2))


class LabourEvent(Event, IndividualScopeEventMixin):

    """Moves a pregnant woman into labour/spontaneous abortion based on gestation distribution """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

# ===================================== LABOUR STATE  ==================================================================

        # Based on gestational age the woman in labour is allocated to either term, earl/late preterm or post term
        # labour
        gestation_date = df.at[individual_id, 'la_due_date'] - df.at[individual_id, 'date_of_last_pregnancy']
        gestation_weeks = gestation_date / np.timedelta64(1, 'W')
        gestation_weeks = int(gestation_weeks)

        if df.at[individual_id, 'is_pregnant']:
            df.at[individual_id, 'la_gestation_at_labour'] = gestation_weeks
            if df.at[individual_id, 'la_gestation_at_labour'] >= 37 and\
                df.at[individual_id, 'la_gestation_at_labour'] < 42:
                df.at[individual_id, 'la_labour'] = "term_labour"

            elif df.at[individual_id, 'la_gestation_at_labour'] >= 24 and\
                df.at[individual_id, 'la_gestation_at_labour'] < 34:
                df.at[individual_id, 'la_labour'] = "early_preterm_labour"
                df.at[individual_id, 'la_previous_ptb'] = True

            elif df.at[individual_id, 'la_gestation_at_labour'] >= 34 and\
                df.at[individual_id, 'la_gestation_at_labour'] < 37:
                df.at[individual_id, 'la_labour'] = "late_preterm_labour"
                df.at[individual_id, 'la_previous_ptb'] = True

            elif df.at[individual_id, 'la_gestation_at_labour'] > 41:
                df.at[individual_id, 'la_labour'] = "post_term_labour"

# ================================== COMPLICATIONS DURING LABOUR ====================================================
        # Todo: Discuss with Helen allot impact of gestation on complications of the mother
        # Todo: Consider where to apply  labour care bundle (clean delivery, monitoring, AMTSL etc etc)

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

        riskfactors = rf1*rf2*rf3
        eff_prob_ptl= riskfactors * params['prob_pl_ol']
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_ptl:
            df.at[individual_id,'la_obstructed_labour'] = True

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

        if (df.at[individual_id, 'la_parity'] >= 1)  &(df.at[individual_id, 'age_years'] >= 30) & \
                (df.at[individual_id, 'age_years'] <= 34):
            rf3 = params['rr_an_eclampsia_30_34']
        else:
            rf3 = 1

        riskfactors = rf1 * rf2 * rf3
        eff_prob_eclampsia = riskfactors * params['prob_an_eclampsia']
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_eclampsia:
            df.at[individual_id, 'la_eclampsia'] = True

# ============================================ APH ====================================================================

        if df.at[individual_id, 'li_ed_lev'] == 0:
                rf1 = params['rr_an_aph_noedu']

        else:
            rf1 = 1

        riskfactors = rf1
        eff_prob_aph = riskfactors * params['prob_an_aph']
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_aph:
            df.at[individual_id, 'la_aph'] = True

# ==========================================  SEPSIS =================================================================

        # Todo: include risk factors

        #  if df.at[individual_id, 'li_ed_lev'] == 0:
        #          rf1 = params['rr_an_aph_noedu']

        # else:
            rf1 = 1

        riskfactors = 1  # rf1
        eff_prob_sepsis = riskfactors * params['prob_an_sepsis']
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_sepsis:
            df.at[individual_id, 'la_sepsis'] = True

# ====================================== UTERINE RUPTURE  =============================================================

        if (df.at[individual_id, 'la_parity'] >= 5) & (df.at[individual_id, 'la_previous_cs'] == 0) & \
            ~df.at[individual_id, 'la_obstructed_labour']:
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

        eff_prob_aph = riskfactors * params['prob_an_ur']
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_aph:
            df.at[individual_id, 'la_uterine_rupture'] = True

        # Todo: CONSIDER IF WE NEED TO MOVE THROUGH PRETERM/POST TERM LABOUR COMPLICATIONS OR IF THEY WILL JUST BE A
        #  RISK FACTOR/PROTECTION FOR THE COMPLICATIONS

# !~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~ DUMMY SCHEDULING !~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~

        # DUMMY CODE TO SWITCH ON/OFF SCHEDULING OF INTERVENTIONS PRIOR TO INTEGRATION WITH THE HEALTH SYSTEM


#        if df.at[individual_id,'la_eclampsia']:
#           self.sim.schedule_event(eclampsia_treatment.EclampsiaTreatmentEvent(self.sim.modules['EclampsiaTreatment'],
#                                                                                individual_id, cause='eclampsia'),
#                                    self.sim.date)

#        if df.at[individual_id,'la_sepsis']:
#            self.sim.schedule_event(sepsis_treatment.SepsisTreatmentEvent(self.sim.modules['SepsisTreatment'],
#                                                                                individual_id, cause='Sepsis'),
#                                    self.sim.date)

#        if df.at[individual_id, 'la_aph']:
#           self.sim.schedule_event(haemorrhage_treatment.AntepartumHaemorrhageTreatmentEvent
#                                    (self.sim.modules['HaemorrhageTreatment'], individual_id,
#                                     cause='antepartum haemorrhage'), self.sim.date)


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

        riskfactors = 1  # rf1
        eff_prob_pph = riskfactors * params['prob_pn_pph']
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_pph:
            df.at[individual_id, 'la_pph'] = True

# ============================================= RISK OF SEPSIS =========================================================

        riskfactors = 1  # rf1
        eff_prob_pn_sepsis = riskfactors * params['prob_pn_sepsis']
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_pn_sepsis:
            df.at[individual_id, 'la_sepsis'] = True

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
        random = self.sim.rng.random_sample(size=1)
        if random < eff_prob_eclampsia:
            df.at[individual_id, 'la_eclampsia'] = True

        if df.at[individual_id, 'la_pph']:
           self.sim.schedule_event(haemorrhage_treatment.PostpartumHaemorrhageTreatmentEvent
                                    (self.sim.modules['HaemorrhageTreatment'], individual_id,
                                     cause='postpartum haemorrhage'), self.sim.date)

#  =============================================== RESET LABOUR STATUS =================================================

        # Following the end of this event we reset properties related to labour to ensure future labours run correctly
        # (commented out so can view effects at this point)
    #    if df.at[individual_id, 'is_alive']:
    #        df.at[individual_id, 'la_labour'] = "not_in_labour"
    #        df.at[individual_id, 'la_due_date'] = pd.NaT
    #        df.at[individual_id, 'la_gestation_at_labour'] = 0


class PostMiscarriageEvent(Event, IndividualScopeEventMixin):

    """applies probability of postpartum complications to women who have just experience a miscarriage """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # TODO: consider stage of pregnancy loss and its impact on likelihood of complications i.e retained product

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
                        {
                            'age': df.at[individual_id, 'age_years'],
                            'cause': self.cause,
                            'person_id': individual_id
                        })


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

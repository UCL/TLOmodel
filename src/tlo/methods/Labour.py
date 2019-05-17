"""
First draft of Labour module (Natural history)
Documentation:
"""
import logging

import pandas as pd

import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

from tlo.methods import demography


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
        'rr_PL_OL_baby>3.5g': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her baby weighs more than 3.5'
                        'kilograms'),
        'rr_PL_OL_baby<1.5g': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her baby weighs less than 1.5'
                        'kilograms'),
        'rr_PL_OL_bmi<18': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her BMI is less than 18'),
        'rr_PL_OL_bmi>25': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if her BMI is greater than 25'),
        'prob_ptl': Parameter (
            Types.REAL, 'probability of a woman entering labour at <37 weeks gestation'),
        'rr_ptl_persistent_malaria': Parameter (
            Types.REAL, 'relative risk of a woman with presistant malaria entering preterm labour '
                        '(peripheral slide positive at booking and at 28–32 weeks)'),
        'prob_an_eclampsia_ptb': Parameter(
            Types.REAL, 'probability of eclampsia during preterm labour'),
        'prob_an_aph_ptb': Parameter(
            Types.REAL, 'probability of an antepartum haemorrhage in preterm labour'),
        'prob_an_sepsis_ptb': Parameter(
            Types.REAL, 'probability of sepsis in preterm labour'),
        'prob_an_eclampsia_utl': Parameter(
            Types.REAL, 'probability of eclampsia in  unobstructed term labours for women under 30 who have '
                        'previously delivered a child'),
        'rr_an_eclampsia_30_34': Parameter(
            Types.REAL, 'relative risk of eclampsia for women ages between 30 and 34'),
        'rr_an_eclampsia_35': Parameter(
            Types.REAL, 'relative risk of eclampsia for women ages older than 35'),
        'rr_an_eclampsia_nullip': Parameter(
            Types.REAL, 'relative risk of eclampsia for women who have not previously delivered a child'),
        'prob_an_sepsis_utl': Parameter(
            Types.REAL, 'probability of sepsis in  unobstructed term labour'),
        'rr_an_sepsis_anc_4': Parameter(
            Types.REAL, 'relative risk of sepsis for women who have attended greater than 4 ANC visits'),
        'prob_an_aph_utl': Parameter(
            Types.REAL, 'probability of antepartum haemorrhage in  unobstructed term labour'),
        'rr_an_aph_noedu': Parameter(
            Types.REAL, 'relative risk of antepartum haemorrhage for women with education of primary level or lower'),
        'prob_an_eclampsia_pl_ol': Parameter(
            Types.REAL, 'probability of eclampsia in prolonged or obstructed labour'),
        'prob_an_sepsis_pl_ol': Parameter(
            Types.REAL, 'probability of sepsis in prolonged or obstructed labour'),
        'prob_an_aph_pl_ol': Parameter(
            Types.REAL, 'probability of antepartum haemorrhage in prolonged or obstructed labour'),
        'prob_an_ur_pl_ol': Parameter(
            Types.REAL, 'probability of uterine rupture in prolonged or obstructed labour'),
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
        'prob_pn_eclampsia_ptb': Parameter(
            Types.REAL, 'probability of eclampsia following delivery for women who were in spotaneous unobstructed '
                        'labour'),
        'prob_pn_pph_ptb': Parameter(
            Types.REAL, 'probability of an postpartum haemorrhage for women who were in preterm labour'),
        'prob_pn_sepsis_ptb': Parameter(
            Types.REAL, 'probability of sepsis following delivery for women in preterm labour'),
        'prob_pn_eclampsia_utl': Parameter(
            Types.REAL, 'probability of eclampsia following delivery for women who were in spotaneous unobstructed '
                        'labour'),
        'prob_pn_sepsis_utl': Parameter(
            Types.REAL, 'probability of sepsis following delivery for women who were in unobstructed term labour'),
        'prob_pn_pph_utl': Parameter(
            Types.REAL, 'probability of postpartum haemorrhage for women who were in  unobstructed term labour'),
        'prob_pn_eclampsia_pl_ol': Parameter(
            Types.REAL, 'probability of eclampsia following delivery for women who were in prolonged or obstructed'
                        ' labour'),
        'prob_pn_sepsis_pl_ol': Parameter(
            Types.REAL, 'probability of sepsis following delivery for women who were  in prolonged or obstructed '
                        'labours'),
        'prob_pn_pph_pl_ol': Parameter(
            Types.REAL, 'relative risk of a woman going into pre term labour if she has had a preterm birth for any '
                        'prior deliveries'),
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
            Types.REAL, 'case fatality rate for sepsis following delivery s')

      }

    PROPERTIES = {

        'la_labour': Property(Types.CATEGORICAL, 'not in labour, unobstructed term labour, prolonged or '
                              'obstructed labour, Preterm Labour, Post term labour',
                              categories=['not_in_labour', 'unobstructed_term_labour',
                                          'prolonged_or_obstructed_labour', 'preterm_labour', 'post_term_labour']),
        'la_still_birth_this_delivery': Property(Types.BOOL,'whether this womans most recent pregnancy has ended in a '
                                                            'stillbirth'),
        'la_miscarriage': Property(Types.INT, 'the number of miscarriages a woman has experienced'),
        'la_miscarriage_date': Property(Types.DATE, 'the date this woman has last experienced spontaneous miscarriage'),
        'la_parity': Property(Types.INT, 'total number of previous deliveries'),
        'la_previous_cs': Property(Types.INT, 'number of previous deliveries by caesarean section'),
        'la_previous_ptb': Property(Types.BOOL, 'whether the woman has had a previous preterm delivery for any of her'
                                                'previous deliveries'),
        'la_aph': Property(Types.BOOL, 'whether the woman has experienced an antepartum haemorrhage in this delivery'),
        'la_uterine_rupture': Property(Types.BOOL, 'whether the woman has experienced uterine rupture in this delivery'),
        'la_sepsis': Property(Types.BOOL, 'whether the woman has developed sepsis associated with in this delivery'),
        'la_eclampsia': Property(Types.BOOL, 'whether the woman has experienced an eclamptic seizure in this delivery'),
        'la_pph': Property(Types.BOOL, 'whether the woman has experienced an postpartum haemorrhage in this delivery'),
        'la_died_in_labour': Property(Types.BOOL, 'whether the woman has died during this labour')
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters

        params['prob_pregnancy'] = 0.03  # DUMMY PREGNANCY GENERATOR
        params['prob_miscarriage'] = 0.186
        params['rr_miscarriage_prevmiscarriage'] = 2.23
        params['rr_miscarriage_35'] = 4.02
        params['rr_miscarriage_3134'] = 2.13
        params['rr_miscarriage_grav4'] = 1.63
        params['prob_pl_ol'] = 0.06
        params['rr_PL_OL_nuliparity'] = 1.8
        params['rr_PL_OL_parity_3'] = 0.8
        params['rr_PL_OL_age_less20'] = 1.3
        params['prob_ptl'] = 0.18
        params['prob_an_eclampsia_ptb'] = 0.02
        params['prob_an_aph_ptb'] = 0.03
        params['prob_an_sepsis_ptb'] = 0.15
        params['prob_an_eclampsia_utl'] = 0.10
        params['rr_an_eclampsia_30_34'] = 1.4
        params['rr_an_eclampsia_35'] = 1.95
        params['rr_an_eclampsia_nullip'] = 2.04
        params['prob_an_sepsis_utl'] = 0.15
        params['rr_an_sepsis_anc_4'] = 0.5
        params['prob_an_aph_utl'] = 0.05
        params['rr_an_aph_noedu'] = 1.72
        params['prob_an_eclampsia_pl_ol'] = 0.13
        params['prob_an_sepsis_pl_ol'] = 0.25
        params['prob_an_aph_pl_ol'] = 0.25
        params['prob_an_ur_pl_ol'] = 0.06
        params['cfr_aph'] = 0.05
        params['cfr_eclampsia'] = 0.03
        params['cfr_sepsis'] = 0.05
        params['cfr_uterine_rupture'] = 0.045
        params['prob_still_birth_aph'] = 0.35
        params['prob_still_birth_aph_md'] = 0.90
        params['prob_still_birth_sepsis'] = 0.25
        params['prob_still_birth_sepsis_md'] = 0.90
        params['prob_still_birth_ur'] = 0.75
        params['prob_still_birth_ur_md'] = 0.90
        params['prob_still_birth_eclampsia'] = 0.55
        params['prob_still_birth_eclampsia_md'] = 0.90
        params['prob_pn_eclampsia_ptb'] = 0.01
        params['prob_pn_pph_ptb'] = 0.01
        params['prob_pn_sepsis_ptb'] = 0.05
        params['prob_pn_eclampsia_utl'] = 0.05
        params['prob_pn_sepsis_utl'] = 0.02
        params['prob_pn_pph_utl'] = 0.05
        params['prob_pn_eclampsia_pl_ol'] = 0.07
        params['prob_pn_sepsis_pl_ol'] = 0.18
        params['prob_pn_pph_pl_ol'] = 0.25
        params['prob_sa_pph'] = 0.12
        params['prob_sa_sepsis'] = 0.20
        params['cfr_pn_pph'] = 0.03
        params['cfr_pn_eclampsia'] = 0.05
        params['cfr_pn_sepsis'] = 0.04

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self
        rng = m.rng

    # ----------------------------------------- DEFAULTS----------------------------------------------------------------

        df['la_labour'] = 'not_in_labour'
        df['la_miscarriage'] = 0
        df['la_miscarriage_date'] =pd.NaT
        df['la_still_birth_this_delivery'] = False
        df['la_partiy'] = 0
        df['la_previous_cs'] = 0
        df['la_previous_ptb'] = False
        df['la_gestation'] = 0
        df['la_due_date'] = pd.NaT
        df['la_aph'] = False
        df['la_uterine_rupture'] = False
        df['la_sepsis'] = False
        df['la_eclampsia'] = False
        df['la_pph'] = False
        df['la_died_in_labour'] = False
    # -----------------------------------ASSIGN PREGNANCY AND DUE DATE AT BASELINE (DUMMY) ----------------------------

        # TODO: code in effect of contraception on likelihood of pregnancy at baseline (copy demography code?)

        women_idx = df.index[(df.age_years >= 15) & (df.age_years <= 49) & df.is_alive & (df.sex == 'F')]

        eff_prob_preg = pd.Series(m.prob_pregnancy, index=women_idx)

        random_draw = pd.Series(rng.random_sample(size=len(women_idx)),
                                index=df.index[(df.age_years >= 15) & (df.age_years <= 49) & df.is_alive
                                               & (df.sex == 'F')])

        dfx = pd.concat([eff_prob_preg, random_draw], axis=1)
        dfx.columns = ['eff_prob_pregnancy', 'random_draw']
        idx_pregnant = dfx.index[dfx.eff_prob_pregnancy > dfx.random_draw]
        df.loc[idx_pregnant, 'is_pregnant'] = True

# ---------------------------------    GESTATION AT BASELINE  ---------------------------------------------------------

        # Assigns a date of last pregnancy and due date for women who are pregnant at baseline

        # TODO: ensure women at baseline can go into preterm labour or be overdue

        pregnant_idx = df.index[df.is_pregnant & df.is_alive]

        random_draw = pd.Series(rng.random_sample(size=len(pregnant_idx)),
                                index=df.index[df.is_pregnant & df.is_alive])

        simdate = pd.Series(self.sim.date, index=pregnant_idx)
        dfx = pd.concat((simdate, random_draw), axis=1)
        dfx.columns = ['simdate', 'random_draw']
        dfx['gestational_age'] = (39 - 39 * dfx.random_draw)
        dfx['la_conception_date'] = dfx['simdate'] - pd.to_timedelta(dfx['gestational_age'], unit='w')
        dfx['due_date_mth'] = 39 - dfx['gestational_age']
        dfx['due_date'] = dfx['simdate'] + pd.to_timedelta(dfx['due_date_mth'], unit='w')

        df.loc[pregnant_idx, 'date_of_last_pregnancy'] = dfx.la_conception_date
        df.loc[pregnant_idx, 'due_date'] = dfx.due_date

#  ----------------------------ASSIGNING PARITY AT BASELINE (DUMMY)-----------------------------------------------------

        women_parity_1524_idx = df.index[(df.age_years >= 15) & (df.age_years <= 24) & (df.is_alive == True)
                                    & (df.sex == 'F')]

        baseline_p = pd.Series(0, index=women_parity_1524_idx)

        random_draw2 = pd.Series(self.rng.choice(range(0, 5), p=[0.40, 0.35, 0.15, 0.06, 0.04],
                                                 size=len(women_parity_1524_idx)),
                                 index=df.index[(df.age_years >= 15) & (df.age_years <= 24)
                                                & df.is_alive & (df.sex == 'F')])

        dfi = pd.concat([baseline_p, random_draw2], axis=1)
        dfi.columns = ['baseline_p', 'random_draw2']
        idx_parity = dfi.index[dfi.baseline_p < dfi.random_draw2]
        df.loc[idx_parity, 'la_parity'] = random_draw2

        women_parity_2540_idx = df.index[(df.age_years >= 25) & (df.age_years <= 40) & (df.is_alive == True)
                                         & (df.sex == 'F')]

        baseline_p = pd.Series(0, index=women_parity_2540_idx)

        random_draw = pd.Series(self.rng.choice(range(0, 6), p=[0.05, 0.15, 0.30, 0.20, 0.2, 0.1],
                                                size=len(women_parity_2540_idx)), index=df.index[(df.age_years >= 25) &
                                                                                                 (df.age_years <= 40)
                                                                                                 & (df.is_alive == True)
                                                                                                 & (df.sex == 'F')])

        dfi = pd.concat([baseline_p, random_draw], axis=1)
        dfi.columns = ['baseline_p', 'random_draw']
        idx_parity = dfi.index[dfi.baseline_p < dfi.random_draw]
        df.loc[idx_parity, 'la_parity'] = random_draw

        women_parity_4149_idx = df.index[(df.age_years >= 41) & (df.age_years <= 49) & (df.is_alive == True)
                                         & (df.sex == 'F')]

        baseline_p = pd.Series(0, index=women_parity_4149_idx)

        random_draw = pd.Series(self.rng.choice(range(0, 7), p=[0.05, 0.10, 0.25, 0.30, 0.25, 0.03, 0.02],
                                                size=len(women_parity_4149_idx)), index=df.index[(df.age_years >= 41)
                                                                                                 & (df.age_years <= 49)
                                                                                                 & (df.is_alive == True)
                                                                                                 & (df.sex == 'F')])

        dfi = pd.concat([baseline_p, random_draw], axis=1)
        dfi.columns = ['baseline_p', 'random_draw']
        idx_parity = dfi.index[dfi.baseline_p < dfi.random_draw]
        df.loc[idx_parity, 'la_parity'] = random_draw

#   ------------------------------ ASSIGN PREVIOUS CS AT BASELINE -----------------------------------------------

        women_para1_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') & (df.la_parity == 1)]
        women_para2_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') & (df.la_parity >= 2)]

        baseline_cs1 = pd.Series(df.la_previous_cs, index=women_para1_idx)
        baseline_cs2 = pd.Series(df.la_previous_cs, index=women_para2_idx)

        random_draw1 = pd.Series(self.rng.choice(range(0, 2), p=[0.91, 0.09], size=len(women_para1_idx)),
                               index=df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') &
                                              (df.la_parity == 1)])

        random_draw2 = pd.Series(self.rng.choice(range(0, 3), p=[0.90, 0.07, 0.03], size=len(women_para2_idx)),
                                 index=df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') &
                                                (df.la_parity >= 2)])

        dfx = pd.concat([baseline_cs1, random_draw1], axis=1)
        dfx.columns = ['baseline_cs', 'random_draw']
        idx_prev_cs = dfx.index[dfx.baseline_cs < dfx.random_draw]
        df.loc[idx_prev_cs, 'la_previous_cs'] = random_draw1

        dfx = pd.concat([baseline_cs2, random_draw2], axis=1)
        dfx.columns = ['baseline_cs', 'random_draw']
        idx_prev_cs = dfx.index[dfx.baseline_cs < dfx.random_draw]
        df.loc[idx_prev_cs, 'la_previous_cs'] = random_draw2

        # ------------------------------ ASSIGN PREVIOUS PTB AT BASELINE ----------------------------------------------

        # TODO: apply risk factors for preterm birth at baseline

        women_para1_nocs_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') & (df.la_parity == 1) &
                                        (df.la_previous_cs ==0)]
        women_para2_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') & (df.la_parity >= 2)]

        baseline_ptb = pd.Series(m.prob_ptl, index=women_para1_nocs_idx)
        baseline_ptb_p2 = pd.Series(m.prob_ptl, index=women_para2_idx)

        random_draw = pd.Series(rng.random_sample(size=len(women_para1_nocs_idx)),
                                index=df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') &
                                               (df.la_parity == 1) & (df.la_previous_cs == 0)])
        random_draw2 = pd.Series(rng.random_sample(size=len(women_para2_idx)),
                                index=df.index[ (df.age_years >= 15) & df.is_alive & (df.sex == 'F') &
                                                (df.la_parity >= 2)])

        dfx = pd.concat([baseline_ptb, random_draw], axis=1)
        dfx.columns = ['baseline_ptb', 'random_draw']
        idx_prev_ptb = dfx.index[dfx.baseline_ptb > dfx.random_draw]
        df.loc[idx_prev_ptb, 'la_previous_ptb'] = True

        dfx = pd.concat([baseline_ptb_p2, random_draw2], axis=1)
        dfx.columns = ['baseline_ptb_p2', 'random_draw2']
        idx_prev_ptb = dfx.index[dfx.baseline_ptb_p2 > dfx.random_draw2]
        df.loc[idx_prev_ptb, 'la_previous_ptb'] = True

    def baseline_labour_scheduler(self, population):
        """Schedules labour for women who are pregnant at baseline"""

        df = population.props

        pregnant_baseline = df.index[(df.is_pregnant == True) & df.is_alive]
        for person in pregnant_baseline:
            scheduled_labour_date = df.at[person, 'due_date']
            labour = LabourEvent(self, individual_id=person, cause='labour')
            self.sim.schedule_event(labour, scheduled_labour_date)

    def initialise_simulation(self, sim):

        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        event = MiscarriagePoll(self)
        sim.schedule_event(MiscarriagePoll(self), sim.date + DateOffset(months=2))

        event = LabourLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

        self.baseline_labour_scheduler(sim.population)

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        df.at[child_id, 'la_labour'] = 'not_in_labour'
        df.at[child_id, 'la_miscarriage'] = 0
        df.at[child_id, 'la_miscarriage_date'] = pd.NaT
        df.at[child_id, 'la_still_birth_this_delivery'] = False
        df.at[child_id, 'la_parity'] = 0
        df.at[child_id, 'la_previous_cs'] = 0
        df.at[child_id, 'la_previous_ptb'] = False
        df.at[child_id, 'la_aph'] = False
        df.at[child_id, 'la_uterine_rupture'] = False
        df.at[child_id, 'la_sepsis'] = False
        df.at[child_id, 'la_eclampsia'] = False
        df.at[child_id, 'la_pph'] = False
        df.at[child_id, 'la_died_in_labour'] = False


class MiscarriagePoll(RegularEvent, PopulationScopeEventMixin):
    """Poll of all pregnant women and applies likelihood of miscarriage (loss of pregnancy before 28 weeks gestation)
    dependent on risk factors """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = population.props
        params = self.module.parameters
        m = self

        # Polling all pregnant women who have yet to reach 28 weeks gestation

        pregnant_women = df.index[((df.is_pregnant == True) & df.is_alive & (df.la_labour == 'not_in_labour'))]

        conception = pd.Series(df.date_of_last_pregnancy, index=pregnant_women)
        simdate = pd.Series(self.sim.date, index=pregnant_women)

        dfx = pd.concat([conception, simdate], axis=1)
        dfx['gestweeks'] = simdate-conception
        dfx['gestweeks'] = dfx['gestweeks'] / np.timedelta64(1,'W')
        poss_miscarriage = dfx.index[dfx.gestweeks <= 28]
        prob_mc = params['prob_miscarriage']

        # Apply cumulative probability of miscarriage in the first 28 weeks

        eff_prob_miscarriage = pd.Series(prob_mc, index=poss_miscarriage)

        eff_prob_miscarriage.loc[(df.is_alive & (df.is_pregnant == True) & (df.la_labour == 'not_in_labour') &
                                          (df.la_miscarriage >= 1) & (df.la_parity + df.la_miscarriage <= 3) &
                                          (df.age_years <= 30))] *= params['rr_miscarriage_prevmiscarriage']

        eff_prob_miscarriage.loc[(df.is_alive & (df.is_pregnant == True) & (df.la_labour == 'not_in_labour') &
                                          (df.la_miscarriage == 0) & (df.age_years >= 31) & (df.age_years <= 34))]\
            *= params['rr_miscarriage_3134']
        eff_prob_miscarriage.loc[(df.is_alive & (df.is_pregnant == True) & (df.la_labour == 'not_in_labour') &
                                            (df.la_miscarriage == 0) & (df.age_years >= 35))]\
            *= params['rr_miscarriage_35']
        eff_prob_miscarriage.loc[(df.is_alive & (df.is_pregnant == True) & (df.la_labour == 'not_in_labour') &
                                           (df.la_miscarriage == 0) + (df.la_parity >= 4) & (df.age_years <= 30))]\
            *= params['rr_miscarriage_grav4']

        random_draw = pd.Series(self.sim.rng.random_sample(size=len(poss_miscarriage)),
                                index=dfx.index[dfx.gestweeks <= 28])

        dfx = pd.concat([eff_prob_miscarriage, random_draw], axis=1)
        dfx.columns = ['eff_prob_miscarriage', 'random_draw']
        idx_miscarriage = dfx.index[dfx.eff_prob_miscarriage > dfx.random_draw]

        # For woman who experience miscarriage, pregnancy status is reset, number of miscarriages recorded including
        # date and woman is scheduled to move to postpartum event to determine if she experiences complications

        df.loc[idx_miscarriage, 'is_pregnant'] = False
        df.loc[idx_miscarriage, 'la_miscarriage'] += 1
        df.loc[idx_miscarriage, 'due_date'] = pd.NaT
        df.loc[idx_miscarriage, 'la_miscarriage_date'] = self.sim.date

        new_miscarriage_idx = df.index[(df.la_miscarriage_date == self.sim.date)]
        for mother_id in new_miscarriage_idx:
            self.sim.schedule_event(PostpartumLabourEvent(self.module, mother_id,
                                                             cause='post miscarriage'), self.sim.date)


class LabourEvent(Event, IndividualScopeEventMixin):

    """Moves a pregnant woman into labour/spontaneous abortion based on gestation distribution """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

# ------------------------------------LABOUR STATE---------------------------------------------------------------------

#       Application of risk factors for women to enter prolonged/obstructed labour

        if df.at[individual_id, 'la_parity'] == 0:
            nullip_risk = params['rr_PL_OL_nuliparity']
        else:
            nullip_risk = 0
        if df.at[individual_id, 'la_parity'] >= 3:
            multip_risk = params['rr_PL_OL_parity_3']
        else:
            multip_risk = 0
        if df.at[individual_id, 'age_years'] <= 20:
            young_risk = params['rr_PL_OL_age_less20']
        else:
            young_risk = 0

        risk_factors = (nullip_risk + multip_risk + young_risk)

        if risk_factors == 0:
            risk_of_pl_ol = params['prob_pl_ol']
        else:
            risk_of_pl_ol = risk_factors * params['prob_pl_ol']

        gestation_date = df.at[individual_id, 'due_date'] - (df.at[individual_id, 'date_of_last_pregnancy'])
        gestation_weeks = gestation_date / np.timedelta64(1, 'W')

        if gestation_weeks > 37:
            random_draw = (self.sim.rng.random_sample(size=1))
            if random_draw < risk_of_pl_ol:
                df.at[individual_id, 'la_labour'] = "prolonged_or_obstructed_labour"
            else:
                df.at[individual_id, 'la_labour'] = 'unobstructed_term_labour'

            # TODO: apply risk factors for preterm birth

        elif gestation_weeks > 28 < 37:
            df.at[individual_id, 'la_labour'] = "preterm_labour"
            df.at[individual_id, 'la_previous_ptb'] = True

        if gestation_weeks >42:
            df.at[individual_id,'la_labour'] = "post_term_labour"

    # ======================= COMPLICATIONS FOR UNOBSTRUCTED TERM LABOUR ===============================================

    # TODO: N.b only allow women who have pre-eclampsia to have an eclamptic fit?
    # Applying risk factors for complications

        risk_sepsis = params['prob_an_sepsis_utl']

        if df.at[individual_id, 'la_parity'] == 0:
            nullip_risk = params['rr_an_eclampsia_nullip']
        else:
            nullip_risk = 0
        if df.at[individual_id, 'age_years'] >= 35:
            over35_risk = params['rr_an_eclampsia_35']
        else:
            over35_risk = 0
        if df.at[individual_id, 'age_years'] >= 30 <= 34:
            over30_risk = params['rr_an_eclampsia_30_34']
        else:
            over30_risk = 0

        risk_factors = (nullip_risk + over35_risk + over30_risk)

        if risk_factors == 0:
            risk_eclampsia = params['prob_an_eclampsia_utl']
        else:
            risk_eclampsia = risk_factors * params['prob_an_eclampsia_utl']

#        if df.at[individual_id, 'anc_4'] == True: (commented out- no ANC property generated yet)
#            risk_sepsis =  params['rr_an_sepsis_anc_4']* params['prob_an_sepsis_utl']

        if df.at[individual_id, 'age_years'] >= 35:
            risk_aph = params['rr_an_aph_noedu'] * params['prob_an_aph_utl']
        else:
            risk_aph = params['prob_an_aph_utl']

        if df.at[individual_id, 'la_labour'] == 'unobstructed_term_labour':
            p_comp_utl = pd.DataFrame(data=[(risk_eclampsia, self.sim.rng.random_sample(size=1), False,
                                            params['cfr_eclampsia'], self.sim.rng.random_sample(size=1), False),
                                            (risk_sepsis, self.sim.rng.random_sample(size=1), False,
                                            params['cfr_sepsis'], self.sim.rng.random_sample(size=1), False),
                                         (risk_aph, self.sim.rng.random_sample(size=1), False,
                                          params['cfr_aph'], self.sim.rng.random_sample(size=1), False)],
                                   columns=['complication_prob', 'random_draw','complication', 'prob_cfr',
                                            'random_draw2', 'maternal_death'], index=('eclampsia', 'sepsis', 'aph'))

            # Determine is a woman who is in normal labour will experience a complication during delivery

            if p_comp_utl.at['eclampsia', 'random_draw'] < p_comp_utl.at['eclampsia', 'complication_prob']:
                p_comp_utl.at['eclampsia', 'complication'] = True
                df.at[individual_id, 'la_eclampsia'] = True
                if self.sim.rng.random_sample() < params['prob_still_birth_eclampsia']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

            if p_comp_utl.at['sepsis', 'random_draw'] < p_comp_utl.at['sepsis', 'complication_prob']:
                p_comp_utl.at['sepsis', 'complication'] = True
                df.at[individual_id, 'la_sepsis'] = True
                if self.sim.rng.random_sample() < params['prob_still_birth_sepsis']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

            if p_comp_utl.at['aph', 'random_draw'] < p_comp_utl.at['aph', 'complication_prob']:
                p_comp_utl.at['aph', 'complication'] = True
                df.at[individual_id, 'la_aph'] = True
                if self.sim.rng.random_sample() < params['prob_still_birth_aph']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

            # If a complication is experienced the case fatality rate is used to determine if this will cause death

            if p_comp_utl.at['eclampsia', 'complication'] and p_comp_utl.at['eclampsia', 'random_draw2'] \
                < p_comp_utl.at['eclampsia', 'prob_cfr']:
                    p_comp_utl.at['eclampsia', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True
                    if self.sim.rng.random_sample() < params['prob_still_birth_eclampsia_md']:
                        df.at[individual_id, 'la_still_birth_this_delivery'] = True

            if p_comp_utl.at['sepsis', 'complication'] and p_comp_utl.at['sepsis', 'random_draw2'] \
                < p_comp_utl.at['sepsis', 'prob_cfr']:
                    p_comp_utl.at['sepsis', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True
                    if self.sim.rng.random_sample() < params['prob_still_birth_sepsis_md']:
                        df.at[individual_id, 'la_still_birth_this_delivery'] = True

            if p_comp_utl.at['aph', 'complication'] and p_comp_utl.at['aph', 'random_draw2'] \
                < p_comp_utl.at['aph', 'prob_cfr']:
                    p_comp_utl.at['aph', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True
                    if self.sim.rng.random_sample() < params['prob_still_birth_aph_md']:
                        df.at[individual_id, 'la_still_birth_this_delivery'] = True

            deaths = pd.Series(p_comp_utl['maternal_death'])
            if deaths.any():
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='labour'), self.sim.date)

    # ============================ COMPLICATIONS FOR OBSTRUCTED LABOUR ===============================================

    #  TODO: Apply risk factors for complications in obstructed labour

        risk_eclampsia = params['prob_an_eclampsia_pl_ol']
        risk_sepsis = params['prob_an_sepsis_pl_ol']
        risk_aph = params['prob_an_aph_pl_ol']
        risk_ur= params ['prob_an_ur_pl_ol']

        if df.at[individual_id, 'la_labour'] == 'prolonged_or_obstructed_labour':
            p_comp_sl = pd.DataFrame(data=[(risk_eclampsia, self.sim.rng.random_sample(size=1), False,
                                            params['cfr_eclampsia'], self.sim.rng.random_sample(size=1), False),
                                           (risk_sepsis, self.sim.rng.random_sample(size=1), False,
                                            params['cfr_sepsis'], self.sim.rng.random_sample(size=1), False),
                                           (risk_aph, self.sim.rng.random_sample(size=1), False,
                                            params['cfr_aph'], self.sim.rng.random_sample(size=1), False),
                                           (risk_ur, self.sim.rng.random_sample(size=1), False,
                                            params['cfr_uterine_rupture'], self.sim.rng.random_sample(size=1), False)],
                                     columns=['complication_prob', 'random_draw', 'complication', 'prob_cfr',
                                              'random_draw2', 'maternal_death'], index=('eclampsia', 'sepsis', 'aph',
                                                                                        'uterine_rupture'))

            if p_comp_sl.at['eclampsia', 'random_draw'] < p_comp_sl.at['eclampsia', 'complication_prob']:
                p_comp_sl.at['eclampsia', 'complication'] = True
                df.at[individual_id, 'la_eclampsia'] = True
                if self.sim.rng.random_sample() < params['prob_still_birth_eclampsia']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

            if p_comp_sl.at['sepsis', 'random_draw'] < p_comp_sl.at['sepsis', 'complication_prob']:
                p_comp_sl.at['sepsis', 'complication'] = True
                df.at[individual_id, 'la_sepsis'] = True
                if self.sim.rng.random_sample() < params['prob_still_birth_sepsis']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

            if p_comp_sl.at['aph', 'random_draw'] < p_comp_sl.at['aph', 'complication_prob']:
                p_comp_sl.at['aph', 'complication'] = True
                df.at[individual_id, 'la_aph'] = True
                if self.sim.rng.random_sample() < params['prob_still_birth_aph']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

            if p_comp_sl.at['uterine_rupture', 'random_draw'] < p_comp_sl.at['uterine_rupture', 'complication_prob']:
                p_comp_sl.at['uterine_rupture', 'complication'] = True
                df.at[individual_id, 'la_uterine_rupture'] = True
                if self.sim.rng.random_sample() < params['prob_still_birth_ur']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

            # If a complication is experienced the case fatality rate is used to determine if this will cause death

            if p_comp_sl.at['eclampsia', 'complication'] and p_comp_sl.at['eclampsia', 'random_draw2'] \
                < p_comp_sl.at['eclampsia', 'prob_cfr']:
                    p_comp_sl.at['eclampsia', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True
                    if self.sim.rng.random_sample() < params['prob_still_birth_eclampsia_md']:
                        df.at[individual_id, 'la_still_birth_this_delivery'] = True

            if p_comp_sl.at['sepsis', 'complication'] and p_comp_sl.at['sepsis', 'random_draw2'] \
                < p_comp_sl.at['sepsis', 'prob_cfr']:
                    p_comp_sl.at['sepsis', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True
                    if self.sim.rng.random_sample() < params['prob_still_birth_sepsis_md']:
                        df.at[individual_id, 'la_still_birth_this_delivery'] = True

            if p_comp_sl.at['aph', 'complication'] and p_comp_sl.at['aph', 'random_draw2'] \
                < p_comp_sl.at['aph', 'prob_cfr']:
                    p_comp_sl.at['aph', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True
                    if self.sim.rng.random_sample() < params['prob_still_birth_aph_md']:
                        df.at[individual_id, 'la_still_birth_this_delivery'] = True

            if p_comp_sl.at['uterine_rupture', 'complication'] and p_comp_sl.at['uterine_rupture', 'random_draw2'] \
                < p_comp_sl.at['uterine_rupture', 'prob_cfr']:
                    p_comp_sl.at['uterine_rupture', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True
                    if self.sim.rng.random_sample() < params['prob_still_birth_ur_md']:
                        df.at[individual_id, 'la_still_birth_this_delivery'] = True

            deaths = pd.Series(p_comp_sl['maternal_death'])
            if deaths.any():
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='labour'), self.sim.date)

    # ============================ COMPLICATIONS FOR PRETERM LABOUR ===================================================

        #  TODO: Apply riskfactors for complications of preterm labour?

        risk_eclampsia = params['prob_an_eclampsia_ptb']
        risk_sepsis = params['prob_an_sepsis_ptb']
        risk_aph = params['prob_an_aph_ptb']

        if df.at[individual_id, 'la_labour'] == 'preterm_labour':
            p_comp_sl = pd.DataFrame(data=[(risk_eclampsia, self.sim.rng.random_sample(size=1), False,
                                            params['cfr_eclampsia'], self.sim.rng.random_sample(size=1), False),
                                           (risk_sepsis, self.sim.rng.random_sample(size=1), False,
                                            params['cfr_sepsis'], self.sim.rng.random_sample(size=1), False),
                                           (risk_aph, self.sim.rng.random_sample(size=1), False,
                                            params['cfr_aph'], self.sim.rng.random_sample(size=1), False)],
                                     columns=['complication_prob', 'random_draw', 'complication', 'prob_cfr',
                                              'random_draw2', 'maternal_death'], index=('eclampsia', 'sepsis', 'aph'))

            if p_comp_sl.at['eclampsia', 'random_draw'] < p_comp_sl.at['eclampsia', 'complication_prob']:
                p_comp_sl.at['eclampsia', 'complication'] = True
                df.at[individual_id, 'la_eclampsia'] = True
                if self.sim.rng.random_sample() < params['prob_still_birth_eclampsia']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

            if p_comp_sl.at['sepsis', 'random_draw'] < p_comp_sl.at['sepsis', 'complication_prob']:
                p_comp_sl.at['sepsis', 'complication'] = True
                df.at[individual_id, 'la_sepsis'] = True
                if self.sim.rng.random_sample() < params['prob_still_birth_sepsis']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

            if p_comp_sl.at['aph', 'random_draw'] < p_comp_sl.at['aph', 'complication_prob']:
                p_comp_sl.at['aph', 'complication'] = True
                df.at[individual_id, 'la_aph'] = True
                if self.sim.rng.random_sample() < params['prob_still_birth_aph']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

            # If a complication is experienced the case fatality rate is used to determine if this will cause death

            if p_comp_sl.at['eclampsia', 'complication'] and p_comp_sl.at['eclampsia', 'random_draw2'] \
                < p_comp_sl.at['eclampsia', 'prob_cfr']:
                    p_comp_sl.at['eclampsia', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True
                    if self.sim.rng.random_sample() < params['prob_still_birth_eclampsia_md']:
                        df.at[individual_id, 'la_still_birth_this_delivery'] = True

            if p_comp_sl.at['sepsis', 'complication'] and p_comp_sl.at['sepsis', 'random_draw2'] \
                < p_comp_sl.at['sepsis', 'prob_cfr']:
                    p_comp_sl.at['sepsis', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True
                    if self.sim.rng.random_sample() < params['prob_still_birth_sepsis_md']:
                        df.at[individual_id, 'la_still_birth_this_delivery'] = True

            if p_comp_sl.at['aph', 'complication'] and p_comp_sl.at['aph', 'random_draw2'] \
                < p_comp_sl.at['aph', 'prob_cfr']:
                    p_comp_sl.at['aph', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True
                    if self.sim.rng.random_sample() < params['prob_still_birth_aph_md']:
                        df.at[individual_id, 'la_still_birth_this_delivery'] = True

            deaths = pd.Series(p_comp_sl['maternal_death'])
            if deaths.any():
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='labour'), self.sim.date)


class PostpartumLabourEvent(Event, IndividualScopeEventMixin):

        """applies probability of postpartum complications to women who have just delivered """

        def __init__(self, module, individual_id, cause):
            super().__init__(module, person_id=individual_id)

        def apply(self, individual_id):
            df = self.sim.population.props
            params = self.module.parameters
            m = self

# ============================== POSTPARTUM COMPLICATIONS FOLLOWING UNOBSTRUCTED TERM LABOUR ==========================

            # TODO: risk factors
            risk_eclampsia = params['prob_pn_eclampsia_utl']
            risk_sepsis = params['prob_pn_sepsis_utl']
            risk_pph = params['prob_pn_pph_utl']

            if df.at[individual_id, 'la_labour'] == 'unobstructed_term_labour':
                p_comp_sl = pd.DataFrame(data=[(risk_eclampsia, self.sim.rng.random_sample(size=1), False,
                                                params['cfr_pn_eclampsia'], self.sim.rng.random_sample(size=1), False),
                                               (risk_sepsis, self.sim.rng.random_sample(size=1), False,
                                                params['cfr_pn_sepsis'], self.sim.rng.random_sample(size=1), False),
                                               (risk_pph, self.sim.rng.random_sample(size=1), False,
                                                params['cfr_pn_pph'], self.sim.rng.random_sample(size=1), False)],
                                         columns=['complication_prob', 'random_draw', 'complication', 'prob_cfr',
                                                  'random_draw2', 'maternal_death'],
                                         index=('eclampsia', 'sepsis', 'pph'))

                # Determine is a woman who is in normal labour will experience a complication during delivery

                # TODO: consider increased likelihood of eclamptic fit in women who fitted in labour

                if p_comp_sl.at['eclampsia', 'random_draw'] < p_comp_sl.at['eclampsia', 'complication_prob']:
                    p_comp_sl.at['eclampsia', 'complication'] = True
                    df.at[individual_id, 'la_eclampsia'] = True
                if p_comp_sl.at['eclampsia', 'complication'] and p_comp_sl.at['eclampsia', 'random_draw2'] \
                    < p_comp_sl.at['eclampsia', 'prob_cfr']:
                    p_comp_sl.at['eclampsia', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True

                # TODO: consider women who are diagnosed as septic during labour

                if p_comp_sl.at['sepsis', 'random_draw'] < p_comp_sl.at['sepsis', 'complication_prob']:
                    p_comp_sl.at['sepsis', 'complication'] = True
                    df.at[individual_id, 'la_sepsis'] = True
                if p_comp_sl.at['sepsis', 'complication'] and p_comp_sl.at['sepsis', 'random_draw2'] \
                    < p_comp_sl.at['sepsis', 'prob_cfr']:
                    p_comp_sl.at['sepsis', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True

                if p_comp_sl.at['pph', 'random_draw'] < p_comp_sl.at['pph', 'complication_prob']:
                    p_comp_sl.at['pph', 'complication'] = True
                    df.at[individual_id, 'la_pph'] = True
                if p_comp_sl.at['pph', 'complication'] and p_comp_sl.at['pph', 'random_draw2'] \
                    < p_comp_sl.at['pph', 'prob_cfr']:
                    p_comp_sl.at['pph', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True

                deaths = pd.Series(p_comp_sl['maternal_death'])
                if deaths.any():
                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                          cause='labour'), self.sim.date)

# TODO NEED TO CONSIDER THE EFFECTS FOLLOWING ABORTION/SPONT MISCARRIAGE

# ============================ POSTPARTUM COMPLICATIONS FOLLOWING PROLONGED/OBSTRUCTED LABOUR =========================

            # TODO: risk factors
            risk_eclampsia = params['prob_pn_eclampsia_pl_ol']
            risk_sepsis = params['prob_pn_sepsis_pl_ol']
            risk_pph = params['prob_pn_pph_pl_ol']

            if df.at[individual_id, 'la_labour'] == 'prolonged_or_obstructed_labour':
                p_comp_sl = pd.DataFrame(data=[(risk_eclampsia, self.sim.rng.random_sample(size=1), False,
                                                params['cfr_pn_eclampsia'], self.sim.rng.random_sample(size=1), False),
                                               (risk_sepsis, self.sim.rng.random_sample(size=1), False,
                                                params['cfr_pn_sepsis'], self.sim.rng.random_sample(size=1), False),
                                               (risk_pph, self.sim.rng.random_sample(size=1), False,
                                                params['cfr_pn_pph'], self.sim.rng.random_sample(size=1), False)],
                                         columns=['complication_prob', 'random_draw', 'complication', 'prob_cfr',
                                                  'random_draw2', 'maternal_death'],
                                         index=('eclampsia', 'sepsis', 'pph'))

                # Determine is a woman who is in normal labour will experience a complication during delivery

                # TODO: consider increased likelihood of eclamptic fit in women who fitted in labour

                if p_comp_sl.at['eclampsia', 'random_draw'] < p_comp_sl.at['eclampsia', 'complication_prob']:
                    p_comp_sl.at['eclampsia', 'complication'] = True
                    df.at[individual_id, 'la_eclampsia'] = True
                if p_comp_sl.at['eclampsia', 'complication'] and p_comp_sl.at['eclampsia', 'random_draw2'] \
                    < p_comp_sl.at['eclampsia', 'prob_cfr']:
                    p_comp_sl.at['eclampsia', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True

                # TODO: consider women who are diagnosed as septic during labour

                if p_comp_sl.at['sepsis', 'random_draw'] < p_comp_sl.at['sepsis', 'complication_prob']:
                    p_comp_sl.at['sepsis', 'complication'] = True
                    df.at[individual_id, 'la_sepsis'] = True
                if p_comp_sl.at['sepsis', 'complication'] and p_comp_sl.at['sepsis', 'random_draw2'] \
                    < p_comp_sl.at['sepsis', 'prob_cfr']:
                    p_comp_sl.at['sepsis', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True

                # TODO: consider bleeding in relation to UR

                if p_comp_sl.at['pph', 'random_draw'] < p_comp_sl.at['pph', 'complication_prob']:
                    p_comp_sl.at['pph', 'complication'] = True
                    df.at[individual_id, 'la_pph'] = True
                if p_comp_sl.at['pph', 'complication'] and p_comp_sl.at['pph', 'random_draw2'] \
                    < p_comp_sl.at['pph', 'prob_cfr']:
                    p_comp_sl.at['pph', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True

                deaths = pd.Series(p_comp_sl['maternal_death'])
                if deaths.any():
                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                          cause='labour'), self.sim.date)

# ============================ POSTPARTUM COMPLICATIONS FOLLOWING PRETERM LABOUR =======================================

            # TODO: risk factors
            risk_eclampsia = params['prob_pn_eclampsia_ptb']
            risk_sepsis = params['prob_pn_sepsis_ptb']
            risk_pph = params['prob_pn_pph_ptb']

            if df.at[individual_id, 'la_labour'] == 'prolonged_or_obstructed_labour':
                p_comp_sl = pd.DataFrame(data=[(risk_eclampsia, self.sim.rng.random_sample(size=1), False,
                                                params['cfr_pn_eclampsia'], self.sim.rng.random_sample(size=1), False),
                                               (risk_sepsis, self.sim.rng.random_sample(size=1), False,
                                                params['cfr_pn_sepsis'], self.sim.rng.random_sample(size=1), False),
                                               (risk_pph, self.sim.rng.random_sample(size=1), False,
                                                params['cfr_pn_pph'], self.sim.rng.random_sample(size=1), False)],
                                         columns=['complication_prob', 'random_draw', 'complication', 'prob_cfr',
                                                  'random_draw2', 'maternal_death'],
                                         index=('eclampsia', 'sepsis', 'pph'))

                if p_comp_sl.at['eclampsia', 'random_draw'] < p_comp_sl.at['eclampsia', 'complication_prob']:
                    p_comp_sl.at['eclampsia', 'complication'] = True
                    df.at[individual_id, 'la_eclampsia'] = True
                if p_comp_sl.at['eclampsia', 'complication'] and p_comp_sl.at['eclampsia', 'random_draw2'] \
                    < p_comp_sl.at['eclampsia', 'prob_cfr']:
                    p_comp_sl.at['eclampsia', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True

                if p_comp_sl.at['sepsis', 'random_draw'] < p_comp_sl.at['sepsis', 'complication_prob']:
                    p_comp_sl.at['sepsis', 'complication'] = True
                    df.at[individual_id, 'la_sepsis'] = True
                if p_comp_sl.at['sepsis', 'complication'] and p_comp_sl.at['sepsis', 'random_draw2'] \
                    < p_comp_sl.at['sepsis', 'prob_cfr']:
                    p_comp_sl.at['sepsis', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True

                if p_comp_sl.at['pph', 'random_draw'] < p_comp_sl.at['pph', 'complication_prob']:
                    p_comp_sl.at['pph', 'complication'] = True
                    df.at[individual_id, 'la_pph'] = True
                if p_comp_sl.at['pph', 'complication'] and p_comp_sl.at['pph', 'random_draw2'] \
                    < p_comp_sl.at['pph', 'prob_cfr']:
                    p_comp_sl.at['pph', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True

                deaths = pd.Series(p_comp_sl['maternal_death'])
                if deaths.any():
                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                          cause='labour'), self.sim.date)

# ==============================  COMPLICATIONS FOLLOWING SPONTANEOUS MISCARRIAGE ======================================

        # TODO: consider stage of pregnancy loss and its impact on liklihood of complications i.e retained product

            risk_sepsis= params['prob_sa_sepsis']
            risk_pph = params['prob_sa_pph']

            if df.at[individual_id, 'la_miscarriage_date'] == self.sim.date:
                p_comp_mc = pd.DataFrame(data=[(risk_sepsis, self.sim.rng.random_sample(size=1), False,
                                                        params['cfr_pn_sepsis'], self.sim.rng.random_sample(size=1),
                                                False), (risk_pph, self.sim.rng.random_sample(size=1), False,
                                                        params['cfr_pn_pph'], self.sim.rng.random_sample(size=1), False)],
                                                  columns=['complication_prob', 'random_draw', 'complication', 'prob_cfr',
                                                           'random_draw2', 'maternal_death'],
                                                  index=('sepsis', 'pph'))

                if p_comp_mc.at['sepsis', 'random_draw'] < p_comp_mc.at['sepsis', 'complication_prob']:
                    p_comp_mc.at['sepsis', 'complication'] = True
                    df.at[individual_id, 'la_sepsis'] = True
                if p_comp_mc.at['sepsis', 'complication'] and p_comp_mc.at['sepsis', 'random_draw2'] \
                    < p_comp_mc.at['sepsis', 'prob_cfr']:
                    p_comp_mc.at['sepsis', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True

                if p_comp_mc.at['pph', 'random_draw'] < p_comp_mc.at['pph', 'complication_prob']:
                    p_comp_mc.at['pph', 'complication'] = True
                    df.at[individual_id, 'la_pph'] = True
                if p_comp_mc.at['pph', 'complication'] and p_comp_mc.at['pph', 'random_draw2'] \
                    < p_comp_mc.at['pph', 'prob_cfr']:
                    p_comp_mc.at['pph', 'maternal_death'] = True
                    df.at[individual_id, 'la_died_in_labour'] = True

                deaths = pd.Series(p_comp_mc['maternal_death'])
                if deaths.any():
                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                          cause='miscarriage complications'),
                                            self.sim.date)

#  =============================================== RESET LABOUR STATUS =================================================

        # (commented out so can view effects at this point)
#            if df.at[individual_id, 'is_alive']:
#                df.at[individual_id, 'la_labour'] = "not_in_labour"


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

"""
First draft of Labour module (Natural history)
Documentation:
"""
import logging

import pandas as pd

import numpy as np


from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Labour (Module):

    """
    This module models labour and delivery and generates the properties for "complications" of delivery """

    PARAMETERS = {

        'prob_SL': Parameter(
            Types.REAL, 'probability of a woman entering spontaneous unobstructed labour'),
        'prob_abortion': Parameter(
            Types.REAL, 'probability of a woman experiencing an abortion before 28 weeks gestation'),
        'prob_PL_OL': Parameter(
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
        'rr_ptl_prev_ptb': Parameter(
            Types.REAL, 'relative risk of a woman going into pre term labour if she has had a preterm birth for any '
                        'prior deliveries'),
        'prob_live_birth_SL': Parameter(
             Types.REAL, 'probability of live birth following spontaneous labour'),
        'prob_live_birth_IL': Parameter(
            Types.REAL, 'probability of live birth following induced labour'),
        'prob_live_birth_planned_CS': Parameter(
            Types.REAL, 'probability of live birth following a planned caesarean section'),
        'prob_live_birth_PL/OL': Parameter(
            Types.REAL, 'probability of live birth following prolonged/obstructed labour'),
        'prob_still_birth_SL': Parameter (
            Types.REAL, 'probability of still birth following spontaneous labour'),
        'prob_still_birth_IL': Parameter(
            Types.REAL, 'probability of still birth following induced labour'),
        'prob_still_birth_PL/OL': Parameter(
            Types.REAL, 'probability of still birth following prolonged labour'),
        'prob_still_birth_abortion': Parameter(
            Types.REAL, 'probability of still birth an abortion'),
        'prob_pregnancy' :Parameter(
            Types.REAL, 'baseline probability of pregnancy'), #DUMMY PARAMTER
        'base_init_parity':Parameter(
            Types.REAL, 'value of parity on initialisation')

      }

    PROPERTIES = {

        'la_labour': Property(Types.CATEGORICAL,'not in labour, spontaneous unobstructed labour, prolonged or '
                                                'obstructed labour, Preterm Labour', categories=['not_in_labour',
                                                'spontaneous_unobstructed_labour','prolonged_or_obstructed_labour',
                                                                                                 'pretterm_labour']),
        'la_abortion': Property(Types.DATE, 'the date on which a pregnant has had an abortion'),
        'la_live_birth': Property(Types.BOOL, 'labour ends in a live birth'),
        'la_still_birth': Property(Types.BOOL, 'labour ends in a still birth'),
        'la_parity': Property(Types.INT, 'total number of previous deliveries'),
        'la_previous_cs': Property(Types.INT, 'number of previous deliveries by caesarean section'),
        'la_immediate_postpartum': Property(Types.BOOL, ' postpartum period from delivery to 48 hours post'),
        'la_previous_ptb': Property(Types.BOOL, 'whether the woman has had a previous preterm delivery for any of her'
                                                'previous deliveries'),
        'la_conception_date': Property(Types.DATE, 'date on which current pregnancy was conceived'),
        'la_due_date': Property(Types.DATE, 'date on which the woman would be due to give birth if she carries her '
                                            'pregnancy to term'),
        'la_gestational_age': Property(Types.DATE, 'number of weeks since conception, measured in weeks')
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters

        params['prob_SL'] = 0.80
        params['prob_induction'] = 0.10
        params['prob_planned_CS'] = 0.03
        params['prob_abortion'] = 0.15
        params['prob_PL_OL'] = 0.06
        params['rr_PL_OL_nuliparity'] = 1.8
        params['rr_PL_OL_parity_3'] = 0.8
        params['rr_PL_OL_age_less20'] = 1.3
        params['rr_PL_OL_baby>3.5kg'] = 1.2
        params['rr_PL_OL_baby<1.5kg'] = 0.7
        params['rr_PL_OL_bmi<18'] = 0.8
        params['rr_PL_OL_bmi>25'] = 1.4
        params['prob_ptl'] = 0.20
        params['rr_ptl_persistent_malaria'] = 1.4
        params['rr_ptl_prev_ptb'] = 2.13
        params['prob_live_birth_SL'] = 0.85
        params['prob_live_birth_IL'] = 0.8
        params['prob_live_birth_PL'] = 0.75
        params['prob_live_birth_OL'] = 0.7
        params['prob_still_birth_SL'] = 0.15
        params['prob_still_birth_IL'] = 0.2
        params['prob_still_birth_PL'] = 0.25
        params['prob_still_birth_OL'] = 0.3
        params['prob_pregnancy'] = 0.03  # DUMMY PREGNANCY GENERATOR
        params['base_init_parity'] = 0  # DUMMY

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
        df['la_abortion'] = pd.NaT
        df['la_live_birth'] = False
        df['la_still_birth'] = False
        df['la_partiy'] = 0
        df['la_previous_cs'] = 0
        df['la_immediate_postpartum'] = False
        df['la_previous_ptb'] = False
        df['la_conception_date'] = pd.NaT
        df['la_gestation'] = 0
        df['la_due_date'] = pd.NaT
        df['la_gestational_age'] = pd.NaT

    # -----------------------------------ASSIGN PREGNANCY AND DUE DATE AT BASELINE (DUMMY) ----------------------------

        # Dummy code to generate pregnancies from Labour.py

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
        # Assigns a date of conception for women who are pregnant at baseline

        pregnant_idx = df.index[df.is_pregnant & df.is_alive]

        random_draw = pd.Series(rng.random_sample(size=len(pregnant_idx)),
                                index=df.index[df.is_pregnant & df.is_alive])

        simdate = pd.Series(self.sim.date, index=pregnant_idx)
        dfx = pd.concat((simdate, random_draw), axis=1)
        dfx.columns = ['simdate', 'random_draw']
        df.loc[pregnant_idx, 'la_gestational_age'] = (42 - 42 * dfx.random_draw)

        pregnant_idx = df.index[df.is_pregnant & df.is_alive]

        simdate = pd.Series(self.sim.date, index=pregnant_idx)
        gest_weeks = pd.Series(df.la_gestational_age, index=pregnant_idx)
        dfx = pd.concat([simdate, gest_weeks], axis=1)
        dfx.columns = ['simdate', 'gest_weeks']
        dfx['la_conception_date'] = dfx['simdate'] - pd.to_timedelta(dfx['gest_weeks'], unit='w')
        dfx['due_date_mth'] = 40 - dfx['gest_weeks']

        # Dummy to set women who are pregnant at birth a due_date (cannot be preterm)

        dfx['due_date'] = dfx['simdate'] + pd.to_timedelta(dfx['due_date_mth'], unit='w')
        df.loc[pregnant_idx, 'la_conception_date'] = dfx.la_conception_date
        df.loc[pregnant_idx,'due_date'] =dfx.due_date


#  ----------------------------ASSIGNING PARITY AT BASELINE (DUMMY)-----------------------------------------------------

#       Will eventually pull in from the DHS (normal distribution?)
#       Currently roughly weighted to account for age - this is much too long

        women_parity_1524_idx = df.index[(df.age_years >= 15) & (df.age_years <= 24) & (df.is_alive == True)
                                    & (df.sex == 'F')]

        baseline_p = pd.Series(m.base_init_parity, index=women_parity_1524_idx)

        random_draw2 = pd.Series(self.rng.choice(range(0, 4), p=[0.40, 0.35, 0.15, 0.10],
                                                 size=len(women_parity_1524_idx)),
                                 index=df.index[(df.age_years >= 15) & (df.age_years <= 24)
                                                & df.is_alive & (df.sex == 'F')])

        dfi = pd.concat([baseline_p, random_draw2], axis=1)
        dfi.columns = ['baseline_p', 'random_draw2']
        idx_parity = dfi.index[dfi.baseline_p < dfi.random_draw2]
        df.loc[idx_parity, 'la_parity'] = random_draw2

        women_parity_2540_idx = df.index[(df.age_years >= 25) & (df.age_years <= 40) & (df.is_alive == True)
                                         & (df.sex == 'F')]

        baseline_p = pd.Series(m.base_init_parity, index=women_parity_2540_idx)

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

        baseline_p = pd.Series(m.base_init_parity, index=women_parity_4149_idx)

        random_draw = pd.Series(self.rng.choice(range(0, 7), p=[0.05, 0.10, 0.25, 0.30, 0.25, 0.03, 0.02],
                                                size=len(women_parity_4149_idx)), index=df.index[(df.age_years >= 41)
                                                                                                 & (df.age_years <= 49)
                                                                                                 & (df.is_alive == True)
                                                                                                 & (df.sex == 'F')])

        dfi = pd.concat([baseline_p, random_draw], axis=1)
        dfi.columns = ['baseline_p', 'random_draw']
        idx_parity = dfi.index[dfi.baseline_p < dfi.random_draw]
        df.loc[idx_parity, 'la_parity'] = random_draw

#   ------------------------------ ASSIGN PREVIOUS CS AT BASELINE (dummy)-----------------------------------------------

        women_para_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') & (df.la_parity >= 1)]

        baseline_cs = pd.Series(df.la_previous_cs, index=women_para_idx)

        # (Confused as to why this range has to be 0-3 when it you cant get 3 as an option)

        random_draw = pd.Series(self.rng.choice(range(0, 3), p=[0.85, 0.10, 0.05], size=len(women_para_idx)),
                               index=df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F')
                                              & (df.la_parity >= 1)])

        dfx = pd.concat([baseline_cs, random_draw], axis=1)
        dfx.columns = ['baseline_cs', 'random_draw']
        idx_prev_cs = dfx.index[dfx.baseline_cs < dfx.random_draw]
        df.loc[idx_prev_cs, 'la_previous_cs'] = random_draw

        # ------------------------------ ASSIGN PREVIOUS PTB AT BASELINE (dummy)---------------------------------------

        women_ptb_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') & (df.la_parity >= 1)]

        # do we want to exclude para1 who delivered by CS- not sure of indication or prevelance of CS in PTB
        # need to consider how we incorporate risk factors
        # does this also need to be an interger

        baseline_ptb = pd.Series(m.prob_ptl, index=women_ptb_idx)

        random_draw = pd.Series(rng.random_sample(size=len(women_ptb_idx)),
                                index=df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') &
                                               (df.la_parity >= 1)])

        dfx = pd.concat([baseline_ptb, random_draw], axis=1)
        dfx.columns = ['baseline_ptb', 'random_draw']
        idx_prev_ptb = dfx.index[dfx.baseline_ptb > dfx.random_draw]
        df.loc[idx_prev_ptb, 'la_previous_ptb'] = True


    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        event = LabourPollEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=1))

        event = LabourLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        df.at[child_id, 'la_labour'] = 'not_in_labour'
        df.at[child_id, 'la_abortion'] = pd.NaT
        df.at[child_id, 'la_live_birth'] = False
        df.at[child_id, 'la_still_birth'] = False
        df.at[child_id, 'la_parity'] = 0
        df.at[child_id, 'la_previous_cs'] = 0
        df.at[child_id, 'la_immediate_postpartum'] = False
        df.at[child_id, 'la_previous_ptb'] = False
        df.at[child_id, 'la_conception_date'] = pd.NaT
        df.at[child_id, 'la_due_date'] = pd.NaT
        df.at[child_id, 'la_gestational_age']= pd.NaT


class LabourPollEvent(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(days=1))

    # Need to consider frequency

    def apply(self, population):

        df = population.props
        m = self.module
        rng = m.rng

        # Poll for women who are due to go into labour on the date

        due_labour_idx = df.index[(df.is_pregnant == True) & df.is_alive & (df.due_date == self.sim.date)]

        due_date = pd.Series(df.due_date, index=due_labour_idx)
        conception_date = pd.Series(df.date_of_last_pregnancy, index=due_labour_idx)
        dfx = pd.concat([due_date, conception_date], axis=1)
        dfx.columns = ['due_date', 'conception_date']

        dfx['gestation'] = dfx['due_date'] - dfx['conception_date']
        dfx['gestation'] = dfx['gestation'] / np.timedelta64(1, 'W')

        spontlab_idx = dfx.index[dfx.gestation >= 37]

        pretermlab_idx = dfx.index[(dfx.gestation >= 28) & (dfx.gestation <= 36)]
        abortion_idx = dfx.index[dfx.gestation <= 27]

        df.loc[spontlab_idx, 'la_labour'] = 'spontaneous_labour'

# Obstuction??

        df.loc[pretermlab_idx, 'la_labour'] = 'pretterm_labour'

# df.loc[pretermlab_idx, 'la_prev_ptb'] = True

        df.loc[abortion_idx, 'is_pregnant'] = False

# do we need an 'miscarriage' property

# INDUCTION AND PLANNED CS


class LabourLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles lifestyle logging"""
    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(days=self.repeat))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props

        logger.debug('%s|person_one|%s',
                          self.sim.date, df.loc[0].to_dict())

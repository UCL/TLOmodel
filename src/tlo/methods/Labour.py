"""
First draft of Labour module (Natural history)
Documentation:
"""

import pandas as pd
import numpy as np


from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent


class Labour (Module):
    """
    This module models labour and delivery and generates the properties for "complications" of delivery """

    PARAMETERS = {

        'prob_SL': Parameter(
            Types.REAL, 'probability of a woman entering spontaneous unobstructed labour'),
        'prob_induction': Parameter(
            Types.REAL, 'probability of a womans labour being induced'),
        'prob_planned_CS': Parameter(
            Types.REAL, 'probability of a woman undergoing a planned caesarean section'),
        'prob_abortion': Parameter(
            Types.REAL, 'probability of a woman experiencing an abortion before 28 weeks gestation'),
        'prob_PL_OL': Parameter(
            Types.REAL, 'probability of a woman entering prolonged/obstructed labour'),
        'rr_PL_OL_nuliparity': Parameter(
            Types.REAL, 'relative risk of a woman entering prolonged/obstructed labour if they are nuliparous'),
        'rr_PL_OL_parity_3':Parameter(
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
            Types.REAL, 'value of parity on initialisation'),
        'base_init_prev_cs': Parameter(
            Types.REAL, 'baseline probability of having had a previous caesarean for woman where parity =>1')
      }

    PROPERTIES = {

        'la_is_pregnant':Property(Types.BOOL, 'currently pregnant'),
        'la_labour': Property(Types.CATEGORICAL,'not in labour, spontaneous unobstructed labour, prolonged or '
                                                'obstructed labour, Preterm Labour',
                                categories=['not_in_labour', 'spontaneous_unobstructed_labour',
                                            'prolonged_or_obstructed_labour','pretterm_labour']),
        'la_planned_cs': Property(Types.BOOL, 'pregnant woman undergoes an elective caesarean section'),
        'la_induced_labour': Property(Types.BOOL, 'pregnant woman has labour induced'),
        'la_abortion': Property(Types.DATE, 'the date on which a pregnant has had an abortion'),
        'la_live_birth': Property(Types.BOOL, 'labour ends in a live birth'),
        'la_still_birth': Property(Types.BOOL, 'labour ends in a still birth'),
        'la_parity': Property(Types.INT, 'total number of previous deliveries'),
        'la_previous_cs': Property(Types.BOOL, 'has the woman ever had a previous caesarean'), # REFINE TO INTEGER
        'la_immediate_postpartum': Property(Types.BOOL, ' postpartum period from delivery to 48 hours post'),
        'la_previous_ptb': Property(Types.BOOL, 'whether the woman has had a previous preterm delivery for any of her'
                                                  'previous deliveries')
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
        params['prob_pregnancy'] = 0.03 # DUMMY PREGNANCY GENERATOR
        params['base_init_parity'] = 0  # DUMMY
        params['base_init_prev_cs'] = 0.10 # DUMMY

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

        df['la_is_pregnant'] = False
        df['la_labour'] = 'not_in_labour'
        df['la_planned_CS'] = False
        df['la_induced_labour'] = False
        df['la_abortion'] = pd.NaT
        df['la_live_birth'] = False
        df['la_still_birth'] = False
        df['la_partiy'] = 0
        df['la_previous_cs'] = False
        df['la_immediate_postpartum'] = False
        df ['la_previous_ptb'] = False

    # -----------------------------------ASSIGN PREGNANCY AT BASELINE (DUMMY) --------------------------------------

        # Dummy code to generate pregnancies (remove once running from demography)

        women_idx = df.index[(df.age_years >= 15) & (df.age_years <= 49) & df.is_alive & (df.sex == 'F')]

        eff_prob_preg = pd.Series(m.prob_pregnancy, index=women_idx)

        random_draw = pd.Series(rng.random_sample(size=len(women_idx)),
                                index=df.index[(df.age_years >= 15) & (df.age_years <= 49) & df.is_alive
                                               & (df.sex == 'F')])

        dfx = pd.concat([eff_prob_preg, random_draw], axis=1)
        dfx.columns = ['eff_prob_pregnancy', 'random_draw']
        idx_pregnant = dfx.index[dfx.eff_prob_pregnancy > dfx.random_draw]
        df.loc[idx_pregnant, 'la_is_pregnant'] = True

#  ----------------------------ASSIGNING PARITY AT BASELINE (DUMMY)-----------------------------------------------------

#      (Currently parity is assigned to all women of childbearing age as a random integer between 0-5
#      There is presently no probability or age weighting for parity which will need to be added

        women_parity_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F')]

        baseline_p = pd.Series(m.base_init_parity, index=women_parity_idx)

        random_draw2 = pd.Series(self.rng.choice([0, 1, 2, 3, 4, 5 ], size=len(women_parity_idx)),
                                 index=df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F')])

        dfi = pd.concat([baseline_p, random_draw2], axis=1)
        dfi.columns = ['baseline_p', 'random_draw2']
        idx_parity = dfi.index[dfi.baseline_p < dfi.random_draw2]
        df.loc[idx_parity, 'la_parity'] = random_draw2

# ------------------------------ ASSIGN PREVIOUS CS AT BASELINE (dummy)-----------------------------------------------

        # DUMMY- needs to be an integer value, paramter changed to boolean T/F for previous CS

        women_para_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') & (df.la_parity >= 1)]

        baseline_cs = pd.Series(m.base_init_prev_cs, index=women_para_idx)

        random_draw= pd.Series(rng.random_sample(size=len(women_para_idx)),
                                index=df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') &
                                               (df.la_parity >= 1)])

        dfx = pd.concat([baseline_cs, random_draw], axis=1)
        dfx.columns = ['baseline_cs', 'random_draw']
        idx_prev_cs = dfx.index[dfx.baseline_cs > dfx.random_draw]
        df.loc[idx_prev_cs, 'la_previous_cs'] = True

        # ------------------------------ ASSIGN PREVIOUS PTB AT BASELINE (dummy)---------------------------------------

        women_ptb_idx = df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') & (df.la_parity >= 1)]

        baseline_ptb = pd.Series(m.prob_ptl, index=women_ptb_idx)

        random_draw = pd.Series(rng.random_sample(size=len(women_ptb_idx)),
                                index=df.index[(df.age_years >= 15) & df.is_alive & (df.sex == 'F') &
                                               (df.la_parity >= 1)])

        dfx = pd.concat([baseline_ptb, random_draw], axis=1)
        dfx.columns = ['baseline_ptb', 'random_draw']
        idx_prev_ptb = dfx.index[dfx.baseline_ptb > dfx.random_draw]
        df.loc[idx_prev_ptb, 'la_previous_ptb'] = True

        # Interaction between previous CS and PTB- currently you can be para1 and the delivery was both PTB and CS

        # TO DO consider women who enter labour through induction and planned CS on initialisation?

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """

        event = LabourEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(weeks=1))

        event = LabourLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))


    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        df.at[child_id, 'la_labour'] = 'not_in_labour'
        df.at[child_id, 'la_planned_CS'] = False
        df.at[child_id, 'la_induced_labor'] = False
        df.at[child_id, 'la_abortion'] = pd.NaT
        df.at[child_id, 'la_live_birth'] = False
        df.at[child_id, 'la_still_birth'] = False
        df.at[child_id, 'la_parity'] = 0
        df.at[child_id, 'la_previous_cs'] = 0
        df.at[child_id, 'la_immediate_postpartum'] = False


class LabourEvent(RegularEvent, PopulationScopeEventMixin):
    """ Event moving pregnant women into labour

    A skeleton class for an event

    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """One line summary here

        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.

        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(weeks=1))

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """

        df = population.props
        m = self.module
        rng = m.rng

        # Weekly (review) event which decides which of the pregnant women in the simulation enter a "labour state"
        # based on gestation and risk factors

        #   ------------------------------------- UNOBSTRUCTED SPONTANEOUS LABOUR ------------------------------------

        # Must add impact of gestation on likelihood of women entering states of labour

        due_labour_idx = df.index[(df.la_is_pregnant == True) & df.is_alive]   # AND IS TERM!!!!

        eff_prob_spont_lab = pd.Series(m.prob_SL, index=due_labour_idx)

        random_draw = pd.Series(rng.random_sample(size=len(due_labour_idx)),
                                index=df.index[(df.la_is_pregnant == True)])

        dfx = pd.concat([eff_prob_spont_lab, random_draw], axis=1)
        dfx.columns = ['eff_prob_spont_lab', 'random_draw']
        idx_spont_lab = dfx.index[dfx.eff_prob_spont_lab > dfx.random_draw]
        df.loc[idx_spont_lab, 'la_labour'] = 'spontaneous_unobstructed_labour'

    #   --------------------------------------PROLONGED/ OBSTRUCTED LABOUR ---------------------------------------------

        due_pl_ol_idx = df.index[(df.la_is_pregnant == True) & df.is_alive]   # AND IS TERM!!!

        p_pl_or_ol = pd.Series(m.prob_PL_OL, index=due_pl_ol_idx)

        # why does this generate a person with a probability of less than 0.06?

        p_pl_or_ol.loc[(df.la_is_pregnant == True) & (df.la_parity == 0) & df.is_alive] *= m.rr_PL_OL_nuliparity
        p_pl_or_ol.loc[(df.la_is_pregnant == True) & (df.la_parity >= 3) & df.is_alive] *= m.rr_PL_OL_parity_3
        p_pl_or_ol.loc[(df.la_is_pregnant == True) & (df.age_years < 20) & df.is_alive] *= m.rr_PL_OL_age_less20

        random_draw = pd.Series(rng.random_sample(size=len(p_pl_or_ol)),index=df.index[(df.la_is_pregnant == True)
                                                                                       & df.is_alive])
        dfx = pd.concat([p_pl_or_ol, random_draw], axis=1)
        dfx.columns = ['p_pl_or_ol', 'random_draw']
        idx_pl_ol = dfx.index[dfx.p_pl_or_ol > dfx.random_draw]
        df.loc[idx_pl_ol, 'la_labour'] = 'prolonged_or_obstructed_labour'

        # I'm not sure if this is doing what I think it is

        df = population.props

    #   -------------------------------------------PRETERM LABOUR ---------------------------------------------------

        # will need to consider early vs late term as states?

        due_preterm_idx = df.index[(df.la_is_pregnant == True) & df.is_alive]   # AND IS PRETERM!!!

        p_preterm_lab = pd.Series(m.prob_PLT, index=due_preterm_idx)

        p_preterm_lab.loc[(df.la_is_pregnant == True) & df.is_alive] *=m.rr_ptl_persistent_malaria
        # To do: Identify risk factors for preterm labour

        random_draw = pd.Series(rng.random_sample(size=len(p_preterm_lab)),index=df.index[(df.la_is_pregnant == True)
                                                                                       & df.is_alive])
        dfx = pd.concat([p_preterm_lab, random_draw], axis=1)
        dfx.columns = ['p_preterm_lab', 'random_draw']
        idx_ptl = dfx.index[dfx.p_preterm_lab > dfx.random_draw]
        df.loc[idx_ptl, 'la_labour'] = 'preterm_labour'

        preterm_idx= df.index[(df.la_labour == 'preterm_labour')]


    # TO DO:

# -----------------------------------------   PLANNED CS   ------------------------------------------------------------
# ----------------------------------------- PLANNED INDUCTION ---------------------------------------------------------
# ---------------------------------------  ABORTION (MISCARRIGE) ------------------------------------------------------
# ----------------------------------------------BIRTH------------------------------------------------------------------
# ------------------------------------------- POSTPARTUM PERIOD  ------------------------------------------------------

    # Pass woman back to demography? Or take over completely?

    # How does time work in this instance?


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

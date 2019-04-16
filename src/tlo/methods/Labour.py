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
            Types.REAL, 'baseline probability of pregnancy'),  # DUMMY PARAMTER
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
        'prob_an_eclampsia_sl': Parameter(
            Types.REAL, 'probability of eclampsia in spontaneous unobstructed labours'),
        'prob_an_sepsis_sl': Parameter(
            Types.REAL, 'probability of sepsis in spontaneous unobstructed labour'),
        'prob_an_aph_sl': Parameter(
            Types.REAL, 'probability of antepartum haemorrhage in spontaneous unobstructed labour'),
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
            Types.REAL, 'probability of a still birth following antepartum haemorrhage'),
        'prob_still_birth_sepsis': Parameter(
            Types.REAL, 'probability of a still birth following sepsis in labour'),
        'prob_still_birth_ur': Parameter(
            Types.REAL, 'probability of a still birth following uterine rupture in labour'),
        'prob_still_birth_eclampsia': Parameter(
            Types.REAL, 'probability of still birth following eclampsia in labour'),
        'prob_pn_eclampsia_ptb': Parameter(
            Types.REAL, 'probability of eclampsia following delivery for women who were in spotaneous unobstructed '
                        'labour'),
        'prob_pn_pph_ptb': Parameter(
            Types.REAL, 'probability of an postpartum haemorrhage for women who were in preterm labour'),
        'prob_pn_sepsis_ptb': Parameter(
            Types.REAL, 'probability of sepsis following delivery for women in preterm labour'),
        'prob_pn_eclampsia_sl': Parameter(
            Types.REAL, 'probability of eclampsia following delivery for women who were in spotaneous unobstructed '
                        'labour'),
        'prob_pn_sepsis_sl': Parameter(
            Types.REAL, 'probability of sepsis following delivery for women who were in spontaneous unobstructed labour'),
        'prob_pn_pph_sl': Parameter(
            Types.REAL, 'probability of postpartum haemorrhage for women who were in spontaneous unobstructed labour'),
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

        'la_labour': Property(Types.CATEGORICAL,'not in labour, spontaneous unobstructed labour, prolonged or '
                                                'obstructed labour, Preterm Labour, Immediate Postpartum'
                                                '', categories=['not_in_labour', 'spontaneous_unobstructed_labour',
                                                                'prolonged_or_obstructed_labour', 'pretterm_labour',
                                                                'immediate_postpartum']),
        'la_still_birth_this_delivery': Property(Types.BOOL,'whether this womans pregnancy has ended in a stillbirth'),
        'la_abortion': Property(Types.DATE, 'the date on which a pregnant has had an abortion'),
        'la_parity': Property(Types.INT, 'total number of previous deliveries'),
        'la_previous_cs': Property(Types.INT, 'number of previous deliveries by caesarean section'),
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

        params['prob_pregnancy'] = 0.03  # DUMMY PREGNANCY GENERATOR
        params['prob_SL'] = 0.80
        params['prob_induction'] = 0.10
        params['prob_planned_CS'] = 0.03
        params['prob_abortion'] = 0.15
        params['prob_pl_ol'] = 0.06
        params['rr_PL_OL_nuliparity'] = 1.8
        params['rr_PL_OL_parity_3'] = 0.8
        params['rr_PL_OL_age_less20'] = 1.3
        params['rr_PL_OL_baby>3.5kg'] = 1.2
        params['rr_PL_OL_baby<1.5kg'] = 0.7
        params['rr_PL_OL_bmi<18'] = 0.8
        params['rr_PL_OL_bmi>25'] = 1.4
        params['prob_ptl'] = 0.2
        params['prob_an_eclampsia_ptb'] = 0.02
        params['prob_an_aph_ptb'] = 0.03
        params['prob_an_sepsis_ptb'] = 0.15
        params['prob_an_eclampsia_sl'] = 0.10
        params['prob_an_sepsis_sl'] = 0.15
        params['prob_an_aph_sl'] = 0.05
        params['prob_an_eclampsia_pl_ol'] = 0.13
        params['prob_an_sepsis_pl_ol'] = 0.25
        params['prob_an_aph_pl_ol'] = 0.25
        params['prob_an_ur_pl_ol'] = 0.06
        params['cfr_aph'] = 0.05
        params['cfr_eclampsia'] = 0.03
        params['cfr_sepsis'] = 0.02
        params['cfr_uterine_rupture'] = 0.045
        params['prob_still_birth_aph'] = 0.35
        params['prob_still_birth_sepsis'] = 0.25
        params['prob_still_birth_ur'] = 0.75
        params['prob_still_birth_eclampsia'] = 0.55
        params['prob_pn_eclampsia_ptb'] = 0.01
        params['prob_pn_pph_ptb'] = 0.01
        params['prob_pn_sepsis_ptb'] = 0.05
        params['prob_pn_eclampsia_sl'] = 0.05
        params['prob_pn_sepsis_sl'] = 0.02
        params['prob_pn_pph_sl'] = 0.05
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
        df['la_abortion'] = pd.NaT
        df['la_still_birth_this_delivery'] = False
        df['la_partiy'] = 0
        df['la_previous_cs'] = 0
        df['la_previous_ptb'] = False
        df['la_gestation'] = 0
        df['la_due_date'] = pd.NaT

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

        # Assigns a date of last pregnancy and due date for women who are pregnant at baseline

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

        #  No one at baseline can be "over due"

#  ----------------------------ASSIGNING PARITY AT BASELINE (DUMMY)-----------------------------------------------------

#       Will eventually pull in from the DHS (normal distribution?)
#       Currently roughly weighted to account for age - this is much too long

        women_parity_1524_idx = df.index[(df.age_years >= 15) & (df.age_years <= 24) & (df.is_alive == True)
                                    & (df.sex == 'F')]

        baseline_p = pd.Series(0, index=women_parity_1524_idx)

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
        df.at[child_id, 'la_abortion'] = pd.NaT
        df.at[child_id, 'la_still_birth_this_delivery'] = False
        df.at[child_id, 'la_parity'] = 0
        df.at[child_id, 'la_previous_cs'] = 0
        df.at[child_id, 'la_previous_ptb'] = False


class LabourEvent(Event, IndividualScopeEventMixin):

    """Moves a pregnant woman into labour/spontaneous abortion based on gestation distribution """

    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

# ------------------------------------LABOUR STATE---------------------------------------------------------------------

        gestation_date = df.at[individual_id, 'due_date'] - (df.at[individual_id, 'date_of_last_pregnancy'])
        gestation_weeks = gestation_date / np.timedelta64(1, 'W')

        if gestation_weeks > 37:
            random_draw = (self.sim.rng.random_sample(size=1))
            if random_draw < params['prob_pl_ol']:
                df.at[individual_id, 'la_labour'] = "prolonged_or_obstructed_labour"
            else:
                df.at[individual_id, 'la_labour'] = 'spontaneous_labour'

            # To Do: Apply the effect of risk factors
            # and of having a planned CS or Induction

        elif gestation_weeks < 37 and gestation_weeks >28:
            df.at[individual_id, 'la_labour'] = "preterm_labour"
            df.at[individual_id, 'la_previous_ptb'] = True  # or should this be following live birth?

        else:
            df.at[individual_id, 'la_labour'] = "not_in_labour"
            df.at[individual_id, 'is_pregnant'] = False
            df.at[individual_id, 'la_abortion'] = self.sim.date

            # need to consider the benefits of a "previous spont miscarriage" property
            # also need to incorporate induction and planned CS

    #  Complications During Delivery: (HOW DOES ORDER EFFECT THIS)

#  ----------------------------Uterine Rupture--------------------------------------------------------------------

    #   Need to consider women in SL who have had previous CS section/uterine scar?

        if df.at[individual_id, 'la_labour'] == 'prolonged_or_obstructed_labour' and self.sim.rng.random_sample() < \
            params['prob_an_ur_pl_ol']:   # To Do: need to think about how to capture UR (fertility)
                random_draw = (self.sim.rng.random_sample(size=1))
                if random_draw < params['cfr_uterine_rupture']:  # Women dies based on CFR of UR
                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                          cause='uterine rupture'),
                                            self.sim.date)
                else:  # here deal with women who survive but still have UR
                    if self.sim.rng.random_sample() < params['prob_still_birth_ur']:
                       df.at[individual_id,'la_still_birth_this_delivery'] = True

                #  If the woman does have a still birth because of this complication- the child is still generated
                #  in demography, if the above condition is met then instanenous death is sheduled with a cause of
                #  death being still birth


#  ----------------------------------Eclampsia--------------------------------------------------------------------------

        # Will we just use baseline incidence of eclampsia, ignoring pre eclampisa as a risk factor

        if df.at[individual_id, 'la_labour'] == 'spontaneous_unobstructed_labour' and self.sim.rng.random_sample() < \
            params['prob_an_eclampsia_sl']:
            random_draw = (self.sim.rng.random_sample(size=1))
            if random_draw < params['cfr_eclampsia']:  # Women dies based on CFR of eclampsia in labour
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='eclampsia'),
                                        self.sim.date)
            else:  # here deal with women who survive but still have experienced eclampsia in labour
                if self.sim.rng.random_sample() < params['prob_still_birth_eclampsia']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

        if df.at[individual_id, 'la_labour'] == 'prolonged_or_obstructed_labour' and self.sim.rng.random_sample() < \
            params['prob_an_eclampsia_pl_ol']:
            random_draw = (self.sim.rng.random_sample(size=1))
            if random_draw < params['cfr_eclampsia']:  # Women dies based on CFR of eclampsia in labour
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='eclampsia'),
                                        self.sim.date)
            else:  # here deal with women who survive but still have experienced eclampsia in labour
                if self.sim.rng.random_sample() < params['prob_still_birth_eclampsia']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True

        if df.at[individual_id, 'la_labour'] == 'pretterm_labour' and self.sim.rng.random_sample() < \
            params['prob_an_eclampsia_ptb']:
            random_draw = (self.sim.rng.random_sample(size=1))
            if random_draw < params['cfr_eclampsia']:  # Women dies based on CFR of eclampsia in labour
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='eclampsia'),
                                        self.sim.date)
            else:  # here deal with women who survive but still have experienced eclampsia in labour
                if self.sim.rng.random_sample() < params['prob_still_birth_eclampsia']:
                    df.at[individual_id, 'la_still_birth_this_delivery'] = True


#  ----------------------------Sepsis--------------------------------------------------------------------------
#  ----------------------------Antepartum Haemorrhage -----------------------------------------------------------

    # Postpartum complications

#  ----------------------------Postpartum Haemorrhage Rupture-----------------------------------------------------
#  ----------------------------Eclampsia--------------------------------------------------------------------------
#  ----------------------------Sepsis--------------------------------------------------------------------------

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

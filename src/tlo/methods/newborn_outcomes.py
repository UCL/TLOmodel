import logging

import pandas as pd

import numpy as np

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent

from tlo.methods import demography, eclampsia_treatment, sepsis_treatment, labour, newborn_outcomes


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NewbornOutcomes(Module):
    """
   This module is responsible for the outcomes of newborns immediately following delivery
    """

    PARAMETERS = {
        'prob_cba': Parameter(
            Types.REAL, 'baseline probability of a child being born with a congenital anomaly'),
        'prob_neonatal_sepsis': Parameter(
            Types.REAL, 'baseline probability of a child developing sepsis following birth'),
        'prob_neonatal_enceph': Parameter(
            Types.REAL, 'baseline probability of a child developing neonatal encephalopathy following delivery'),
        'prob_ptb_comps': Parameter(
            Types.REAL, 'baseline probability of a child developing complications associated with pre-term birth'),
        'cfr_cba': Parameter(
            Types.REAL, 'case fatality rate for a newborn with a congenital birth anomaly'),
        'cfr_neonatal_sepsis': Parameter(
            Types.REAL, 'case fatality rate for a newborn due to neonatal sepsis'),
        'cfr_neonatal_enceph': Parameter(
            Types.REAL, 'case fatality rate for a newborn due to neonatal encephalopathy'),
        'cfr_ptb_comps': Parameter(
            Types.REAL, 'case fatality rate for a newborn due to preterm birth complications'),
    }

    PROPERTIES = {
        'nb_congenital_anomaly': Property(Types.BOOL, 'this child has been born with a congenital anomaly'),
        'nb_neonatal_sepsis': Property(Types.BOOL, 'this child has developed neonatal sepsis following birth'),
        'nb_neonatal_enchep': Property(Types.BOOL, 'this child has developed neonatal encephalopathy secondary to '
                                                   'intrapartum related complications'),
        'nb_ptb_comps': Property(Types.BOOL, 'this child has developed complications associated with pre-term birth')

    }

    def read_parameters(self, data_folder):

        params = self.parameters

        params['prob_cba'] = 0.1  # DUMMY
        params['prob_neonatal_sepsis'] = 0.15  # DUMMY
        params['prob_neonatal_enceph'] = 0.16  # DUMMY
        params['prob_ptb_comps'] = 0.15  # DUMMY
        params['cfr_cba'] = 0.2  # DUMMY
        params['cfr_neonatal_sepsis'] = 0.1  # DUMMY
        params['cfr_neonatal_enceph'] = 0.15  # DUMMY
        params['cfr_ptb_comps'] = 0.3  # DUMMY

    def initialise_population(self, population):

        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self
        rng = m.rng

        df['nb_congenital_anomaly'] = False
        df['nb_neonatal_sepsis'] = False
        df['nb_neonatal_enchep'] = False
        df['nb_ptb_comps'] = False

    def initialise_simulation(self, sim):

        event = NewbornOutcomesLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(days=0))

    def on_birth(self, mother_id, child_id):

        df = self.sim.population.props

        df.at[child_id, 'nb_congenital_anomaly'] = False
        df.at[child_id, 'nb_neonatal_sepsis'] = False
        df.at[child_id, 'nb_neonatal_enchep'] = False
        df.at[child_id, 'nb_ptb_comps'] = False

        if df.at[child_id, 'is_alive']:
            self.sim.schedule_event(newborn_outcomes.NewbornOutcomeEvent(self.sim.modules['NewbornOutcomes'], child_id,
                                                            cause='newborn outcomes event'), self.sim.date)


class NewbornOutcomeEvent(Event, IndividualScopeEventMixin):

    """ This event determines if , following delivery, a newborn has experienced any complications """
    def __init__(self, module, individual_id, cause):
        super().__init__(module, person_id=individual_id)

    def apply(self, population):
        df = self.sim.population.props
        params = self.module.parameters
        m = self

        # Get and hold all newborns who are born that day
        newborns = df.index[(df.date_of_birth == self.sim.date) & df.is_alive]

        # Create a series containing probability of complications for this index of newborns
        eff_prob_cba = pd.Series(params['prob_cba'], index=newborns)
        eff_prob_sepsis = pd.Series(params['prob_neonatal_sepsis'], index=newborns)
        eff_prob_enceph = pd.Series(params['prob_neonatal_enceph'], index=newborns)
        eff_prob_ptb_comp = pd.Series(params['prob_ptb_comps'], index=newborns)

        # Todo: Apply risk factors that will influence the probability of newborns experiencing these complications

        random_draw = pd.Series(self.sim.rng.random_sample(size=len(newborns)),
                                index=df.index[(df.date_of_birth == self.sim.date) & df.is_alive])
        random_draw2 = pd.Series(self.sim.rng.random_sample(size=len(newborns)),
                                index=df.index[(df.date_of_birth == self.sim.date) & df.is_alive])
        random_draw3 = pd.Series(self.sim.rng.random_sample(size=len(newborns)),
                                index=df.index[(df.date_of_birth == self.sim.date) & df.is_alive])
        random_draw4 = pd.Series(self.sim.rng.random_sample(size=len(newborns)),
                                index=df.index[(df.date_of_birth == self.sim.date) & df.is_alive])

        # Create a data frame comparing likelihood of complication with a random draw
        dfx = pd.concat([eff_prob_cba, random_draw, eff_prob_sepsis, random_draw2, eff_prob_enceph, random_draw3,
                         eff_prob_ptb_comp, random_draw4], axis=1)

        dfx.columns = ['eff_prob_cba', 'random_draw','eff_prob_sepsis', 'random_draw2', 'eff_prob_enceph',
                       'random_draw3', 'eff_prob_ptb_comp','random_draw4']

        # Base on the results of the random draw a newborn will experience complications
        idx_cba = dfx.index[dfx.eff_prob_cba > dfx.random_draw]
        idx_sepsis = dfx.index[dfx.eff_prob_sepsis > dfx.random_draw2]
        idx_enceph = dfx.index[dfx.eff_prob_enceph > dfx.random_draw3]
        idx_ptb_comp = dfx.index[dfx.eff_prob_ptb_comp > dfx.random_draw4]

        df.loc[idx_cba, 'nb_congenital_anomaly'] = True
        df.loc[idx_sepsis, 'nb_neonatal_sepsis'] = True
        df.loc[idx_enceph, 'nb_neonatal_enchep'] = True
        df.loc[idx_ptb_comp, 'nb_ptb_comps'] = True

        # Apply a case fatality rate to those newborns who experience a complication and schedule the death
        # todo: decide if this is the best way to organise newborn deaths and not in another event

        for individual_id in idx_cba:
            random = self.sim.rng.random_sample()
            if random > params['cfr_cba']:
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='congenital birth anomoly'), self.sim.date)
        for individual_id in idx_sepsis:
            random = self.sim.rng.random_sample()
            if random > params['cfr_neonatal_sepsis']:
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='neonatal sepsis'), self.sim.date)
        for individual_id in idx_enceph:
            random = self.sim.rng.random_sample()
            if random > params['cfr_neonatal_enceph']:
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='neonatal encephalopathy'), self.sim.date)
        for individual_id in idx_ptb_comp:
            random = self.sim.rng.random_sample()
            if random > params['cfr_ptb_comps']:
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, individual_id,
                                                                      cause='preterm birth complications'),
                                        self.sim.date)


class NewbornOutcomesLoggingEvent(RegularEvent, PopulationScopeEventMixin):
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

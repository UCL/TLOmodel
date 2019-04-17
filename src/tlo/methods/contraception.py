"""
Contraception module covering baseline contraception methods use, failure (pregnancy),
Switching contraceptive methods, and discontiuation rates by age
please see Dropbox/Thanzi la Onse/05 - Resources/Programming Notes/Contraception-Pregnancy.pdf
for conceptual diagram
"""
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, IndividualScopeEventMixin, RegularEvent
from tlo.methods.demography import Demography

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Contraception(Module):
    """
    Contraception module covering baseline contraception methods use, failure (pregnancy),
    Switching contraceptive methods, and discontiuation rates by age
    please see Dropbox/Thanzi la Onse/05 - Resources/Programming Notes/Contraception-Pregnancy.pdf
    for conceptual diagram (lucid chart)
    """

    def __init__(self, name=None, workbook_path=None):
        super().__init__(name)
        self.workbook_path = workbook_path

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'fertility_schedule': Parameter(Types.DATA_FRAME, 'Age_spec fertility'),
        # TODO: fertlity schedule (baseline fertlity) is currently in demography should it only be here in contraception?
        'contraception_initiation1': Parameter(Types.DATA_FRAME, 'irate1_'), # 'irate_1_' sheet created manually as a work around to address to do point below
        # TODO: irate1 is specified incorrectly as expos is too low for each method - should be total exposure time and the same for each method i.e. exposure should be those exposed to 'not_using'. The irate1 sheet also needs to be transposed to have one row for monthly, quarterly, year rate and columns for each method as in the Age_spec fertility rate, and column names spelt exactly the same as the contraception categories in the module
        'contraception_initiation2': Parameter(Types.DATA_FRAME, 'irate2'),
        'contraception_discontinuation': Parameter(Types.DATA_FRAME, 'Discontinuation')
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'contraception': Property(Types.CATEGORICAL, 'Current contraceptive method',
                                  categories=['not_using', 'pill', 'IUD', 'injections', 'implant', 'male_condom',
                                              'female_sterilization', 'other_modern', 'periodic_abstinence',
                                              'withdrawal', 'other_traditional'])
        # These are the 11 categories of contraception ('not using' + 10 methods) from the DHS analysis of initiation,
        # discontinuation, failure and switching rates
        # 'other modern' includes Male sterilization, Female Condom, Emergency contraception
        # 'other traditional' includes lactational amenohroea (LAM), standard days method (SDM), 'other traditional method'
        # Have replaced Age-spec fertility sheet in demography.xlsx (in this branch) with the one in contraception.xlsx
        # (has 11 categories and one row for each age with baseline contraceptopn prevalences for each of the 11 categories)
    }

    def read_parameters(self, data_folder):
        """
        Please see Contraception-Pregnancy.pdf for lucid chart explaining the relationships between
        baseline fertility rate, intitiation rates, discontinuation, failure and switching rates, and being on
        contraception or not, and being pregnant
        """
        workbook = pd.read_excel(self.workbook_path, sheet_name=None)
        self.parameters['fertility_schedule'] = workbook['Age_spec fertility']
        self.parameters['contraception_initiation1'] = workbook['irate1_']  # 'irate_1_' sheet created manually as a work around to address to do point on line 39
        # this Excel sheet is irate1_all.csv output from 'initiation rates_age_stcox.do' Stata analysis of DHS contraception calendar data
        self.parameters['contraception_initiation2'] = workbook['irate2']
        # this Excel sheet is irate2_all.csv output from 'initiation rates_age_stcox.do' Stata analysis of DHS contraception calendar data
        self.parameters['contraception_discontinuation'] = workbook['Discontinuation']
        # this Excel sheet is from contraception_failure_discontinuation_switching.csv output from 'discontinuation & switching rates_age.do' Stata analysis of DHS contraception calendar data

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.
        """
        df = population.props
        fertility = self.parameters['fertility_schedule']

        # Assign contraception method
        # 1. select females aged 15-49 from population, for current year
        possibly_using = df.is_alive & (df.sex == 'F') & df.age_years.between(15, 49)
        females1549 = df.index[possibly_using]
        # 2. Get probabilities of using each method of contraception by age
        #       and merge the probabilities into each row in sim population
        df_new = df.merge(fertility, left_on=['age_years'], right_on=['age'], how='left')
        probabilities = df_new.loc[females1549, ['not_using', 'pill', 'IUD', 'injections', 'implant', 'male_condom',
                                              'female_sterilization', 'other_modern', 'periodic_abstinence',
                                              'withdrawal', 'other_traditional']]

        # 3. apply probabilities of each contraception type to sim population
        for woman in probabilities.index:
            her_p=np.asarray(probabilities.loc[woman,:])
            her_op=np.asarray(probabilities.columns)

            her_method=self.rng.choice(her_op,p=her_p/her_p.sum())  # /her_p.sum() added becasue probs sometimes add to not quite 1 due to rounding

            df.loc[woman,'contraception']=her_method


        # TODO: need to do it without above for loop:
        # probabilities['p_list'] = probabilities.apply(lambda row: row[:].tolist(), axis=1)  # doesn't work as p_list is dtype['O'] (object) rather than float64
        #categories = ['not_using', 'pill', 'IUD', 'injections', 'implant', 'male_condom', 'female_sterilization',
        #              'other_modern', 'periodic_abstinence', 'withdrawal', 'other_traditional']
        #random_choice = self.rng.choice(categories, size=len(df), p=probabilities['p_list'])
        #df.loc[females1549, 'contraception'].values[:] = random_choice


    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        # check all population to determine if contraception starts i.e. category should change from 'not_using' (repeats every month)
        sim.schedule_event(Init1(self), sim.date + DateOffset(months=1))

        # check all population to determine if contraception discontinues i.e. category should change to 'not_using' (repeats every month)
        sim.schedule_event(Discontinue(self), sim.date + DateOffset(months=1))

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        df = self.sim.population.props

         # Log the birth:
        logger.info('%s|on_birth|%s',
                    self.sim.date,
                    {
                        'mother': mother_id,
                        'child': child_id,
                        'mother_age': df.at[mother_id, 'age_years'],
                        'xxx': 0
                    })


class Init1(RegularEvent, PopulationScopeEventMixin):
    """
    This event looks across all women who are 'not_using' contraception to determine if they start using each
    method according to initiation_rate1 (irate_1)
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):
        logger.debug('Checking to see if anyone should start using contraception')

        df = population.props  # get the population dataframe
        m = self.module
        rng = self.module.rng

        # get the indices of the women from the population with the relevant characterisitcs
        not_using_idx = df.index[(df.sex == 'F') &
                                 df.is_alive &
                                 df.age_years.between(self.age_low, self.age_high) &
                                 ~df.is_pregnant &
                                 (df.contraception == 'not_using')]

        # prepare the probabilities
        c_worksheet = m.parameters['contraception_initiation1']
        c_probs = c_worksheet.loc[0].values.tolist()
        c_names = c_worksheet.columns.tolist()

        # sample contraceptive method for everyone not using
        # and put in a series which has index of all currently not using
        sampled_method = pd.Series(rng.choice(c_names, p=c_probs, size=len(not_using_idx), replace=True),
                                   index=not_using_idx)

        # only update those starting on contraception
        now_using_idx = not_using_idx[sampled_method != 'not_using']
        df.loc[now_using_idx, 'contraception'] = sampled_method[now_using_idx]

        # output some logging if any start contraception
        if len(now_using_idx):
            for woman_id in now_using_idx:
                logger.info('%s|start_contraception|%s',
                            self.sim.date,
                            {
                                'woman_age': df.at[woman_id, 'age_years'],
                                'contraception': df.at[woman_id, 'contraception']
                            })


class Discontinue(RegularEvent, PopulationScopeEventMixin):
    """
    This event looks across all women who are using a contraception method to determine if they stop using it
    according to discontinuation_rate
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        self.age_low = 15
        self.age_high = 49

    def apply(self, population):
        logger.debug('Checking to see if anyone should stop using contraception')

        df = population.props  # get the population dataframe
        m = self.module
        rng = self.module.rng

        # get the indices of the women from the population with the relevant characterisitcs
        using_idx = df.index[(df.contraception != 'not_using')]

        # prepare the probabilities
        c_worksheet = m.parameters['contraception_discontinuation']

        # add the probabilities and copy to each row of the sim population (population dataframe)
        df_new = pd.concat([df, c_worksheet], axis=1)
        df_new.loc[:, ['pill', 'IUD', 'injections', 'implant', 'male_condom', 'female_sterilization', 'other_modern',
                   'periodic_abstinence', 'withdrawal', 'other_traditional']] = df_new.loc[0, ['pill', 'IUD', 'injections',
                                                                                           'implant', 'male_condom',
                                                                                           'female_sterilization',
                                                                                           'other_modern',
                                                                                           'periodic_abstinence',
                                                                                           'withdrawal',
                                                                                           'other_traditional']].tolist()
        probabilities = df_new.loc[using_idx, ['contraception','pill', 'IUD', 'injections', 'implant', 'male_condom',
                                                 'female_sterilization', 'other_modern', 'periodic_abstinence',
                                                 'withdrawal', 'other_traditional']]
        probabilities['prob'] = probabilities.lookup(probabilities.index, probabilities['contraception'])
        probabilities['1-prob'] = 1-probabilities['prob']
        probabilities['not_using'] = 'not_using'

        # apply the probabilities of discontinuation for each contraception method
        # to series which has index of all currently using
        # need to use a for loop to loop through each method
        for woman in using_idx:
            her_p = np.asarray(probabilities.loc[woman,['prob','1-prob']], dtype='float64')
            her_op = np.asarray(probabilities.loc[woman,['not_using', 'contraception']])

            her_method = rng.choice(her_op,p=her_p)

            df.loc[woman,'contraception']=her_method
            # output some logging if any stop contraception
            if her_method == 'not_using':
                logger.info('%s|stop_contraception|%s',
                                self.sim.date,
                                {
                                    'woman_age': df.at[woman, 'age_years'],
                                    'contraception': df.at[woman, 'contraception']
                                })


class ContraceptionEvent(RegularEvent, PopulationScopeEventMixin):
    """A skeleton class for an event

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
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """
        raise NotImplementedError
